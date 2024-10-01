import xarray as xr
import numpy as np
from dask.base import is_dask_collection
import jax.numpy as jnp
from scipy.sparse import coo_matrix, csr_matrix
from numba import njit, int64, prange

class Tracker:
        
    def __init__(self, da, scratch_dir, radius, resolution, min_size_quartile, timedim, xdim, neighbours, land_mask):
        
        self.da = da
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.land_mask = land_mask
        self.radius = radius
        self.resolution = resolution
        self.scratch_dir = scratch_dir
        
        if ((timedim, xdim) != da.dims):
            try:
                da = da.transpose(timedim, xdim) 
            except:
                raise ValueError(f'Ocetrac-unstruct currently only supports 2D DataArrays. The dimensions should only contain ({timedim} and {xdim}). Found {list(da.dims)}')

        if not is_dask_collection(da.data):
            raise ValueError('The input DataArray is not backed by a Dask array. Please chunk (in time), and try again.  :)')
        
        if not da.data.dtype == bool:
            raise ValueError('The input DataArray must have a dtype of bool.')
        
        
        ## Initialise the dilation array
        
        self.neighbours_int = neighbours.astype(np.int32).compute().data - 1 # Convert to 0-based indexing (negative values will be dropped)
        if self.neighbours_int.shape[0] != 3:
            raise ValueError('The neighbours array must have a shape of (3, ncells).')
        
        self.n_connections = int(radius / self.resolution) # Number connections for effective structure element
        
        ## Construct the sparse dilation matrix
        
        # Create row indices (i) and column indices (j) for the sparse matrix
        row_indices = jnp.repeat(jnp.arange(self.neighbours_int.shape[1]), 3)
        col_indices = self.neighbours_int.T.flatten()

        # Filter out negative values
        valid_mask = col_indices >= 0
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]

        dilate_coo = coo_matrix((jnp.ones_like(row_indices, dtype=bool), (row_indices, col_indices)), shape=(self.neighbours_int.shape[1], self.neighbours_int.max() + 1))
        self.dilate_sparse = csr_matrix(dilate_coo) #**4
        
        
    def track(self):
        '''
        Label and track image features.
        
        Parameters
        ----------
        da : xarray.DataArray
            The data to label. Must represent an underlying dask array of type bool.

        radius : float
            The size of the structuring element used in morphological opening and closing. Radius specified in the same units as resolution
        
        resolution : float
            The (effective) unstructured grid resolution. 

        min_size_quartile : float
            The quantile used to define the threshold of the smallest area object retained in tracking. Value should be between 0 and 1.

        timedim : str
            The name of the time dimension
        
        xdim : str
            The name of the ncells dimension
        
        land_mask : bool
            A boolean mask of the land cells. True for land, False for ocean.

        Returns
        -------
        labels : xarray.DataArray
            Integer labels of the connected regions.
        '''

        # Convert data to binary, define structuring element, and perform morphological closing then opening
        self._morphological_operations()  # NB: Scipy+Dask memory leak issues requires these 2 intermediary saves to clear memory...
        binary_images = xr.open_zarr(self.scratch_dir+'/02_binary_images_temp.zarr', chunks={}).binary_images

        # Filter area to remove small objects
        areas, min_area, binary_images_filtered, N_initial = self._filter_area(binary_images)

        # Label objects -- connectivity now in time !
        labelled_global_time, N_final = self._label_unstruct_time(binary_images_filtered)

        final_labels = xr.where(labelled_global_time<0, np.nan, labelled_global_time)  # Set -1 labels to NaN


        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(areas.sum().item())

        reject_area = np.where(areas <= min_area, areas, 0) 
        sum_reject_area = int(reject_area.sum().item())
        percent_area_reject = (sum_reject_area/sum_tot_area)

        accept_area = np.where(areas > min_area, areas, 0) 
        sum_accept_area = int(accept_area.sum().item())
        percent_area_accept = (sum_accept_area/sum_tot_area)
        
        effective_radius_ncells = int(self.n_connections / 4) * 4

        final_labels = final_labels.rename('labels')
        final_labels.attrs['Inital Objects Identified'] = int(N_initial)
        final_labels.attrs['Final Objects Tracked'] = int(N_final)
        final_labels.attrs['Closing Radius'] = self.radius
        final_labels.attrs['Resolution'] = self.resolution
        final_labels.attrs['Effective Closing Radius (N)'] = effective_radius_ncells
        final_labels.attrs['Size Quantile Threshold'] = self.min_size_quartile
        final_labels.attrs['Minimum Size (N)'] = min_area
        final_labels.attrs['Percent Area Reject'] = percent_area_reject
        final_labels.attrs['Percent Area Accept'] = percent_area_accept

        print('Inital Time-Independent Objects Identified : \t', int(N_initial))
        print('Final Objects Tracked through Time :\t', int(N_final))

        return final_labels


    ### PRIVATE METHODS - not meant to be called by user ###
    

    def _morphological_operations(self): 
        '''Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        '''
        
        exponent = int(self.n_connections)
        land_mask = self.land_mask
        data = self.dilate_sparse.data
        indices = self.dilate_sparse.indices
        indptr = self.dilate_sparse.indptr

        def binary_open_close(bitmap_binary):
            
            bitmap_binary_opened = sparse_bool_power(bitmap_binary, data, indices, indptr, exponent)  # This sparse_bool_power assumes the xdim (multiplying the sparse matrix) is in dim=1
            
            # Set the land values to True (to avoid artificially eroding the shore)
            bitmap_binary_opened[:, land_mask] = True
            
            ## Opening is just the negated closing of the negated image
            bitmap_binary_closed = ~sparse_bool_power(~bitmap_binary_opened, data, indices, indptr, exponent)
            
            return bitmap_binary_closed
        

        mo_binary = xr.apply_ufunc(binary_open_close, self.da,
                                    input_core_dims=[[self.xdim]],
                                    output_core_dims=[[self.xdim]],
                                    output_dtypes=[np.bool_],
                                    vectorize=False,
                                    dask='parallelized')
        
        mo_binary.name = 'binary_images'
        mo_binary.to_zarr(self.scratch_dir+'/02_binary_images_temp.zarr', mode='w')
        return True

    
    def _filter_area(self, binary_images):
        '''Calculate area with regionprops and remove small objects'''
        
        ## Unstructured Union-Find (Disjoint Set Union) Clustering Algorithm
        self._label_union_find_unstruct(binary_images, '/02_cluster_labels_1_temp.zarr')
        # NB: Scipy+Dask memory leak issues requires these 2 intermediary saves to clear memory...
        cluster_labels = xr.open_zarr(self.scratch_dir+'/02_cluster_labels_1_temp.zarr', chunks={}).cluster_labels
        
        max_label = cluster_labels.max().compute().data+1
        
        
        ## Calculate areas: 
        
        def count_cluster_sizes(cluster_labels):
            unique, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
            padded_sizes = np.zeros(max_label, dtype=np.int64)
            padded_sizes[:len(counts)] = counts
            return padded_sizes  # ith element corresponds to label=i
        
        cluster_sizes = xr.apply_ufunc(count_cluster_sizes, 
                                cluster_labels, 
                                input_core_dims=[[self.xdim]],
                                output_core_dims=[['labels']],
                                output_sizes={'labels': max_label}, 
                                output_dtypes=[np.int64],
                                vectorize=True,
                                dask='parallelized')
        
        cluster_sizes = cluster_sizes.compute()
        
        areas = cluster_sizes.values.flatten()
        areas = areas[areas>1]


        ## Filter small areas: 
        
        N_initial = len(areas)
        
        min_area = np.percentile(areas, self.min_size_quartile*100)
        print(f'{self.min_size_quartile*100}th Area Percentile gives the Minimum Number of Cells = {min_area}') 
        
        def filter_area_binary(cluster_labels_0, keep_labels_0):
            keep_labels_0 = keep_labels_0[keep_labels_0>=0]
            keep_where = np.isin(cluster_labels_0, keep_labels_0)
            return keep_where
        
        keep_labels = xr.where(cluster_sizes>=min_area, cluster_sizes.labels, -10)
        
        binary_images_filtered = xr.apply_ufunc(filter_area_binary, 
                                cluster_labels, keep_labels, 
                                input_core_dims=[[self.xdim],['labels']],
                                output_core_dims=[[self.xdim]],
                                output_dtypes=[binary_images.dtype],
                                vectorize=True,
                                dask='parallelized')
        
        binary_images_filtered = binary_images_filtered #.persist()
        

        return areas, min_area, binary_images_filtered, N_initial
    
    
        
    def _label_unstruct_time(self, binary_images_filtered):
        '''Label all object, with connectivity in time.
           This is much less trivial to do in parallel and OOC memory......
        '''
        
        ## Step 1: Label all time slices -- parallel in time. Then make each label globally unique in time.
        
        self._label_union_find_unstruct(binary_images_filtered, '/02_cluster_labels_2_temp.zarr')
        # NB: Scipy+Dask memory leak issues requires these 2 intermediary saves to clear memory...
        cluster_labels = xr.open_zarr(self.scratch_dir+'/02_cluster_labels_2_temp.zarr', chunks={}).cluster_labels

        cumsum_labels = (cluster_labels.max(dim=self.xdim) + 1).cumsum(self.timedim)
        
        min_int64 = np.iinfo(np.int64).min
        cluster_labels_nan = xr.where(cluster_labels<0, min_int64, cluster_labels)  # Otherwise the -1 labels get added to !!!
        cluster_labels_unique = cluster_labels_nan + cumsum_labels
        cluster_labels_unique = cluster_labels_unique.persist()
        
        
        ## Step 2: Go again (parallel in time) but now checking just for overlap with previous and next time slice.
        #           Keep a running list of all equivalent labels.

        def check_overlap(labels_t, labels_prev, labels_next):
            
            overlap_prev = np.logical_and(labels_t > 0, labels_prev > 0)
            overlap_next = np.logical_and(labels_t > 0, labels_next > 0)
            
            pairs_prev = np.stack((labels_t[overlap_prev], labels_prev[overlap_prev]), axis=1)
            pairs_next = np.stack((labels_t[overlap_next], labels_next[overlap_next]), axis=1)
            
            pairs_all = np.concatenate((pairs_prev, pairs_next), axis=0)
            pairs_all = np.sort(pairs_all, axis=1)
            pairs_unique = np.unique(pairs_all, axis=0)
            
            return pairs_unique.astype(np.int64)
        
        overlap_pairs = xr.apply_ufunc(
                            check_overlap,
                            cluster_labels_unique,
                            cluster_labels_unique.shift(time=1, fill_value=-10),
                            cluster_labels_unique.shift(time=-1, fill_value=-10),
                            input_core_dims=[[self.xdim], [self.xdim], [self.xdim]],
                            output_core_dims=[[]],
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[object]
                        )
        
        overlap_pairs = overlap_pairs.compute()
        overlap_pairs_all = np.concatenate(overlap_pairs.values)
        
        
        ## Step 3: Cluster the overlap_pairs into groups of equivalent labels.
        
        def cluster_time_pairs(overlap_pairs_all):
            # Get unique labels from the overlap pairs
            labels = np.unique(overlap_pairs_all)
            
            # Initialize disjoint set data structure
            n = len(labels)
            parent = np.arange(n)
            rank = np.zeros(n, dtype=int)
            
            # Map labels to indices in the disjoint set
            label_to_index = {label: index for index, label in enumerate(labels)}
            
            # Perform union-find on the overlap pairs
            for pair in overlap_pairs_all:
                x = label_to_index[pair[0]]
                y = label_to_index[pair[1]]
                union(parent, rank, x, y)
            
            # Find the root label for each label
            root_labels = [find(parent, label_to_index[label]) for label in labels]
            
            # Group labels by their root label
            clusters = {}
            for label, root in zip(labels, root_labels):
                root_label = labels[root]
                if root_label not in clusters:
                    clusters[root_label] = []
                clusters[root_label].append(label)
            
            return list(clusters.values())
        
        equivalent_labels = cluster_time_pairs(overlap_pairs_all)
        
        
        ## Step 4: Replace all labels in cluster_labels_unique that match the equivalent_labels with the list index:  This is the new/final label.
        
        # Create a dictionary to map labels to the new cluster indices
        min_int32 = np.iinfo(np.int32).min
        label_to_cluster_index_array = np.full(cluster_labels_unique.max().compute().data + 1, min_int32, dtype=np.int32)
        for index, cluster in enumerate(equivalent_labels):
            for label in cluster:
                label_to_cluster_index_array[label] = np.int32(index) # Because these are the connected labels, there are many fewer!
        
        # Create a vectorized function to replace labels with cluster indices
        relabeled_unique = xr.apply_ufunc(
            lambda x: np.where(x < 0, min_int32, label_to_cluster_index_array[np.maximum(0,x)]),
            cluster_labels_unique,
            input_core_dims=[[self.xdim]],
            output_core_dims=[[self.xdim]],
            dask="parallelized",
            output_dtypes=[np.int32]
        )
                
        relabeled_unique = relabeled_unique #.persist()
        
        N_final = len(equivalent_labels)
        
        return relabeled_unique, N_final

    
    
    def _label_union_find_unstruct(self, binary_images, temp_filename):
        '''Label all object, with no connectivity in time.
           Utilise highly efficient Unstructured Union-Find (Disjoint Set Union) Clustering Algorithm
        '''
        
        neighbours_int = self.neighbours_int

        
        def cluster_true_values(data_block):
            n = data_block.shape[-1]
            cluster_labels = -1 * np.ones_like(data_block, dtype=np.int64)
            parent = np.arange(n, dtype=np.int64)
            rank = np.zeros(n, dtype=np.int64)
            root_to_cluster = {}
            cluster_count = 0

            for t in range(data_block.shape[0]):
                data = data_block[t]
                parent[:] = np.arange(n, dtype=np.int64)
                rank[:] = 0
                root_to_cluster.clear()
                cluster_count = 0

                for i in data.nonzero()[0]:
                    for neighbor in neighbours_int[:, i]:
                        if neighbor != -1 and data[neighbor]:
                            root = union_find(parent, rank, i, neighbor)
                            if root not in root_to_cluster:
                                root_to_cluster[root] = cluster_count
                                cluster_count += 1
                            cluster_labels[t, i] = root_to_cluster[root]

            return cluster_labels
        
        
        ## Label time-independent in 2D (i.e. no time connectivity!)
        binary_images_mask = binary_images.where(~self.land_mask, other=True).persist()  # Mask land
        
        cluster_labels = xr.apply_ufunc(cluster_true_values, 
                                binary_images_mask, 
                                input_core_dims=[[self.xdim]],
                                output_core_dims=[[self.xdim]],
                                output_dtypes=[np.int64],
                                vectorize=False,
                                dask='parallelized')
        
        # label = -1 if False
        cluster_labels.name = 'cluster_labels'
        cluster_labels.to_zarr(self.scratch_dir+temp_filename, mode='w') 
        
        return True


## Helper Functions for Unstructured Union-Find (Disjoint Set Union) Clustering
@njit(fastmath=True, parallel=True)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

@njit(fastmath=True, parallel=True)
def union(parent, rank, x, y):
    root_x, root_y = find(parent, x), find(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1


@njit(int64(int64[:], int64[:], int64, int64), fastmath=True)
def union_find(parent, rank, x, y):
    root_x = x
    while parent[root_x] != root_x:
        root_x = parent[root_x]
        
    root_y = y
    while parent[root_y] != root_y:
        root_y = parent[root_y]
        
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
            
    return root_x



## Helper Function for Super Fast Sparse Bool Multiply (*without the scipy+Dask Memory Leak*)
@njit(fastmath=True, parallel=True)
def sparse_bool_power(vec, sp_data, indices, indptr, exponent):
    vec = vec.T
    num_rows = indptr.size - 1
    num_cols = vec.shape[1]
    result = vec.copy()

    for _ in range(exponent):
        temp_result = np.zeros((num_rows, num_cols), dtype=np.bool_)

        for i in prange(num_rows):
            for j in range(indptr[i], indptr[i + 1]):
                if sp_data[j]:
                    for k in range(num_cols):
                        if result[indices[j], k]:
                            temp_result[i, k] = True

        result = temp_result

    return result.T
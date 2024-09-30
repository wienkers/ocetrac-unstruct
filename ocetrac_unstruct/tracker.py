import xarray as xr
import numpy as np
from dask.base import is_dask_collection
import jax.numpy as jnp
from scipy.sparse import coo_matrix, csr_matrix
from numba import njit, prange

class Tracker:
        
    def __init__(self, da, radius, resolution, min_size_quartile, timedim, xdim, neighbours, land_mask):
        
        self.da = da
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.land_mask = land_mask
        self.radius = radius
        self.resolution = resolution
        
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
        
        self.neighbours_int = neighbours.astype(int).compute().data - 1 # Convert to 0-based indexing (negative values will be dropped)
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
        binary_images = self._morphological_operations()

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
        
        return mo_binary

    
    def _filter_area(self, binary_images):
        '''Calculate area with regionprops and remove small objects'''
        
        ## Unstructured Union-Find (Disjoint Set Union) Clustering Algorithm
        cluster_labels = self._label_union_find_unstruct(binary_images).persist()
        
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
        
        cluster_labels = self._label_union_find_unstruct(binary_images_filtered) #.persist()

        cumsum_labels = (cluster_labels.max(dim=self.xdim) + 1).cumsum(self.timedim)
        
        min_int64 = np.iinfo(np.int64).min
        cluster_labels_nan = xr.where(cluster_labels<0, min_int64, cluster_labels)  # Otherwise the -1 labels get added to !!!
        cluster_labels_unique = cluster_labels_nan + cumsum_labels
        
        
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
        label_to_cluster_index = {}
        for index, cluster in enumerate(equivalent_labels):
            for label in cluster:
                label_to_cluster_index[label] = index
        
        # Create a vectorized function to replace labels with cluster indices
        def replace_labels(labels):
            return np.vectorize(label_to_cluster_index.get)(labels, min_int64)
        
        relabeled_unique = xr.apply_ufunc(
            replace_labels,
            cluster_labels_unique,
            input_core_dims=[[self.xdim]],
            output_core_dims=[[self.xdim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64]
        )
        
        relabeled_unique = relabeled_unique.persist()
        
        N_final = len(equivalent_labels)
        
        return relabeled_unique, N_final

    
    
    def _label_union_find_unstruct(self, binary_images):
        '''Label all object, with no connectivity in time.
           Utilise highly efficient Unstructured Union-Find (Disjoint Set Union) Clustering Algorithm
        '''
        
        def cluster_true_values(data):
            n = len(data)
            parent = np.arange(n)
            rank = np.zeros(n, dtype=np.int32)

            for i in range(n):
                if data[i]:
                    for j in range(3):
                        neighbor = self.neighbours_int[j, i]
                        if neighbor != -1 and data[neighbor]:
                            union(parent, rank, i, neighbor)

            cluster_labels = np.full(n, -1, dtype=np.int32)
            cluster_count = 0
            root_to_cluster = {}

            for i in range(n):
                if data[i]:
                    root = find(parent, i)
                    if root not in root_to_cluster:
                        root_to_cluster[root] = cluster_count
                        cluster_count += 1
                    cluster_labels[i] = root_to_cluster[root]

            return cluster_labels #, cluster_count
        
        
        ## Label time-independent in 2D (i.e. no time connectivity!)
        binary_images_mask = xr.where(self.land_mask, False, binary_images)  # Mask land
        
        cluster_labels = xr.apply_ufunc(cluster_true_values, 
                                binary_images_mask, 
                                input_core_dims=[[self.xdim]],
                                output_core_dims=[[self.xdim]],
                                output_dtypes=[np.int64],
                                vectorize=True,
                                dask='parallelized')
        
        cluster_labels = cluster_labels # .persist()  # label = -1 if False
        
        return cluster_labels


## Helper Functions for Unstructured Union-Find (Disjoint Set Union) Clustering
@njit(fastmath=True)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

@njit(fastmath=True)
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
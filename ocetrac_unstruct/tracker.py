import xarray as xr
import numpy as np
from dask.base import is_dask_collection
import dask.array as dsa
from dask import persist
import jax.numpy as jnp
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.csgraph import connected_components
from numba import njit, int64, int32, prange

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
        
        self.neighbours_int = neighbours.astype(np.int32) - 1 # Convert to 0-based indexing (negative values will be dropped)  ##.compute().data
        if self.neighbours_int.shape[0] != 3:
            raise ValueError('The neighbours array must have a shape of (3, ncells).')
        if self.neighbours_int.dims != ('nv', self.xdim):
            raise ValueError('The neighbours array must have dimensions of (nv, xdim).')
        
        
        self.n_connections = int(radius / self.resolution) # Number connections for effective structure element
        
        ## Construct the sparse dilation matrix
        
        # Create row indices (i) and column indices (j) for the sparse matrix
        row_indices = jnp.repeat(jnp.arange(self.neighbours_int.shape[1]), 3)
        col_indices = self.neighbours_int.data.compute().T.flatten()

        # Filter out negative values
        valid_mask = col_indices >= 0
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]
        
        max_neighbour = self.neighbours_int.max().compute().item() + 1

        dilate_coo = coo_matrix((jnp.ones_like(row_indices, dtype=bool), (row_indices, col_indices)), shape=(self.neighbours_int.shape[1], max_neighbour))
        self.dilate_sparse = csr_matrix(dilate_coo)
        
        # _Need to add identity!!!_ —- ... otherwise get checkerboard >_< 
        identity = eye(self.neighbours_int.shape[1], dtype=bool, format='csr')
        self.dilate_sparse = self.dilate_sparse + identity
        
        
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
        ###TEMPPP  self._morphological_operations()  # NB: Scipy+Dask memory leak issues requires these 2 intermediary saves to clear memory...
        binary_images = xr.open_zarr(self.scratch_dir+'/02_binary_images_temp.zarr', chunks={}).binary_images
        print('Done with Morphological Operations')
        
        # Filter area to remove small objects
        areas, min_area, N_initial = self._filter_area(binary_images)
        binary_images_filtered = xr.open_zarr(self.scratch_dir+'/02_binary_images_filtered_temp.zarr', chunks={}).binary_images_filtered
        print('Done with Area Filtering')

        # Label objects -- connectivity now in time !
        final_labels, N_final = self._label_unstruct_time(binary_images_filtered)
        print('Done with Time-Connected Labeling')

        #final_labels = xr.where(labelled_global_time<0, np.nan, labelled_global_time)  # Set -1 labels to NaN  ## Don't, because it converts to float64...

        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(areas.sum().item())

        reject_area = np.where(areas <= min_area, areas, 0) 
        sum_reject_area = int(reject_area.sum().item())
        percent_area_reject = (sum_reject_area/sum_tot_area)

        accept_area = np.where(areas > min_area, areas, 0) 
        sum_accept_area = int(accept_area.sum().item())
        percent_area_accept = (sum_accept_area/sum_tot_area)
        
        effective_radius_ncells = self.n_connections * 1.73 / 2.0 # Accounting for ~equilateral triangle zig-zag (==sin(120)/sin(30)/2)

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
        
        final_labels.to_zarr(self.scratch_dir+'/02_final_labels.zarr', mode='w')
        final_labels = xr.open_zarr(self.scratch_dir+'/02_final_labels.zarr', chunks={})

        return final_labels


    ### PRIVATE METHODS - not meant to be called by user ###
    

    def _morphological_operations(self): 
        '''Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        '''
        
        exponent = int(self.n_connections)
        land_mask = self.land_mask.data
        ## _Put the data into an xarray.DataArray to pass into the apply_ufunc_ -- Needed for correct memory management !!!
        sp_data = xr.DataArray(self.dilate_sparse.data, dims='sp_data')
        indices = xr.DataArray(self.dilate_sparse.indices, dims='indices')
        indptr = xr.DataArray(self.dilate_sparse.indptr, dims='indptr')

        def binary_open_close(bitmap_binary, sp_data, indices, indptr):
            
            bitmap_binary_opened = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, exponent)  # This sparse_bool_power assumes the xdim (multiplying the sparse matrix) is in dim=1
            
            # Set the land values to True (to avoid artificially eroding the shore)
            bitmap_binary_opened[:, land_mask] = True
            
            ## Opening is just the negated closing of the negated image
            bitmap_binary_closed = ~sparse_bool_power(~bitmap_binary_opened, sp_data, indices, indptr, exponent)
            
            return bitmap_binary_closed
        

        mo_binary = xr.apply_ufunc(binary_open_close, self.da, sp_data, indices, indptr,
                                    input_core_dims=[[self.xdim],['sp_data'],['indices'],['indptr']],
                                    output_core_dims=[[self.xdim]],
                                    output_dtypes=[np.bool_],
                                    vectorize=False,
                                    dask='parallelized')
        
        ###TEMPPP  mo_binary = xr.where(~self.land_mask, mo_binary, False)
        ###TEMPPP  mo_binary.name = 'binary_images'
        ###TEMPPP  mo_binary.to_zarr(self.scratch_dir+'/02_binary_images_temp.zarr', mode='w')
        return True

    
    def _filter_area(self, binary_images):
        '''Calculate area with regionprops and remove small objects'''
        
        ## Unstructured Union-Find (Disjoint Set Union) Clustering Algorithm
        ###TEMPPP  self._label_union_find_unstruct(binary_images, '/02_cluster_labels_1_temp.zarr')
        # NB: Scipy+Dask memory leak issues requires these 2 intermediary saves to clear memory...
        cluster_labels = xr.open_zarr(self.scratch_dir+'/02_cluster_labels_1_temp.zarr', chunks={}).cluster_labels
        print('Done with Union-Find Algorithm (1)')
        
        max_label = cluster_labels.max().compute().data+1
        
        
        ## Calculate areas: 
        
        def count_cluster_sizes(cluster_labels):
            unique, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
            padded_sizes = np.zeros(max_label, dtype=np.int32)
            padded_unique = np.zeros(max_label, dtype=np.int32)
            padded_sizes[:len(counts)] = counts
            padded_unique[:len(counts)] = unique
            return padded_sizes, padded_unique  # ith element corresponds to label=i
        
        cluster_sizes, unique_cluster_labels = xr.apply_ufunc(count_cluster_sizes, 
                                cluster_labels, 
                                input_core_dims=[[self.xdim]],
                                output_core_dims=[['labels'],['labels']],
                                output_sizes={'labels': max_label}, 
                                output_dtypes=(np.int32, np.int32),
                                vectorize=True,
                                dask='parallelized')
                
        cluster_sizes, unique_cluster_labels = persist(cluster_sizes, unique_cluster_labels)
        
        cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > 3).data
        cluster_areas_mask = dsa.isfinite(cluster_sizes_filtered_dask)
        areas = cluster_sizes_filtered_dask[cluster_areas_mask].compute()
        print('Done Calculating Areas')

        ## Filter small areas: 
        
        N_initial = len(areas)
        
        min_area = np.percentile(areas, self.min_size_quartile*100)
        print(f'{self.min_size_quartile*100}th Area Percentile gives the Minimum Number of Cells = {min_area}') 
        
        def filter_area_binary(cluster_labels_0, keep_labels_0):
            keep_labels_0 = keep_labels_0[keep_labels_0>=0]
            keep_where = np.isin(cluster_labels_0, keep_labels_0)
            return keep_where
        
        keep_labels = xr.where(cluster_sizes>=min_area, unique_cluster_labels, -10)  # unique_cluster_labels has been mapped in "count_cluster_sizes"
        
        binary_images_filtered = xr.apply_ufunc(filter_area_binary, 
                                cluster_labels, keep_labels, 
                                input_core_dims=[[self.xdim],['labels']],
                                output_core_dims=[[self.xdim]],
                                output_dtypes=[binary_images.dtype],
                                vectorize=True,
                                dask='parallelized')
        
        ###TEMPPP  binary_images_filtered = xr.where(~self.land_mask, binary_images_filtered, False)
        ###TEMPPP  binary_images_filtered.name = 'binary_images_filtered'
        ###TEMPPP  binary_images_filtered.attrs['min_area'] = min_area
        ###TEMPPP  binary_images_filtered.attrs['N_initial'] = N_initial
        ###TEMPPP  binary_images_filtered.to_zarr(self.scratch_dir+'/02_binary_images_filtered_temp.zarr', mode='w') 
        
        return areas, min_area, N_initial  # binary_images_filtered
    
    
        
    def _label_unstruct_time(self, binary_images_filtered):
        '''Label all object, with connectivity in time.
           This is much less trivial to do in parallel and OOC memory......
        '''
        
        ## Step 1: Label all time slices -- parallel in time. Then make each label globally unique in time.
        
        ###TEMPPP  self._label_union_find_unstruct(binary_images_filtered, '/02_cluster_labels_2_temp.zarr')
        # NB: Scipy+Dask memory leak issues requires these 2 intermediary saves to clear memory...
        cluster_labels = xr.open_zarr(self.scratch_dir+'/02_cluster_labels_2_temp.zarr', chunks={}).cluster_labels

        ###TEMPPP  cumsum_labels = (cluster_labels.max(dim=self.xdim) + 1).cumsum(self.timedim).compute()
        
        ###TEMPPP  min_int64 = np.iinfo(np.int64).min
        ###TEMPPP  cluster_labels_nan = xr.where(cluster_labels<0, min_int64, cluster_labels)  # Otherwise the -1 labels get added to !!!
        ###TEMPPP  cluster_labels_unique = cluster_labels_nan + cumsum_labels
        ###TEMPPP  cluster_labels_unique = cluster_labels_unique #.persist()
        ###TEMPPP  cluster_labels_unique.name = 'cluster_labels_unique'
        ###TEMPPP  cluster_labels_unique.to_zarr(self.scratch_dir+'/02_cluster_labels_unique_2_temp.zarr', mode='w')
        print('Done with Union-Find Algorithm (2)')
        cluster_labels_unique = xr.open_zarr(self.scratch_dir+'/02_cluster_labels_unique_2_temp.zarr', chunks={}).cluster_labels_unique  #, chunks={'time': 4}
                
        
        ## Step 2: Go again (parallel in time) but now checking just for overlap with previous and next time slice.
        #           Keep a running list of all equivalent labels.
        
        cluster_labels_unique_prev = cluster_labels_unique.roll(time=1, roll_coords=False)
        cluster_labels_unique_next = cluster_labels_unique.roll(time=-1, roll_coords=False)
        
        def check_overlap(labels_t, labels_prev, labels_next):
            
            valid_mask = labels_t >= 0

            # Create arrays of indices for valid labels
            valid_indices_t = np.nonzero(valid_mask)[0]
            valid_indices_prev = valid_indices_t[labels_prev[valid_mask] >= 0]
            valid_indices_next = valid_indices_t[labels_next[valid_mask] >= 0]

            # Create pairs using advanced indexing
            pairs_prev = np.stack((labels_t[valid_indices_prev], labels_prev[valid_indices_prev]), axis=1)
            pairs_next = np.stack((labels_t[valid_indices_next], labels_next[valid_indices_next]), axis=1)

            # Concatenate the pairs
            pairs_all = np.concatenate((pairs_prev, pairs_next), axis=0)

            # Sort the pairs and find unique pairs
            pairs_unique = np.unique(np.sort(pairs_all, axis=1), axis=0)

            return pairs_unique.astype(np.int32)
        
        overlap_pairs = xr.apply_ufunc(
                            check_overlap,
                            cluster_labels_unique,
                            cluster_labels_unique_prev,
                            cluster_labels_unique_next,
                            input_core_dims=[[self.xdim], [self.xdim], [self.xdim]],
                            output_core_dims=[[]],
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[object]
                        )
        
        overlap_pairs = overlap_pairs.compute()
        overlap_pairs_all = np.concatenate(overlap_pairs.values)
        print('Done with Finding Time-Connected Labels')
        
        
        ## Step 3: Cluster the overlap_pairs into groups of equivalent labels.
        
        def cluster_time_pairs(overlap_pairs_all):
            # Get unique labels from the overlap pairs
            labels = np.unique(overlap_pairs_all) # 1D sorted unique.
            
            # Create a mapping from labels to indices
            label_to_index = {label: index for index, label in enumerate(labels)}
            
            # Convert overlap pairs to indices
            overlap_pairs_indices = np.array([(label_to_index[pair[0]], label_to_index[pair[1]]) for pair in overlap_pairs_all])
            
            # Create a sparse matrix representation of the graph
            n = len(labels)
            row_indices, col_indices = overlap_pairs_indices.T
            data = np.ones(len(overlap_pairs_indices))
            graph = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
            
            # Find connected components
            num_components, component_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
            
            # Group labels by their component label
            clusters = [[] for _ in range(num_components)]
            for label, component_label in zip(labels, component_labels):
                clusters[component_label].append(label)
            
            return clusters
        
        equivalent_labels = cluster_time_pairs(overlap_pairs_all)
        print('Done with Clustering Time Pairs')
        
        
        ## Step 4: Replace all labels in cluster_labels_unique that match the equivalent_labels with the list index:  This is the new/final label.
        
        # Create a dictionary to map labels to the new cluster indices
        min_int32 = np.iinfo(np.int32).min
        max_label = cluster_labels_unique.max().compute().data
        label_to_cluster_index_array = np.full(max_label + 1, min_int32, dtype=np.int32)

        # Fill the lookup array with cluster indices
        for index, cluster in enumerate(equivalent_labels):
            for label in cluster:
                label_to_cluster_index_array[label] = np.int32(index) # Because these are the connected labels, there are many fewer!
        
        # NB: **Need to pass da into apply_ufunc, otherwise it doesn't manage the memory correctly with large shared-mem numpy arrays !!!
        label_to_cluster_index_array_da = xr.DataArray(label_to_cluster_index_array, dims='label', coords={'label': np.arange(max_label + 1)})
        
        def map_labels_to_indices(block, label_to_cluster_index_array):
            mask = block >= 0
            new_block = np.zeros_like(block, dtype=np.int32)
            new_block[mask] = label_to_cluster_index_array[block[mask]]
            new_block[~mask] = -10
            return new_block
        
        relabeled_unique = xr.apply_ufunc(
            map_labels_to_indices,
            cluster_labels_unique, 
            label_to_cluster_index_array_da,
            input_core_dims=[[self.xdim],['label']],
            output_core_dims=[[self.xdim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32]
        )

        N_final = len(equivalent_labels)
        
        return relabeled_unique, N_final

    
    
    def _label_union_find_unstruct(self, binary_images, temp_filename):
        '''Label all object, with no connectivity in time.
           Utilise highly efficient Unstructured Union-Find (Disjoint Set Union) Clustering Algorithm
        '''
        
        def cluster_true_values(arr, neighbours_int):
            t, n = arr.shape
            labels = np.full((t, n), -1, dtype=np.int64)
            
            for i in range(t):
                true_indices = np.where(arr[i])[0]
                mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(true_indices)}
                
                valid_mask = (neighbours_int != -1) & arr[i][neighbours_int]
                row_ind, col_ind = np.where(valid_mask)
                
                mapped_row_ind = []
                mapped_col_ind = []
                for r, c in zip(neighbours_int[row_ind, col_ind], col_ind):
                    if r in mapping and c in mapping:
                        mapped_row_ind.append(mapping[r])
                        mapped_col_ind.append(mapping[c])
                
                graph = csr_matrix((np.ones(len(mapped_row_ind)), (mapped_row_ind, mapped_col_ind)), shape=(len(true_indices), len(true_indices)))
                _, labels_true = connected_components(csgraph=graph, directed=False, return_labels=True)
                labels[i, true_indices] = labels_true
            
            return labels
        
        ## Label time-independent in 2D (i.e. no time connectivity!)
        binary_images_mask = binary_images.where(~self.land_mask, other=True).persist()  # Mask land
        
        cluster_labels = xr.apply_ufunc(cluster_true_values, 
                                binary_images_mask, 
                                self.neighbours_int, 
                                input_core_dims=[[self.xdim],['nv',self.xdim]],
                                output_core_dims=[[self.xdim]],
                                output_dtypes=[np.int32],
                                vectorize=False,
                                dask='parallelized')
        
        # label = -1 if False
        cluster_labels = xr.where(~self.land_mask, cluster_labels, -1)
        cluster_labels.name = 'cluster_labels'
        cluster_labels.to_zarr(self.scratch_dir+temp_filename, mode='w') 
        
        return True


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
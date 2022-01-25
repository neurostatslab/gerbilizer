import os

import numpy as np
import h5py
from tqdm import trange

nt_new = 10
random_state = np.random.RandomState(123)
index_file_path = 'filtered_idx_notail.npy'

with h5py.File('preprocessed_vocalization_dataset.h5') as f_src:
	# All vocalizations.
	all_vocs = np.array(f_src["vocalizations"][..., :2, :])
	# Select only valid indices
	if os.path.exists(index_file_path):
		indices = np.load(index_file_path)
		all_vocs = all_vocs[indices]
		del indices

	n_samples = all_vocs.shape[0]

	# Create random train, val, test set (80-10-10 split)
	idx = random_state.permutation(n_samples)
	i0 = int(n_samples * .8)
	i1 = int(n_samples * .9)
	train_idx = idx[:i0]
	val_idx = idx[i0:i1]
	test_idx = idx[i1:]

	# Resample locations to 10 timepoints
	loc_src = np.array(f_src["locations"])
	nt_old = loc_src.shape[1]
	n_keypoints = loc_src.shape[2]
	n_spatial_dims = loc_src.shape[3]
	all_locs = np.zeros(
		(n_samples, nt_new, n_keypoints, n_spatial_dims)
	).astype("float32")
	for i in trange(n_samples):
		for j in range(n_keypoints):
			for k in range(n_spatial_dims):
				all_locs[i, :, j, k] = np.interp(
					np.linspace(0, 1, nt_new),
					np.linspace(0, 1, nt_old),
					loc_src[i, :, j, k],
				)

	# Save train set.
	with h5py.File('train_set.h5','w') as f_dest:
		f_dest["vocalizations"] = 1e3 * all_vocs[train_idx]
		f_dest["locations"] = 1e-3 * all_locs[train_idx]
	
	# Save validation set.
	with h5py.File('val_set.h5','w') as f_dest:
		f_dest["vocalizations"] = 1e3 * all_vocs[val_idx]
		f_dest["locations"] = 1e-3 * all_locs[val_idx]
	
	# Save test set.
	with h5py.File('test_set.h5','w') as f_dest:
		f_dest["vocalizations"] = 1e3 * all_vocs[test_idx]
		f_dest["locations"] = 1e-3 * all_locs[test_idx]

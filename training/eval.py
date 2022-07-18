import argparse
from os import path
from pathlib import Path
from sys import stderr

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import build_config_from_file, build_config_from_name
from train import Trainer
from dataloaders import GerbilVocalizationDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datafile",
        type=str,
        help="Path to vocalization data to evaluate.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Used to specify model configuration via a hard-coded named configuration or json file.",
    )

    parser.add_argument(
        "--job_id",
        type=int,
        required=False,
        help="Used to indicate location of model weights, if not already provided by a saved config file.",
    )


    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if not path.exists(args.datafile):
        raise ValueError(f"Error: could not find data at path {args.datafile}")
    
    valid_ext = ('.h5', '.hdf', '.hdf5')
    if not any(args.datafile.endswith(ext) for ext in valid_ext):
        raise ValueError(f"Datafile must be an HDF5 file (.h5, .hdf, .hdf5)")
    
    if args.config is None:
        raise ValueError("No config or model name provided.")
    
    if not path.exists(args.config):
        # Assuming a model name was provided rather than a path to a json file
        args.config_data = build_config_from_name(args.config, args.job_id)
    else:
        # Assuming a json file was provided
        args.config_data = build_config_from_file(args.config, args.job_id)


def run():
    args = get_args()
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    dest_path = Path(args.datafile).stem + '_preds.h5'
    dest_path = str(Path(args.datafile).parent / dest_path)

    with h5py.File(dest_path, 'w') as dest:
        with h5py.File(args.datafile, 'r') as source:
            if 'len_idx' in source:
                n_vox = len(source['len_idx']) - 1
            else:
                n_vox = source['vocalizations'].shape[0]

            source.copy(source['locations'], dest['/'], 'locations')
        
        # Close the h5 here to reopen it in the Dataset obj
        arena_dims = (args.config_data['ARENA_WIDTH'], args.config_data['ARENA_LENGTH'])
        make_xcorr = args.config_data['COMPUTE_XCORRS']
        test_set = GerbilVocalizationDataset(args.datafile, segment_len=args.config_data['SAMPLE_LEN'], arena_dims=arena_dims, make_xcorrs=make_xcorr)
        test_set.samp_size = 30  # take 30 samples from each vocalization. Pass them to the model as if each were a full batch of inputs
        #test_set_loader = DataLoader(test_set, args.config_data['TEST_BATCH_SIZE'], shuffle=False)
        test_set_loader = DataLoader(test_set, 1, shuffle=False)  # Only using Dataloader here for the convenience of converting np.ndarray to torch.Tensor

        preds = dest.create_dataset(
            'predictions',
            shape=(n_vox, test_set.samp_size, 2),
            dtype=np.float32
        )
        vars = dest.create_dataset(
            'std',
            shape=(n_vox,),
            dtype=np.float32
        )

        model = Trainer.from_trained_model(
            args.config_data,
            job_id=args.job_id,
            device_override=device
        )

        model.model.eval()
        with torch.no_grad():
            idx = 0
            for audio, _ in iter(test_set_loader):
                audio = audio.squeeze()
                if device == 'gpu':
                    audio = audio.cuda()
                # n_added = len(audio)
                n_added = 1
                # output = model.model(audio)
                output = model.model(audio)
                # output = output.mean(dim=0, keepdims=True)  # Output should have shape (30, 2)
                centimeter_output = GerbilVocalizationDataset.unscale_features(output.cpu().numpy(), arena_dims=arena_dims)
                centroid = centimeter_output.mean(axis=0)
                distances = np.sqrt( ((centroid[None, ...] - centimeter_output)**2).sum(axis=-1) )  # Should have shape (30,)
                dist_spread = distances.std()
                # preds[idx:idx+n_added] = centimeter_output
                preds[idx] = centimeter_output
                vars[idx] = dist_spread
                idx += n_added


if __name__ == '__main__':
    run()

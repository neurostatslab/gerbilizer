"""
Assess covariance models on a dataset, tracking metrics like mean error,
area of the 95% confidence set for each prediction
and whether the true value was in that set, etc.
"""
import argparse
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from gerbilizer.calibration import CalibrationAccumulator
from gerbilizer.training.dataloaders import GerbilVocalizationDataset
from gerbilizer.training.configs import build_config
from gerbilizer.util import make_xy_grids, subplots
from gerbilizer.training.models import build_model, unscale_output


FIRST_N_VOX_TO_PLOT = 16

def assess_model(
    model,
    dataloader: DataLoader,
    outfile: Union[Path, str],
    arena_dims: tuple,
    device='cuda:0',
    visualize=False
    ):
    """
    Assess the provided model with uncertainty, storing model output as well as
    info like error, confidence sets, and a calibration curve in the h5 format
    at path `outfile`.

    Optionally, visualize confidence sets a few times throughout training.
    """
    outfile = Path(outfile)

    N = len(dataloader)
    RAW_OUTPUT_SHAPE = (N, 3, 2)
    LOC_SHAPE = (N, 2)
    COV_SHAPE = (N, 2, 2)

    with h5py.File(outfile, 'w') as f:
        raw_outputs = f.create_dataset('raw_output', shape=RAW_OUTPUT_SHAPE)
        raw_locations = f.create_dataset('raw_locations', shape=LOC_SHAPE)
        scaled_locations = f.create_dataset('scaled_locations', shape=LOC_SHAPE)
        means = f.create_dataset('means', shape=LOC_SHAPE)
        covs = f.create_dataset('covs', shape=COV_SHAPE)

        ca = CalibrationAccumulator(arena_dims)

        model.eval()
        with torch.no_grad():
            for idx, (audio, location) in enumerate(dataloader):
                audio = audio.to(device)
                output = model(audio)
                np_output = output.cpu().numpy()

                raw_outputs[idx] = np_output
                raw_locations[idx] = location

                # unscale location from [-1, 1] square to units in arena (in mm)
                A = 0.5 * np.diag(arena_dims)  # rescaling matrix
                b = 0.5 * np.array(arena_dims)  # recentering vector
                scaled_location = A.dot(location.cpu().numpy().squeeze()) + b
                scaled_locations[idx] = scaled_location

                # process mean + cov matrix from model output, unscaling to
                # arena size from [-1, 1] square
                unscaled = unscale_output(np_output, arena_dims).squeeze()
                mean = unscaled[0]
                cov = unscaled[1:]
                means[idx] = mean
                covs[idx] = cov
                
                # other useful info
                ca.calculate_step(mean, cov, scaled_location)

                if visualize and idx == FIRST_N_VOX_TO_PLOT:
                    # plot the densities
                    visualize_dir = outfile.parent / 'pmfs_visualized'
                    visualize_dir.mkdir(exist_ok=True, parents=True)
                    visualize_outfile = visualize_dir / f'{outfile.stem}_visualized.png'

                    sets_to_plot = ca.confidence_sets[:idx]
                    associated_locations = scaled_locations[:idx]

                    _, axs = subplots(len(sets_to_plot))

                    xgrid, ygrid = make_xy_grids(arena_dims, shape=sets_to_plot[0].shape, return_center_pts=True)
                    for i, ax in enumerate(axs):
                        ax.set_title(f'vocalization {i}')
                        ax.contourf(xgrid, ygrid, sets_to_plot[i])
                        # add a red dot indicating the true location
                        ax.plot(*associated_locations[i], 'ro')
                        ax.set_aspect('equal', 'box')

                    plt.savefig(visualize_outfile)
                    print(f'Model output visualized at file {visualize_outfile}')

        f.attrs['mean_error'] = np.linalg.norm(scaled_locations[:] - means[:], axis=-1).mean()

        results = ca.results()
        f.attrs['calibration_curve'] = results['calibration_curve']

        # array-like quantities outputted by the calibration accumulator
        OUTPUTS = ('confidence_sets', 'confidence_set_areas', 'location_in_confidence_set', 'distances_to_furthest_point')
        for output_name in OUTPUTS:
            f.create_dataset(output_name, data=results[output_name])

    print(f'Model output saved to {outfile}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Used to specify model configuration via a JSON file.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to an h5 file on which to assess the model",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help="Path at which to store results. Must be an h5 file.",
    )

    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Include flag to plot confidence sets occasionally during assessment."
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        raise ValueError(
            f"Requested config JSON file could not be found: {args.config}"
        )

    config_data = build_config(args.config)

    # load the model
    weights_path = config_data.get('WEIGHTS_PATH')
    if not weights_path:
        raise ValueError(
            f"Cannot evaluate model as the config stored at {args.config} doesn't include path to weights."
        )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        config_data['DEVICE'] = 'cpu'
    model, _ = build_model(config_data)
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights, strict=False)

    arena_dims = (config_data["ARENA_WIDTH"], config_data["ARENA_LENGTH"])
    dataset = GerbilVocalizationDataset(
        str(args.data),
        segment_len=config_data["SAMPLE_LEN"],
        arena_dims=arena_dims,
        make_xcorrs=config_data["COMPUTE_XCORRS"],
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    # make the parent directories for the desired outfile if they don't exist
    parent = Path(args.outfile).parent
    parent.mkdir(exist_ok=True, parents=True)

    assess_model(
        model,
        dataloader,
        args.outfile,
        arena_dims,
        device=device,
        visualize=args.visualize,
        )
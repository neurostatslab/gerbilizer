"""
Assess covariance models on a dataset, tracking metrics like mean error,
area of the 95% confidence set for each prediction
and whether the true value was in that set, etc.
"""

import argparse
import json
import logging
from itertools import repeat
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from gerbilizer.architectures.base import GerbilizerArchitecture
from gerbilizer.architectures.ensemble import GerbilizerEnsemble
from gerbilizer.calibration import CalibrationAccumulator
from gerbilizer.outputs.base import ModelOutput, ProbabilisticOutput, Unit
from gerbilizer.training.configs import build_config
from gerbilizer.training.dataloaders import GerbilVocalizationDataset
from gerbilizer.training.models import build_model
from gerbilizer.util import make_xy_grids, subplots

logging.getLogger("matplotlib").setLevel(logging.WARNING)

FIRST_N_VOX_TO_PLOT = 16


def plot_results(f: h5py.File):
    """
    Create and save plots of calibration curve, error distributions, etc.
    """
    point_predictions = f["point_predictions"][:]

    errs = np.linalg.norm(point_predictions - f["scaled_locations"][:], axis=-1)
    if "calibration_curve" not in f.attrs:
        fig, err_ax = plt.subplots(1, 1, sharex=False, sharey=False)

        # convert errs to cm for readability
        errs_cm = errs / 10

        err_ax.hist(errs_cm, color="tab:blue")
        err_ax.set_xlabel("errors (cm)")
        err_ax.set_ylabel("counts")
        err_ax.set_title("error distribution")
        return fig, err_ax
    else:
        fig, axs = plt.subplots(1, 5, sharex=False, sharey=False, figsize=(20, 4))
        (err_ax, calib_ax, cset_area_ax, cset_radius_ax, dist_ax) = axs.flat

        # convert errs to cm for readability
        errs_cm = errs / 10

        err_ax.hist(errs_cm, color="tab:blue")
        err_ax.set_xlabel("errors (cm)")
        err_ax.set_ylabel("counts")
        err_ax.set_title("error distribution")

        calib_ax.plot(np.linspace(0, 1, 11), f.attrs["calibration_curve"][:], "bo")
        calib_ax.set_xlabel("probability assigned to region")
        calib_ax.set_ylabel("proportion of locations in the region")
        calib_ax.set_title("calibration curve")

        # convert confidence set areas to cm^2 for readability
        cset_areas_cm = f["confidence_set_areas"][:] / 100

        cset_area_ax.hist(cset_areas_cm, color="tab:blue")
        cset_area_ax.set_xlabel("confidence set area (cm^2)")
        cset_area_ax.set_ylabel("counts")
        cset_area_ax.set_title("confidence set area distribution")

        cset_radius_ax.plot(np.sqrt(cset_areas_cm), errs_cm, "bo")
        cset_radius_ax.set_xlabel("square root confidence set area (cm)")
        cset_radius_ax.set_ylabel("error (cm)")
        cset_radius_ax.set_title("sqrt confidence set area vs error")

        # convert distances to cm
        dist_ax.plot(f["distances_to_furthest_point"][:] / 10, errs_cm, "bo")
        dist_ax.set_xlabel("distance to furthest point in confidence set (cm)")
        dist_ax.set_ylabel("error (cm)")
        dist_ax.set_title("distance to furthest point vs error")

        return fig, axs


def assess_model(
    model: GerbilizerArchitecture,
    dataloader: DataLoader,
    outfile: Union[Path, str],
    arena_dims: Union[np.ndarray, tuple[float, float]],
    device: Union[str, torch.device] = "cuda:0",
    visualize: bool = False,
    temperature: float = 1.0,
    inference: bool = False,
):
    """
    Assess the provided model with uncertainty, storing model output as well as
    info like error, confidence sets, and a calibration curve in the h5 format
    at path `outfile`.

    Optionally, visualize confidence sets a few times throughout training.

    Args:
        model: instantiated GerbilizerArchitecture object
        dataloader: DataLoader object on which the model should be assessed
        outfile: path to an h5 file in which output should be saved
        arena_dims: arena dimensions, *in millimeters*.
        visualize: optional, indicates whether first few outputs should be plotted
        temperature: optional, adjusts entropy of probabilistic model outputs
    """
    outfile = Path(outfile)

    N = len(dataloader.dataset)

    with h5py.File(outfile, "w") as f:
        # Save the config data in the h5 file for future reference
        config_string = json.dumps(model.config)
        f.attrs["model_config"] = config_string

        scaled_locations_dataset = None

        if isinstance(model, GerbilizerEnsemble):
            raw_output_dataset = []
            for i, constituent in enumerate(model.models):
                raw_output_dataset.append(
                    f.create_dataset(
                        f"constituent_{i}_raw_output", shape=(N, constituent.n_outputs)
                    )
                )
        else:
            raw_output_dataset = f.create_dataset(
                "raw_model_output", shape=(N, model.n_outputs)
            )

        point_predictions = f.create_dataset("point_predictions", shape=(N, 2))

        ca = CalibrationAccumulator(arena_dims)

        model.eval()

        should_compute_calibration = False

        with torch.no_grad():
            idx = 0
            for sounds, locations in tqdm(dataloader):
                sounds = sounds.to(device)
                # If inference, locations will be None

                outputs: list[ModelOutput] = model(sounds, unbatched=True)
                for output, location in zip(outputs, locations):
                    # add batch dimension back
                    if isinstance(model, GerbilizerEnsemble):
                        for out_dataset, constituent_output in zip(
                            raw_output_dataset, output.raw_output
                        ):
                            out_dataset[idx] = (
                                constituent_output.raw_output.squeeze().cpu().numpy()
                            )
                    else:
                        raw_output_dataset[idx] = (
                            output.raw_output.squeeze().cpu().numpy()
                        )

                    point_predictions[idx] = (
                        output.point_estimate(units=Unit.MM).cpu().numpy()
                    )

                    # unscale location from [-1, 1] square to units in arena (in mm)
                    if location is not None:
                        scaled_location = (
                            output._convert(location[None], Unit.ARBITRARY, Unit.MM)
                            .cpu()
                            .numpy()
                        )
                        if scaled_locations_dataset is None:
                            scaled_locations_dataset = f.create_dataset(
                                "scaled_locations", shape=(N, 2)
                            )
                        scaled_locations_dataset[idx] = scaled_location

                        # other useful info
                        if isinstance(output, ProbabilisticOutput):
                            should_compute_calibration = True
                            ca.calculate_step(
                                output, scaled_location, temperature=temperature
                            )

                            if visualize and idx == FIRST_N_VOX_TO_PLOT:
                                # plot the densities
                                visualize_dir = outfile.parent / "pmfs_visualized"
                                visualize_dir.mkdir(exist_ok=True, parents=True)
                                visualize_outfile = (
                                    visualize_dir / f"{outfile.stem}_visualized.png"
                                )

                                _, axs = subplots(FIRST_N_VOX_TO_PLOT)

                                for i, ax in enumerate(axs):
                                    ax.set_title(f"vocalization {i}")
                                    ax.plot(
                                        *point_predictions[i], "ro", label="predicted"
                                    )
                                    # add a green dot indicating the true location
                                    ax.plot(
                                        *scaled_locations_dataset[i], "go", label="true"
                                    )
                                    ax.set_aspect("equal", "box")

                                    if should_compute_calibration:
                                        set_to_plot = ca.confidence_sets[i]

                                        xgrid, ygrid = make_xy_grids(
                                            arena_dims,
                                            shape=set_to_plot.shape,
                                            return_center_pts=True,
                                        )
                                        ax.contourf(
                                            xgrid,
                                            ygrid,
                                            set_to_plot,
                                            label="95% confidence set",
                                        )
                                    ax.legend()

                                plt.savefig(visualize_outfile)
                                print(
                                    f"Model output visualized at file {visualize_outfile}"
                                )

                    # update number of vocalizations seen
                    idx += 1

        # Cannot compute calibration curve on inference data
        if should_compute_calibration and not inference:
            results = ca.results()
            f.attrs["calibration_curve"] = results["calibration_curve"]

            # array-like quantities outputted by the calibration accumulator
            OUTPUTS = (
                "confidence_sets",
                "confidence_set_areas",
                "location_in_confidence_set",
                "distances_to_furthest_point",
            )
            for output_name in OUTPUTS:
                f.create_dataset(output_name, data=results[output_name])

        if not inference:
            _, axs = plot_results(f)
            plt.tight_layout()
            plt.savefig(Path(outfile).parent / f"{Path(outfile).stem}.png")

    print(f"Model output saved to {outfile}")


if __name__ == "__main__":
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
        action="store_true",
        help="Include flag to plot confidence sets occasionally during assessment.",
    )

    parser.add_argument(
        "--use-final",
        action="store_true",
        help="Include flag use the FINAL model weights, not the best.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="Optional flag to apply temperature scaling to a model with probabilistic output.",
    )

    parser.add_argument(
        "--inference",
        action="store_true",
        help="Include flag to evaluate the model on datasets without ground truth.",
    )

    parser.add_argument(
        "--index",
        type=Path,
        required=False,
        help="Optional path to an index file to use for assessement. Should be a numpy int array of indices into the dataset.",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        raise ValueError(
            f"Requested config JSON file could not be found: {args.config}"
        )

    config_data = build_config(args.config)

    model, _ = build_model(config_data)

    best_weights_path = config_data.get("WEIGHTS_PATH", None)
    model.load_weights(
        best_weights_path=best_weights_path, use_final_weights=args.use_final
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        config_data["GENERAL"]["DEVICE"] = "cpu"
    model = model.to(device)

    arena_dims = np.array(config_data["DATA"]["ARENA_DIMS"])
    arena_dims_units = config_data["DATA"].get("ARENA_DIMS_UNITS")

    # if provided in cm, convert to MM
    if arena_dims_units == "CM":
        arena_dims = np.array(arena_dims) * 10
    elif arena_dims_units == "M":
        arena_dims = np.array(arena_dims) * 1000
    else:
        raise ValueError(
            "ARENA_DIMS_UNITS must be one of 'CM' or 'M' to specify the units of the arena dimensions."
        )

    index = None
    if args.index is not None:
        if not Path(args.index).exists():
            raise ValueError(f"Requested index file could not be found: {args.index}")
        index = np.load(args.index)

    dataset = GerbilVocalizationDataset(
        str(args.data),
        arena_dims=arena_dims,
        crop_length=config_data["DATA"]["CROP_LENGTH"],
        inference=args.inference,
        index=index,
    )

    batch_size = config_data["DATA"]["BATCH_SIZE"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate
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
        temperature=args.temperature,
        inference=args.inference,
    )

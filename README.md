# Gerbilizer
==============================

## Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    │
    ├── trained_models                          <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              and a short `-` delimited description, e.g.
    │                                              `1.0-initial-data-exploration`.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment
    │
    ├── gerbilizer                                <- Scripts to create new models and train them.


## Installation
1. Clone this repository: `git clone https://github.com/neurostatslab/gerbilizer.git`
2. Install prerequisites: `cd gerbilizer & pipenv install`
3. Install with pip: `pip install .`

## Usage
1. Create a dataset. This should be an HDF5 file with the following datasets:

| Dataset group/name | Shape             | Data type | Description                                                                                                                                    |
|--------------------|-------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------|
| /audio     | (*, n_channels) | float     | All sound events concatenated along axis 0                                                                                                     |
| /length_idx           | (n + 1,)          | int       | Index into audio dataset. Sound event `i` should span the half open interval [`length_idx[i]`, `length_idx[i+1]`) and the first element should be 0. |
| /locations         | (n, 2)            | float     | Locations associated with each sound event. Only required for training.                                                                        |
1. Optionally, partition the dataset into a validation set and training set with the names `train_set.h5` and `val_set.h5`. Alternatively, the complete dataset can be passed in as a single .h5 file and an 80/20 train/val split will be automatically generated and saved. A directory containing training and validation indices by the names `train_set.npy` and `val_set.npy` can also be passed in through the `--indices` argument.
2. Create a config. This is a JSON file consisting of a single object whose properties correspond to the hyperparameters of the model and optimization algorithm.
3. Train a model:
   1. With SLURM: `sbatch run_gpu.sh /path/to/directory/containing/trainset/ /path/to/config.json`. Note that the first argument is expected to be a directory and the second argument is expected to be a file
   2. Without SLURM: `python -m gerbilizer --data /path/to/directory/containing/trainset/ --config /path/to/config.json --save-path /path/to/model/weight/directory/ --indices /optional/path/to/index/directory`
4. Optionally, run with pretrained weights by pointing to the config.json within the model weight directory of the pretrained model.
5. Perform inference:
   1. With SLURM: `sbatch run_eval.sh /path/to/hdf5/dataset.h5 /path/to/model_dir/trained_models/config_name/#####/config.json /optional/output/path.h5`. 
   2. Without SLURM: `python -m gerbilizer.assess --inference --data /path/to/hdf5/dataset.h5 --config /path/to/model_dir/trained_models/config_name/#####/config.json -o /optional/output/path.h5 --index /optional/index/path.npy`.
   3. Note that here, the data argument is a file rather than a directory and the config should point toward a config.json in a trained model's directory, just as in the finetuning step. If no output path is provided, predictions will be stored in the same directory as the input dataset.
   4. Output predictions will be stored in a dataset labeled "predictions" at the root of the HDF5 hierarchy

## Public datasets
Robot dataset:
  - [Processed H5](https://users.flatironinstitute.org/~atanelus/datasets/robot_dataset_full.h5) 44GiB

Adolescent dataset:
  - [Processed H5](https://users.flatironinstitute.org/~atanelus/datasets/adolescent_dataset_full.h5) 1.8GiB

Speaker dataset:
  - [Processed H5](https://users.flatironinstitute.org/~atanelus/datasets/speaker_dataset_full.h5) (Includes `stimulus_identities`, and `stimulus_names` datasets) 11GiB
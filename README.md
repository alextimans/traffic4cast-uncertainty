## Overview
This is the code repository for the research paper titled "Uncertainty Quantification for Image-based Traffic Prediction across Cities". Full paper here: [Link](https://arxiv.org/). 

#### Abstract :memo:
---
Despite the strong predictive performance of deep learning models for traffic prediction, their widespread deployment in real-world intelligent transportation systems has been restrained by a lack of interpretability. Uncertainty quantification (UQ) methods provide an approach to induce probabilistic reasoning, improve decision-making and enhance model deployment potential. To gain a comprehensive picture on the usefulness of existing UQ methods for traffic prediction and the relation between obtained uncertainties and city-wide traffic dynamics, we investigate their application to a large-scale image-based traffic dataset spanning multiple cities and time periods. We compare two epistemic and two aleatoric UQ methods on both temporal and spatio-temporal transfer tasks, and find that meaningful uncertainty estimates can be recovered. We further demonstrate how uncertainty estimates can be employed for unsupervised outlier detection on changes in city traffic dynamics. We find that our approach can capture both temporal and spatial effects on traffic behaviour in a representative case study for the city of Moscow. Our work presents a further step towards boosting uncertainty awareness in traffic prediction tasks, and aims to highlight the value contribution of UQ methods to a better understanding of city traffic dynamics. The code for our experiments is publicly available.

---

If you find this work useful, please consider citing:
```
@online{timans2023uqtraffic,
  author       = {Timans, Alexander and Wiedemann, Nina and Kumar, Nishant and Hong, Ye and Raubal, Martin},
  title        = {Uncertainty Quantification for Image-based Traffic Prediction across Cities},
  year         = {2023},
  eprinttype   = {arxiv},
  eprint       = {arXiv:XXXX.XXXX},
  url          = {https://arxiv.org/abs/XXXX.XXXX},
}
```

## Folder structure
The folder structure of the repo is quite self explanatory, but below are some comments on each folder or file on the main repository level. The most important folder is ```uq```.
```
traffic4cast-uncertainty/
├── data: contains the custom dataset and a script to generate the correct data folder structure
├── metrics: contains files to compute MSE, calibration and prediction interval metrics, as well as aggregate .txt results files into .csv tables
├── misc: nothing particularly interesting, last layer correlation and image transforms
├── model: contains model architectures, parameter configs and training-related files.
├── plotting: contains scripts to visualise spatial and temporal uncertainty and metrics maps, as well as outlier detection results, and some other plots.
├── uq: contains the implementations of all evaluated UQ methods, as well as the test set and outlier detection evaluation scripts to produce results.
├── util: contains utility functions such as read/write data, set seed or get device. The most important file in here is 'h5_util.py' data utility.
├── env_current.yml: The package environment file as used by myself.
├── env_t4c.yml: The package environment file as given by the Traffic4cast 2021 challenge organizers (https://github.com/iarai/NeurIPS2021-traffic4cast)
└── main.py: The main script to run model training and inference via CLI.
```
The implemented UQ methods as mentioned in the paper correspond to the following names used in the code, in particular regarding argument names for parameter ```uq_method``` in ```main.py``` or names used in the inference scripts ```eval_model.py``` and ```outlier_detection.py``` :
| Paper | Code |
|--|--|
| TTA + Ens | combo |
| Patches + Ens | combopatch |
| CUB | point |
| Ens | ensemble |
| MCBN | bnorm |
| TTA | tta |
| Patches | patches |

## Running the code locally
To run the code, guidelines are provided below.

#### Preparing the local setup
1. Clone repo, e.g. using Github CLI
```
gh repo clone alextimans/traffic4cast-uncertainty
```
- **Note:** Make sure to be in the parent directory of ```traffic4cast-uncertainty``` as working directory. Let's call it ```run-code```, so set working directory to ```run-code```. This directory is also the one from which to launch code runs, which is why the provided sample runs below follow the scheme ```python traffic4cast-uncertainty/[script.py -args]```.

2. Set-up python env (e.g. with conda package manager)
- Either generate python env using the t4c competition environment via ```env_t4c.yml``` and add potentially missing packages manually, or use ```env_curr.yml``` for a more stringent but encompassing environment.
- Uncomment related lines in either ```.yml``` file in case of local machine with GPU support.
```
conda env create -f traffic4cast-uncertainty/env_curr.yml
conda activate t4c
```
- Init code repo for python path (see also [here]((https://github.com/iarai/NeurIPS2021-traffic4cast)))
```
cd traffic4cast-uncertainty
export PYTHONPATH="$PYTHONPATH:$PWD"
cd ..
```

3. Get and prepare the data
- Get the data by visiting the [Traffic4cast webpage](https://www.iarai.ac.at/traffic4cast/2021-competition/challenge/) and following instructions on accessing the data. **Note:** this requires registration with the competition website and may take a few days until IARAI grants access. Otherwise it is perhaps best to reach out directly to the challenge organizers.
- Put all the data in a folder ```run-code/data/raw``` in the working directory in uncompressed city folders as given by the *Traffic4cast* competition.
- Run the following call in CLI to create data folders in line with the data structure used in this work and remove 2020 leap year days.
```
  python traffic4cast-uncertainty/data/set_data_folder_str.py --data_raw_path="./data/raw" --remove_leap_days=True
```
- The data should now be separated into ```train```, ```val``` and ```test``` folders within ```run-code/data/raw/[city]```.

4. Ready! Execute desired code runs as listed below. 

#### Local model runs

**Note:** These commands are all for running code on your local machine. This may not be particularly suitable for e.g. model training or inference on large portions of the data, since runtimes will be extremely long. Most of the heavy computations for this work were executed on GPUs on a computing cluster. The provided runs are primarily sample runs to visualize the use of commands and arguments. To exactly replicate results please reach out to obtain exact job submission script files.

**Note:** To run ```uq_method=combo``` or ```uq_method=combopatch``` correctly one needs to uncomment an import statement in ```main.py```. Similarly, to use ```model_str=unet_pp``` vs. ```model_str=unet``` there are lines that need to be uncommented in the respective UQ method's files in the ```uq``` folder, and in ```uq/eval_patches_ensemble.py``` and ```uq/eval_tta_ensemble.py```if those methods are run. These statements are marked clearly with ```# MODIFY HERE``` statements.

- To check the main script CLI arguments that can be given and their default values and short explanations.
```
python traffic4cast-uncertainty/main.py -h
```

- Training only from scratch
```
python traffic4cast-uncertainty/main.py --model_str=unet --model_id=1 --model_evaluation=False --batch_size=10 --epochs=2 --device=cpu --num_workers=4 --random_seed=9876543 --data_raw_path="path/to/data" --save_checkpoint="path/to/checkpoint"
```

- Continue training only from previous checkpoint (e.g. ```unet_1.pt```)
```
python traffic4cast-uncertainty/main.py --model_str=unet --model_id=1 --model_evaluation=False --batch_size=10 --epochs=2 --device=cpu --num_workers=4 --random_seed=9876543 --data_raw_path="path/to/data" --save_checkpoint="path/to/checkpoint" --resume_checkpoint="path/to/unet_1.pt"
```

- Calibration run from previous checkpoint only. This is used to generate conformal quantiles which are needed for a test set run. It is important to set ```batch_size=1```. ```quantiles_path``` defines where quantile files will be written to. Select the desired UQ method via ```uq_method```. 
```
python traffic4cast-uncertainty/main.py --model_str=unet --model_id=1 --model_training=False --model_evaluation=False --calibration=True --calibration_size=500 --uq_method=tta --device=cpu --num_workers=4 --random_seed=9876543 --data_raw_path="path/to/data" --resume_checkpoint="path/to/unet_1.pt" --quantiles_path="path/to/quantiles"
```

- Test set evaluation run from previous checkpoint only. Again it is important to keep ```batch_size=1```. We also require quantile files that are located in the folder denoted by ```quantiles_path```, as well as a test sample indices file located at ```test_samp_path``` and called by ```test_samp_name```. The result files are written into the folder denoted by ```test_pred_path```.  Select the desired UQ method via ```uq_method```.
```
python traffic4cast-uncertainty/main.py --model_str=unet --model_id=1 --model_training=False --model_evaluation=True --calibration=False --uq_method=tta --device=cpu --num_workers=4 --random_seed=9876543 --data_raw_path="path/to/data" --quantiles_path="path/to/quantiles" --test_pred_path="path/to/test_preds" --test_samp_path="path/to/test_samp" --test_samp_name"test_samp_file.txt"
```

- Outlier detection script using epistemic uncertainty estimates via deep ensembles. There are a number of relevant boolean flags in the script which activate and deactivate certain sections, and whose meaning can be checked by calling ```python traffic4cast-uncertainty/uq/outlier_detection.py -h```. If not specified, the full script is run. Key parameters include ```fix_samp_idx```, which specifies the fixed time slot per day that we evaluate outliers on; ```cities```, which specifies the cities to run outlier detection for; and ```out_bound```, which specifies the fixed outlier bound used to decide on binary outlier labels. Outlier results will be written to ```test_pred_path/out_name```.
```
python traffic4cast-uncertainty/main.py --model_str=unet --model_id=1 --uq_method=ensemble --batch_size=1 --fix_samp_idx 120 100 110 --cities BANGKOK BARCELONA MOSCOW --out_bound=0.001 --device=cpu --num_workers=4 --random_seed=9876543 --data_raw_path="path/to/data" --resume_checkpoint="path/to/unet_1.pt" --test_pred_path="path/to/test_preds" --out_name="outliers"
```

- Create test set random sample indices
```
python traffic4cast-uncertainty/metrics/get_random_sample_idx.py --samp_size=100 --test_samp_path="path/to/test_samp" --test_samp_name="test_samp"
```

- Given obtained results, aggregate into .csv table
```
python traffic4cast-uncertainty/metrics/get_scores_to_table.py --test_pred_path="path/to/test_preds"
```

- Aggregate multiple .csv tables into one by taking means over scores
```
python traffic4cast-uncertainty/metrics/get_scores_to_table.py --test_pred_path="path/to/test_preds" --create_table=False --agg_table=True --agg_nr=5
```

- Plot epistemic histograms with KDE fits for a fixed pixel (visual inspection of part of outlier detection framework)
```
python traffic4cast-uncertainty/plotting/epistemic_hist.py --test_pred_path="path/to/test_preds" --uq_method=ensemble --pixel 100 100 --mask_nonzero=False --fig_path="path/to/figures"
```

- To generate other plots, one has to directly work in the respective scripts in the ```plotting``` folder (there is no access point via CLI). Static values are set at the top of the respective scripts and are clearly marked. Code snippets to generate individual plots are also clearly delineated.


#### Other code info

- Unless surpressed, model runs perform a healthy amount of logging info to keep the model run process transparent.

- To specifically switch off either training or test set evaluation one needs to explicitly set the respective argument to ```False```, i.e. either ```model_training=False``` or ```model_evaluation=False```. Defaults for both are ```True```. ```calibration``` is set by default to ```False```.

- Static parameter values for model settings and UQ methods are located in ```model/configs.py```. This includes model and optimizer hyperparameters, data padding, and parameters for UQ methods such as the ensemble size, number of stochastic forward passes for MCBN or patch size.

- To run a model on the full data (either train + val or test or both) simply omit any data limiting arguments, namely ```data_limit, train_data_limit, val_data_limit, test_data_limit```.

- Running the training from scratch should result in a folder ```unet_1``` created in ```[parent dir]/checkpoints``` with a bunch of loss files. There are: train + val loss files per batch and per epoch. On top there should be model checkpoint files (e.g. ```unet_ep0_05061706.pt``` and ```unet_ep1_05061707.pt``` because we trained for two epochs and save each epoch).

- During training, setting the CLI argument ```model_id``` uniquely is crucial as this will be used to create the checkpoint folder. If not set properly and training a new model, the files in the existing folder ```[parent dir]/checkpoints/unet_{model_id}``` may risk being overwritten and lost. Thus whenever training a new model from scratch make sure that ```model_id``` is unique to create a new checkpoint folder. When continuing training of an already existing model checkpoint, make sure to set the ```model_id``` to the **same** value so that subsequent checkpoint files are saved in the same checkpoint folder (it is not a requirement but makes sense to avoid confusion).

- The device is either automatically inferred from the given machine or explicitly handed via argument ```device```. It takes either values ```cpu``` or ```cuda```. To run on multiple GPUs, set argument ```data_parallel=True``` which will activate PyTorch's DataParallel framework (if those are available). Training etc. should all work smoothly in the same manner as running on e.g. ```cpu```.


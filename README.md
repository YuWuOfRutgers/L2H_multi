# Learning to Help in Multi-Class Settings
This package is the python implemention of the paper *Learning to Help in Multi-Class Settings*

## Installation

You will need anaconda, pytorch and torchvision to run the code in this repository.

## TL;DR

Run the following command to reproduce the results:

```bash
mkdir -p results
# for reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_WORKSPACE_CONFIG=:16:8
(bash train-conf-score.sh run1 && bash eval.sh run1 && bash plot.sh run1) | tee results/run1.log
bash train-async.sh run-async && bash eval-async.sh run-async && bash plot-async.sh run-async
bash train-edgeonly.sh | tee results/edgeonly.log
```

### Explanation

- [train-conf-score.sh](train-conf-score.sh) trains the local and edge models with different confidence scores.
- [eval.sh](eval.sh) evaluates the local and edge models for plotting.
- [plot.sh](plot.sh) plots the results.
- [train-async.sh](train-async.sh) trains edge models and the rejector asynchronously, with different confidence scores and asynchronous frequency.
- [eval-async.sh](eval-async.sh) evaluates the edge models for plotting.
- [plot-async.sh](plot-async.sh) plots the results for the asynchronous training.
- [train-edgeonly.sh](train-edgeonly.sh) trains the edge model only.

## Training

You can mannually train the local model by running the following command:

```bash
python train_localnet.py --dataset <dataset_name> --save_dir <path_to_save_dir>
```

Then train the edge model by running the following command:

```bash
python train_edgenet_async.py --dataset <dataset_name> --save_dir <path_to_save_dir> --cost_1 <cost_1_value> --cost_e <cost_e_value> --async_freq <async_freq_value>
```

Please refer to the training script and sh script details on the arguments.

## Evaluation

To manually evaluate a single model and print out the results, run the following command:

```bash
python eval_fprint.py --dataset <dataset_name> --save_dir <path_to_save_dir> --model_path <path_to_model_to_evaluate>
```

## Citation

If you find our code or paper useful, please cite:

```
@misc{wu2025learninghelpmulticlasssettings,
      title={Learning to Help in Multi-Class Settings},
      author={Yu Wu and Yansong Li and Zeyu Dong and Nitya Sathyavageeswaran and Anand D. Sarwate},
      year={2025},
      eprint={2501.13810},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.13810},
}
```

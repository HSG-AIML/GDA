# Parameter Efficient Self-Supervised Geospatial Domain Adaptation
**[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.pdf)** | **[Poster](assets/16296_poster.pdf)** | **[Slides](https://cvpr.thecvf.com/media/cvpr-2024/Slides/31624.pdf)** | **[Video](https://www.youtube.com/watch?v=SwmF-m5IBEw&t=8s)** | **[Blog](https://hsg-aiml.github.io/2024/06/12/Parameter_Efficient_Self_Supervised_Geospatial_Domain_Adaptation.html)** | **[CVPR Page](https://cvpr.thecvf.com/virtual/2024/poster/31624)**

This repository contains code supporting the CVPR 2024 paper [Parameter Efficient Self-supervised Geospatial Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.html).

Authors: [Linus Scheibenreif](https://scheibenreif.github.io)    [Michael Mommert](https://mommermi.github.io) [Damian Borth](https://ics.unisg.ch/chairs/damian-borth-artificial-intelligence-and-machine-learning/)

![Overview image](assets/overview_v2.jpg "Method Overview")

# Background
This work proposes a three-step approach to adapt geospatial foundation models to new dataset modalities (see figure above):
1. Add SLR adapter parameters to the pre-trained foundation model.
2. Train the adapters via self-supervised masked image modeling on unlabeled data from the target *domain*.
3. Fine-tune the adapters supervised for the target *task*.

This codebase includes scripts to train and evaluate different geospatial foundation models on a number of remote sensing datasets.

# Getting Started
This codebase provides scripts to add SLR adapters to existing, trained, visual foundation models before fine-tuning them on different downstream tasks. To get started, make sure that the trained weights for a visual foundation model are available in the `checkpoints/` directory and download a dataset for training (either via the `torchgeo` library or through the links provided below). For each foundation model, the pre-trained weights should be stored in a different sub-directory (`checkpoints/{mae, sat-mae, scale-mae}`)
See below for the models and datasets used in the paper. 

## Training SLR adapters (Steps 1 + 2)
Follow these steps to add and train SLR adapters for one of the supported foundation model / dataset combinations.
* Update the configuration at `configs/mae/experiment.yaml` with the following values:
    * `wandb.entity`: your wandb id
    * `data.datamodule`: class name of the desired datamodule (see `src/datamodules` for all supported options)
    * `data.modality` if the chosen dataset contains multiple modalities, choose one here.
    * `data.size` similarly, if the dataset contains images of different sizes, choose one here (e.g., TreeSatAI). 
    * `data.root` set to the parent directory of the dataset.
    * `model.name` sets the foundation model, supported options are `mae`, `sat_mae`, and `scale_mae`.
    * to change the bottleneck dimensionality of the low-rank adapters (called *r* in the paper), set `model.adapter_hidden_dim`. 
    * finally, change all other settings (e.g., learning rate or patch size) as you wish.
* Start the training run with `python main_mae.py`. 

Details about the training run will be logged to weights and biases and stored in the `outputs/` directory.

## Fine-tuning / Linear eval (Step 3)
To evaluate a model with trained SLR adapters, follow these steps.
* Update the configuration at `configs/lin_eval/experiment.yaml`:
    * `continual_pretrain_run` set the wandb run id of the SLR adapter pre-training run (see step above).
    * `data.*` make sure the data configuration matches your run from Step 2 above (now you also need to set the number of classes in the dataset). Optionally, few-shot experiments can be run by setting `few_shot_k` and a `few_shot_seed`. 
    * `model.*` settings need to match the pre-training run.
* Start the training run with `python main_linear_eval.py`.

## Evaluation on the test set

To test fine-tuned models (i.e., after Step 3), create a file containing all wandb run ids that should be tested (see `configs/run_ids.csv` for a template). This can also only contain one line in addition to the column names. Then poin the `run_id_file` variable in `main_test_checkpoints.py` to that file. Finally, run `python main_test_checkpoints.py` to run the tests. Results will be written to files in the `logs/tests/` directory.


## Models
**MAE**
* [Project](https://github.com/facebookresearch/mae)
* [ViT-L Encoder](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth)
* [ViT-L Encoder-Decoder](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth)

**SatMAE**
* [Project](https://github.com/sustainlab-group/SatMAE)
* [FMoW Model](https://zenodo.org/record/7369797/files/fmow_pretrain.pth)
* [FMoW Temporal Model](https://zenodo.org/record/7369797/files/pretrain_fmow_temporal.pth)

**Scale-MAE**
* [Project](https://github.com/bair-climate-initiative/scale-mae)

## Datasets
Where possible, use [torchgeo](https://github.com/microsoft/torchgeo) implementations of remote sensing datasets. Please download the other dataset from these locations:
* [TreeSatAI](https://zenodo.org/records/6598391)
* [EuroSAT-SAR](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR)
* [BENGE-8k](https://github.com/HSG-AIML/ben-ge)


# Dependencies
* see `requirements.txt`
* `torchgeo==0.5.0`
* `torch==2.0.1`


# Acknowledgements
If you would like to reference our work, please use the following reference:
```bibtex
@InProceedings{Scheibenreif_2024_CVPR,
    author    = {Scheibenreif, Linus and Mommert, Michael and Borth, Damian},
    title     = {Parameter Efficient Self-Supervised Geospatial Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27841-27851}
}
```

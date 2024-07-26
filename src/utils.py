"""Collection of utility methods for model training and evaluation."""

import copy
import os
import random
from typing import Tuple

import glob
import kornia
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import numpy as np
import torch
import torchgeo.transforms
import wandb
import yaml

import src.datamodules


def update_configs(config: dict) -> dict:
    """Creates a new config dict without Dotdict entries.

    Args:
        config: a dict that might contain Dotdict type entries.

    Returns:
        a new dictionary where Dotdicts objects are resolved into dicts.

    """

    updated_configs = {}
    for k, v in config.__dict__.items():
        if isinstance(v, Dotdict):
            updated_configs[k] = v.__dict__
        else:
            updated_configs[k] = v

    return updated_configs


def set_resources(num_threads: int, wand_cache_dir: str = None):
    """Sets environment variables to control resource usage.

    The environment variables control the number of used threads
    for different vector op libraries and GDAL. The cache dir controls
    where wandb cache is stored locally.

    Args.
        num_threads: the max number of threads.
        wand_cache_dir: path to the desired cache dir

    """

    num_threads = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    os.environ["GDAL_NUM_THREADS"] = num_threads

    if wand_cache_dir:
        os.environ["WANDB_CACHE_DIR"] = wand_cache_dir


def set_seed(seed: int):
    """Set the seed across multiple libraries.

    Sets seed for builtin random, numpy, and torch libraries.

    Args:
        seed: the seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Dotdict:
    """Wraps dictionaries to allow value access in dot notation.

    Instead of data[key], access value as data.key"""

    def __init__(self, data: dict):
        super().__init__()
        for k, v in data.items():
            if isinstance(v, dict):
                # take care of nested dicts
                v = Dotdict(v)
            self.__dict__[k] = v


def setup_wandb(
    config: Dotdict,
) -> Tuple[wandb.run, lightning.pytorch.loggers.WandbLogger, Dotdict]:
    """Sets up wandb logging for a training run.

    This will run wandb.init with arguments based on the config and store
    the used config on disk.

    Args:
        config: config of the training run.

    Returns:
        A tuple consisting of the wandb run, the lightning logger, and the updated config.

    Side-effects:
        Stores the config used to initialize the wandb run to the run directory.
    """

    os.environ["WANDB_CACHE_DIR"] = config.wandb.cache_dir

    run = wandb.init(
        mode=config.wandb.mode,
        entity=config.wandb.entity,
        project=config.wandb.project,
        dir=config.wandb.experiment_dir,
    )
    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        log_model=config.wandb.log_model,
        config=config,
        experiment=run,
        dir=run.dir,
    )

    config.__dict__.update(
        wandb.config
    )  # when using a wandb sweep, the wandb agent might update some params

    # upload up-to-date config to wandb
    wandb.config["setup_config"] = update_configs(config)

    if config.verbose:
        print(run.dir)
    with open(os.path.join(run.dir, "updated_setup_configs.yml"), "w") as outfile:
        yaml.dump(wandb.config["setup_config"], outfile, default_flow_style=False)

    return run, wandb_logger, config


def get_datamodule(
    config: Dotdict,
) -> Tuple[lightning.pytorch.LightningDataModule, Dotdict]:
    """Creates the lightning datamodule for the dataset defined in the config.

    Args:
        config: the training run config.

    Returns:
        a tuple of the lightning datamodule for a dataset and the latest config.
    """

    # get the correct datamodule and dataset objects
    datamodule = src.datamodules.__dict__[config.data.datamodule]
    dataset = src.datasets.__dict__[config.data.datamodule.replace("DataModule", "")]

    # scale images to expected size and standardize
    if (
        "benge" in config.data.datamodule.lower()
        or "treesatai" in config.data.datamodule.lower()
    ):
        if config.data.modality == "s1":
            datamodule.mean = datamodule.s1_mean
            datamodule.std = datamodule.s1_std
        elif config.data.modality == "s2":
            datamodule.mean = datamodule.s2_mean
            datamodule.std = datamodule.s2_std
        elif config.data.modality == "aerial":
            datamodule.mean = datamodule.aerial_mean
            datamodule.std = datamodule.aerial_std
        else:
            raise AttributeError()

    # ensure bands are correctly selected for multi-spectral data
    band_idx = range(len(datamodule.mean))
    if "eurosat" in config.data.datamodule.lower():
        band_idx = []
        for b in config.data.bands:
            band_idx.append(dataset.all_band_names.index(b))
        if config.verbose:
            print(f"Band indices: {band_idx=}")

    data_keys = ["image"]
    if config.task == "segmentation":
        data_keys.append("mask")

    if config.verbose:
        print(f"Augmentation keys: {data_keys=}")

    additional_transforms = torchgeo.transforms.AugmentationSequential(
        kornia.augmentation.Normalize(
            mean=datamodule.mean[band_idx],
            std=datamodule.std[band_idx],
            keepdim=True,
        ),
        kornia.augmentation.Resize(
            (config.data.img_size, config.data.img_size), keepdim=True
        ),
        data_keys=data_keys,
    )

    root = config.data.root
    if config.verbose:
        print(f"Dataset root directory: {root=}")

    # handle directory structure of different datasets
    if hasattr(src.datamodules.__dict__[config.data.datamodule], "folder"):
        root = f"data/{src.datamodules.__dict__[config.data.datamodule].folder}"

    # initialze few-shot datamodule with limited number of samples
    if config.data.few_shot_k is not None:
        if "eurosat" in config.data.datamodule.lower():
            datamodule = datamodule(
                root=root,
                bands=config.data.bands,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                train_split_file_suffix=f"-k{config.data.few_shot_k}-seed{config.data.few_shot_seed}.txt",
                transforms=additional_transforms,
            )
        elif "treesatai" in config.data.datamodule.lower():
            # no few-shot split defined yet for TreeSatAI
            raise NotImplementedError()
        else:
            datamodule = datamodule(
                root=root,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                train_split_file_suffix=f"-k{config.data.few_shot_k}-seed{config.data.few_shot_seed}.txt",
                transforms=additional_transforms,
            )
    else:
        # initialze the full dataset
        if "eurosat" in config.data.datamodule.lower():
            datamodule = datamodule(
                root=root,
                bands=config.data.bands,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                transforms=additional_transforms,
            )
        elif "treesatai" in config.data.datamodule.lower():
            datamodule = datamodule(
                root=root,
                modality=config.data.modality,
                bands=config.data.bands,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                transforms=additional_transforms,
                size=config.data.size,
            )
        else:
            datamodule = datamodule(
                root=root,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                transforms=additional_transforms,
            )

    datamodule.setup("fit")
    config.data.num_classes = len(datamodule.train_dataset.classes)
    config.data.in_chans = datamodule.train_dataset[0]["image"].shape[0]

    return datamodule, config


def get_callbacks(
    dir: str,
) -> Tuple[
    lightning.pytorch.callbacks.ModelCheckpoint,
    lightning.pytorch.callbacks.EarlyStopping,
    lightning.pytorch.callbacks.LearningRateMonitor,
]:
    """Initialze lightning callbacks for checkpointing, early stopping and LR monitoring.

    Args:
        dir: a directory where model checkpoints will be stored.

    Returns:
        a tuple of the three callback objects.
    """

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        # dirpath=args.experiment_dir,
        dirpath=dir,
        save_top_k=1,  # save best
        save_last=True,
    )
    early_stopping_callback = lightning.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
    )

    lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    return checkpoint_callback, early_stopping_callback, lr_monitor


def get_ckpt_path_from_wandb_run(
    config: Dotdict, run_id_key: str = "continual_pretrain_run", state: str = "best"
):
    """Returns the path to a model checkpoint associated with a wandb run.

    Args:
        config: config of the wandb/training run.
        run_id_key: type of the wandb run.
        state: best or latest checkpoint.

    Returns:
        path to the checkpoint
    """

    run_path = glob.glob(
        os.path.join("logs/mae", "wandb", f"*{getattr(config, run_id_key)}")
    )
    if not len(run_path):
        # try other wandb-project
        run_path = glob.glob(
            os.path.join("logs/lin_eval", "wandb", f"*{getattr(config, run_id_key)}")
        )
    if not len(run_path):
        run_path = glob.glob(
            os.path.join("logs/seg", "wandb", f"*{getattr(config, run_id_key)}")
        )
    assert len(run_path) == 1, f"{run_path=}"
    run_path = run_path[0]

    if state == "best":
        best_ckpt = glob.glob(os.path.join(run_path, "files", "epoch=*.ckpt"))[0]
    elif state == "last":
        best_ckpt = os.path.join(run_path, "files", "last.ckpt")
    else:
        raise ValueError(f"{state} not in [best, last]")

    return best_ckpt


def assert_model_compatibility(
    pretrain_config: Dotdict, downstream_config: Dotdict, ignore: list = []
):
    """Performs some checks to ensure pre-training run and downstream tasks are compatible.

    Args:
        pretrain_config: config of the pretraining run.
        downstream_config: config of the downstream task.
        ignore: list of checks to be skipped.

    Returns:
        True if all checks passed.

    Raises:
        AssertionError if any check fails.
    """

    if not "model" in ignore:
        assert (
            pretrain_config["model"] == downstream_config.model.name
        ), f"{pretrain_config['model']=}, {downstream_config.model.name=}"
    assert pretrain_config["in_channels"] == downstream_config.data.in_chans
    if not "embed_dim" in ignore:
        assert pretrain_config["embed_dim"] == downstream_config.model.embed_dim
    assert pretrain_config["input_size"] == downstream_config.data.img_size
    assert pretrain_config["patch_size"] == downstream_config.model.patch_size
    assert pretrain_config["adapter"] == downstream_config.model.adapter
    assert (
        pretrain_config["adapter_type"] == downstream_config.model.adapter_type
    ), f"{pretrain_config['adapter_type']=}, {downstream_config.model.adapter_type=}"
    assert pretrain_config["adapter_shared"] == downstream_config.model.adapter_shared
    assert pretrain_config["adapter_scale"] == downstream_config.model.adapter_scale
    assert (
        pretrain_config["adapter_hidden_dim"]
        == downstream_config.model.adapter_hidden_dim
    ), f"{pretrain_config['adapter_hidden_dim']=}, {downstream_config.model.adapter_hidden_dim=}"
    assert (
        pretrain_config["patch_embed_adapter"]
        == downstream_config.model.patch_embed_adapter
    ), f"{pretrain_config['patch_embed_adapter']=}, {downstream_config.model.patch_embed_adapter=}"
    assert (
        pretrain_config["patch_embed_adapter_scale"]
        == downstream_config.model.patch_embed_adapter_scale
    )

    return True


def get_config_from_wandb_run(
    config: Dotdict,
    run_id_key: str = "continual_pretrain_run",
    return_ckpt_path: bool = False,
) -> Dotdict:
    """Get the config associated with a finished wandb run.

    Args:
        config: the config for the run of interest.
        run_id_key: the type of the run.
        return_ckpt_path: if the path to the checkout should also be returned.

    Returns:
        the config, or a tuple of config and checkpoint path.
    """

    ckpt_path = get_ckpt_path_from_wandb_run(config, run_id_key=run_id_key)
    ckpt = torch.load(ckpt_path)

    args = copy.deepcopy(ckpt["hyper_parameters"])
    del ckpt

    if return_ckpt_path:
        return args, ckpt_path
    return args


def load_weights_from_wandb_run(
    model: torch.nn.Module,
    config: Dotdict,
    prefix: str = None,
    run_id_key: str = "continual_pretrain_run",
    return_ckpt: bool = False,
    which_state: str = "best",
):
    """Load weights from a finished wandb run into a model object.

    Args:
        model: the torch model.
        config: the config of the finished run.
        prefix: prefix in model layer names that wasn't present in the pre-training run.
        run_id_key: type of the pre-training run.
        return_ckpt: if the checkpoint is returned as well.
        which_state: best or latest checkpoint will be used.

    Returns:
        the model initialzed with weights from ´config´, or a tuple with the checkpoint.
    """

    best_ckpt = get_ckpt_path_from_wandb_run(
        config,
        run_id_key=run_id_key,
        state=which_state,
    )
    print(f"Loading checkpoint {best_ckpt=}...")

    ckpt = torch.load(best_ckpt)
    state = ckpt["state_dict"]

    # remove prefix from state dict keys
    for k in list(state.keys()):
        state[k.replace("model.", "")] = state[k]
        del state[k]

    if "cls_token" in model.state_dict() and not "cls_token" in state.keys():
        state["cls_token"] = model.state_dict()["cls_token"]

    if "pos_embed" in model.state_dict() and not "pos_embed" in state.keys():
        state["pos_embed"] = model.state_dict()["pos_embed"]

    if prefix is not None:
        for k in list(state.keys()):
            state[prefix + k] = state[k]
            del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing weights in pre-training: {missing=}")
    print(
        f"Unexpected weights (except decoder): {[k for k in unexpected if not 'decoder' in k]}"
    )

    if return_ckpt:
        return model, ckpt

    return model


def assert_run_validity(row: dict, config: Dotdict, idx: int) -> bool:
    """Checks if a config defines a valid run (no invalid combination of args).

    Args:
        row: the row in the run_id file that is being checked.
        config: the training run config.
        idx: the index of the current run in the run_id file.

    Returns:
        True if the run is valid.

    Raises:
        ValueError if there is an invalid combination of configurations.
    """

    if row["mode"] == "lin_eval":
        assert (
            config.model.adapter == False
        ), f"{idx=}, {row.run_id=}, {config.model.adapter=}"
        # assert (
        #     config.model.patch_embed_adapter == False
        # ), f"{idx=}, {row.run_id=}, {config.model.patch_embed_adapter=}"
        if hasattr(config.model, "norm_trainable"):
            assert (
                config.model.norm_trainable == False
            ), f"{idx=}, {row.run_id=}, {config.norm_trainable=}"
    elif row["mode"] == "lin_eval_slr":
        assert config.model.adapter, f"{idx=}, {row.run_id=}, {config.model.adapter=}"
        assert (
            config.model.adapter_trainable == False
        ), f"{idx=}, {row.run_id=}, {config.model.adapter_trainable=}"
        if hasattr(config.model, "norm_trainable"):
            assert (
                config.model.norm_trainable == False
            ), f"{idx=}, {row.run_id=}, {config.model.norm_trainable=}"
    elif row["mode"] == "slr_ft":
        assert config.continual_pretrain_run is not None
        assert config.model.adapter, f"{idx=}, {row.run_id=}, {config.model.adapter=}"
        assert (
            config.model.adapter_trainable
        ), f"{idx=}, {row.run_id=}, {config.model.adapter_trainable=}"
        assert (
            config.model.norm_trainable
        ), f"{idx=}, {row.run_id=}, {config.model.norm_trainable=}"
    elif row["mode"] == "ft":
        assert config.continual_pretrain_run is None
        assert not config.model.adapter
        assert config.model.train_all_params
    elif row["mode"] == "slr_scale":
        assert config.continual_pretrain_run is not None
        assert config.model.adapter
        assert not config.model.adapter_trainable
        assert config.model.only_scaler_trainable
    elif row["mode"] == "slr_full_ft":
        assert config.model.train_all_params
        assert config.continual_pretrain_run is not None
        assert config.model.adapter
    else:
        raise ValueError(f"{row['mode']=}, {row.run_id=}")

    if row.dataset == "eurosat":
        assert (
            config.data.datamodule == "EuroSATDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset in ["benge_s1_c", "benge_s1_seg"]:
        assert (
            config.data.datamodule == "BENGEDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "resisc45":
        assert (
            config.data.datamodule == "RESISC45DataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "firerisk":
        assert (
            config.data.datamodule == "FireRiskDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "treesatai":
        assert (
            config.data.datamodule == "TreeSatAIDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "ucmerced":
        assert (
            config.data.datamodule == "UCMercedDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "eurosat_sar":
        assert (
            config.data.datamodule == "EuroSATSARDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "caltech256":
        assert (
            config.data.datamodule == "Caltech256DataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    else:
        raise ValueError(f"{row.dataset=}, {row.run_id=}")

    if "seg" in row.dataset:
        assert (
            row.model == config.model.backbone
        ), f"{row.model=}, {config.model.backbone=}"
    else:
        assert row.model == config.model.name, f"{row.model=}, {config.model.name=}"

    return True


# else:
#    raise ValueError(f"{row.dataset=}, {row.run_id=}")
#
#    if "seg" in row.dataset:
#        assert (
#            row.model == config.model.backbone
#        ), f"{row.model=}, {config.model.backbone=}"
#    else:
#        assert row.model == config.model.name, f"{row.model=}, {config.model.name=}"

#    return True

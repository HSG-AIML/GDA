import os


os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import hydra
from omegaconf import OmegaConf
import torch
from lightning.pytorch import Trainer
import wandb

import src.utils as utils
from src.trainers.linear_eval import LinearEvaluationTask
from src.trainers.knn_eval import KNNEval


@hydra.main(version_base=None, config_path="configs/lin_eval", config_name="experiment")
def main(cfg):
    config = utils.Dotdict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    if "scale_mae" in config.model.name:
        assert hasattr(config.model, "input_res"), "input_res is required for scale-mae"
    if config.optim.use_lr_scheduler:
        assert config.val_every_n_epoch == 1

    run, wandb_logger, config = utils.setup_wandb(config)
    utils.set_seed(config.seed)
    datamodule, config = utils.get_datamodule(config)
    callbacks = utils.get_callbacks(run.dir)

    if config.continual_pretrain_run is not None:
        print(f"Reading config of pre-train run.. {config.continual_pretrain_run=}")
        pretrain_args = utils.get_config_from_wandb_run(config)
        utils.assert_model_compatibility(pretrain_args, config)

    if config.resume is not None:
        print(f"Reading config of earlier run.. {config.resume=}")
        pretrain_args, ckpt_path = utils.get_config_from_wandb_run(
            config, run_id_key="resume", return_ckpt_path=True
        )
        utils.assert_model_compatibility(pretrain_args, config, ignore=["embed_dim"])

    task = LinearEvaluationTask(
        model=config.model.name,
        model_type=config.model.type,
        num_classes=config.data.num_classes,
        in_channels=config.data.in_chans,
        input_size=config.data.img_size,
        patch_size=config.model.patch_size,
        loss="ce",
        lr=config.optim.lr,
        head_lr=config.optim.head_lr,
        use_lr_scheduler=config.optim.use_lr_scheduler,
        patience=config.optim.lr_schedule_patience,
        freeze_backbone=config.model.freeze_backbone,
        pretrained=config.model.pretrained,
        callbacks=callbacks,
        input_res=config.model.input_res,
        adapter=config.model.adapter,
        adapter_scale=config.model.adapter_scale,
        adapter_shared=config.model.adapter_shared,
        adapter_type=config.model.adapter_type,
        adapter_hidden_dim=config.model.adapter_hidden_dim,
        patch_embed_adapter=config.model.patch_embed_adapter,
        train_patch_embed=config.model.train_patch_embed,
        patch_embed_adapter_scale=config.model.patch_embed_adapter_scale,
        train_all_params=config.model.train_all_params,
        train_cls_mask_tokens=config.model.train_cls_mask_tokens,
        fixed_output_size=config.model.fixed_output_size,
        adapter_trainable=config.model.adapter_trainable,
        norm_trainable=config.model.norm_trainable,
        only_scaler_trainable=config.model.only_scaler_trainable,
        only_bias_trainable=config.model.only_bias_trainable,
    )

    if config.continual_pretrain_run is not None:
        print(f"Loading weights from pre-train run...")
        task.model = utils.load_weights_from_wandb_run(task.model, config)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if hasattr(config.optim, "max_steps") and config.optim.max_steps is not None:
        trainer = Trainer(
            fast_dev_run=config.wandb.fast_dev_run,
            # callbacks=[checkpoint_callback, early_stopping_callback], these will be overridden by callbacks in the task
            logger=[wandb_logger],
            default_root_dir=config.wandb.experiment_dir,
            min_steps=config.optim.min_steps,
            max_steps=config.optim.max_steps,
            accelerator=accelerator,
            log_every_n_steps=1,
            check_val_every_n_epoch=config.val_every_n_epoch,
        )
    elif hasattr(config.optim, "max_epochs") and config.optim.max_epochs is not None:
        trainer = Trainer(
            fast_dev_run=config.wandb.fast_dev_run,
            # callbacks=[checkpoint_callback, early_stopping_callback], these will be overridden by callbacks in the task
            logger=[wandb_logger],
            default_root_dir=config.wandb.experiment_dir,
            min_epochs=config.optim.min_epochs,
            max_epochs=config.optim.max_epochs,
            accelerator=accelerator,
            log_every_n_steps=1,
            check_val_every_n_epoch=config.val_every_n_epoch,
        )

    config.model.params = sum([p.numel() for p in task.model.parameters()])
    config.model.trainable_params = sum(
        [p.numel() for p in task.model.parameters() if p.requires_grad]
    )
    wandb.config["params"] = config.model.params
    wandb.config["trainable_params"] = config.model.trainable_params

    print("Trainable parameters:")
    for n, p in task.model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)

    # _ = trainer.fit(model=task, datamodule=datamodule)
    if config.resume is not None:
        trainer.fit(
            model=task,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=ckpt_path,
        )
    else:
        trainer.fit(
            model=task,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
        )

    print(
        f"Eval performance: {trainer.test(model=task, dataloaders=datamodule.val_dataloader())}"
    )

    if config.knn.knn_eval:
        knn = KNNEval(
            task.model,
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            k=config.knn.knn_k,
        )
        print(f"Fitting knn model with {config.knn.knn_k=}")
        knn_stats = knn.fit_eval()

        print(f"{knn_stats=}")
        wandb.log(knn_stats)

    updated_configs = {}
    for k, v in config.__dict__.items():
        if isinstance(v, utils.Dotdict):
            updated_configs[k] = v.__dict__
        else:
            updated_configs[k] = v
    wandb.config["final_configs"] = updated_configs


if __name__ == "__main__":
    main()

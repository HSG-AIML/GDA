"""Main script for step 1 (see Fig. 1) of the training pipeline.
Self-supervised training of adapter parameters on the target domain.

Use this script to:
    * load a pre-trained visual foundation model
    * initialize adapter parameters
    * train adapter on a reconstruction objective.
"""

import hydra
import lightning.pytorch
from omegaconf import OmegaConf
import torch
import wandb

import src.trainers
import src.utils

src.utils.set_resources(
    num_threads=4, wand_cache_dir="./cache/"
)


@hydra.main(version_base=None, config_path="configs/mae", config_name="experiment")
def main(cfg):
    config = src.utils.Dotdict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    if config.model.name == "sat_mae":
        assert config.model.type == "mae"
    elif "scale_mae" in config.model.name:
        assert hasattr(
            config.model, "input_res"
        ), "input_res is required for config.model=scale-mae"

    run, wandb_logger, config = src.utils.setup_wandb(config)
    src.utils.set_seed(
        config.seed
    )  # after setup_wandb in case seed is provided by wandb sweep
    datamodule, config = src.utils.get_datamodule(config)
    callbacks = src.utils.get_callbacks(run.dir)

    assert config.model.loss_on_all_patches, f"{config.model.loss_on_all_patches=}"
    task = src.trainers.MaskedAutoencoding(
        model=config.model.name,
        model_type=config.model.type,
        num_classes=config.data.num_classes,
        in_channels=config.data.in_chans,
        input_size=config.data.img_size,
        patch_size=config.model.patch_size,
        lr=config.optim.lr,
        warmup_epochs=config.optim.warmup_epochs,
        mask_ratio=config.model.mask_ratio,
        freeze_backbone=config.model.freeze_backbone,
        pretrained=config.model.pretrained,
        loss_on_all_patches=config.model.loss_on_all_patches,
        callbacks=callbacks,
        input_res=config.model.input_res,
        target_res=config.model.input_res,
        adapter=config.model.adapter,
        adapter_scale=config.model.adapter_scale,
        adapter_hidden_dim=config.model.adapter_hidden_dim,
        adapter_type=config.model.adapter_type,
        adapter_shared=config.model.adapter_shared,
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

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = lightning.pytorch.Trainer(
        fast_dev_run=config.wandb.fast_dev_run,
        # callbacks=[checkpoint_callback, early_stopping_callback], these will be overridden by callbacks in the task
        logger=[wandb_logger],
        # default_root_dir=args.experiment_dir,
        default_root_dir=run.dir,
        # min_epochs=config.min_epochs,
        # max_epochs=config.max_epochs,
        min_steps=config.optim.min_steps,
        max_steps=config.optim.max_steps,
        accelerator=accelerator,
        log_every_n_steps=1,
    )

    # collect number of model parameters for logging
    config.model.params = sum([p.numel() for p in task.model.parameters()])
    config.model.trainable_params = sum(
        [p.numel() for p in task.model.parameters() if p.requires_grad]
    )
    wandb.config["params"] = config.model.params
    wandb.config["trainable_params"] = config.model.trainable_params

    if config.verbose:
        print("Trainable parameters:")
        for n, p in task.model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)

    # start model training
    trainer.fit(
        model=task,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    # run knn eval
    if config.knn.knn_eval:
        knn = src.trainers.KNNEval(
            task.model,
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            k=config.knn.knn_k,
        )
        if config.verbose:
            print(f"Fitting knn model with {config.knn.knn_k=}")
        knn_stats = knn.fit_eval()

        if config.verbose:
            print(f"{knn_stats=}")
        wandb.log(knn_stats)

    wandb.config["final_configs"] = src.utils.update_configs(config)


if __name__ == "__main__":
    main()

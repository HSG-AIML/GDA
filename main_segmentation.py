"""Main script for segmentation tasks."""

import hydra
import lightning.pytorch
import omegaconf
import torch
import wandb

import src.utils
import src.trainers.knn_eval
import src.trainers.segmentation

src.utils.set_resources(num_threads=4)


@hydra.main(version_base=None, config_path="configs/seg", config_name="experiment")
def main(cfg):
    config = src.utils.Dotdict(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
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

    if config.continual_pretrain_run is not None:
        pretrain_args = src.utils.get_config_from_wandb_run(config)
        src.utils.assert_model_compatibility(pretrain_args, config, ignore=["model"])

    task = src.trainers.segmentation.SegmentationTrainer(
        segmentation_model=config.model.name,
        model=config.model.backbone,
        model_type=config.model.backbone_type,
        feature_map_indices=config.model.feature_map_indices,
        aux_loss_factor=config.optim.aux_loss_factor,
        num_classes=config.data.num_classes,
        in_channels=config.data.in_chans,
        loss="ce",
        lr=config.optim.lr,
        input_size=config.data.img_size,
        patch_size=config.model.patch_size,
        patience=config.optim.lr_schedule_patience,
        freeze_backbone=config.model.freeze_backbone,
        pretrained=config.model.pretrained,
        callbacks=callbacks,
        input_res=config.model.input_res,
        adapter=config.model.adapter,
        adapter_hidden_dim=config.model.adapter_hidden_dim,
        norm_trainable=config.model.norm_trainable,
        adapter_scale=config.model.adapter_scale,
        adapter_shared=config.model.adapter_shared,
        fixed_output_size=config.model.fixed_output_size,
        adapter_type=config.model.adapter_type,
        patch_embed_adapter=config.model.patch_embed_adapter,
        use_mask_token=config.model.use_mask_token,
        train_patch_embed=config.model.train_patch_embed,
        patch_embed_adapter_scale=config.model.patch_embed_adapter_scale,
        train_all_params=config.model.train_all_params,
        train_cls_mask_tokens=config.model.train_cls_mask_tokens,
        adapter_trainable=config.model.adapter_trainable,
        only_bias_trainable=config.model.only_bias_trainable,
        only_scaler_trainable=config.model.only_scaler_trainable,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    bb = "backbone."
    if config.model.name == "upernet":
        bb = "vit_backbone."
    if config.continual_pretrain_run is not None:
        task.model = src.utils.load_weights_from_wandb_run(
            task.model,
            config,
            prefix=bb,
        )

    trainer = lightning.pytorch.Trainer(
        fast_dev_run=config.wandb.fast_dev_run,
        # callbacks=[checkpoint_callback, early_stopping_callback], these will be overridden by callbacks in the task
        logger=[wandb_logger],
        default_root_dir=config.wandb.experiment_dir,
        # min_epochs=config.min_epochs,
        # max_epochs=config.max_epochs,
        min_steps=config.optim.min_steps,
        max_steps=config.optim.max_steps,
        accelerator=accelerator,
        log_every_n_steps=1,
    )

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

    trainer.fit(
        model=task,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    if config.verbose:
        print(
            f"Eval performance: {trainer.test(model=task, dataloaders=datamodule.val_dataloader())}"
        )

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

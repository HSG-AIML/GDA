"Main script for knn evaluation."

import functools

import torch
import wandb
import hydra
import omegaconf
import src.trainers.knn_eval
import src.utils
import src.trainers

src.utils.set_resources(num_threads=4)


@hydra.main(version_base=None, config_path="configs/knn", config_name="experiment")
def main(cfg):
    config = src.utils.Dotdict(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    _, _, config = src.utils.setup_wandb(config)
    src.utils.set_seed(config.seed)
    datamodule, config = src.utils.get_datamodule(config)

    if config.continual_pretrain_run is not None:
        pretrain_args = src.utils.get_config_from_wandb_run(config)
        src.utils.assert_model_compatibility(pretrain_args, config)

    # create feature extraction model (which is the same as linear eval feature extraction model)
    task = src.trainers.linear_eval.LinearEvaluationTask(
        model=config.model.name,
        model_type=config.model.type,
        num_classes=config.data.num_classes,
        in_channels=config.data.in_chans,
        input_size=config.data.img_size,
        patch_size=config.model.patch_size,
        loss="ce",
        lr=config.optim.lr,
        patience=config.optim.lr_schedule_patience,
        freeze_backbone=config.model.freeze_backbone,
        pretrained=config.model.pretrained,
        callbacks=[],
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
        use_mask_token=config.model.use_mask_token,
    )

    model = task.model

    if config.continual_pretrain_run is not None:
        model = src.utils.load_weights_from_wandb_run(model, config)

    if config.model.name == "scale_mae":
        assert config.model.type == "mae", f"see "
        model.forward_features = functools.partial(
            model.forward,
            input_res=torch.tensor([config.model.input_res]),
            knn_feats=True,
        )

    knn = src.trainers.knn_eval.KNNEval(
        model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        k=config.knn_k,
    )
    if config.verbose:
        print(f"Fitting knn model with {config.knn_k=}")
    knn_stats = knn.fit_eval()

    if config.verbose:
        print(f"{knn_stats=}")
    wandb.log(knn_stats)

    if config.test:
        datamodule.setup("test")
        test_stats = knn.test(datamodule.test_dataloader())

    if config.verbose:
        print(f"{test_stats=}")
    wandb.log(test_stats)

    wandb.config["final_configs"] = src.utils.update_configs(config)


if __name__ == "__main__":
    main()

"""Trainers for image classification linear evaluation."""

from typing import Any, Optional

import numpy as np
import PIL
import torch
import torch.optim.lr_scheduler
import torchgeo
import torchgeo.trainers

import src.models
import src.utils

src.utils.set_resources(num_threads=4)


class MaskedAutoencoding(torchgeo.trainers.base.BaseTask):
    def __init__(
        self,
        model: str = "resnet50",
        model_type="",
        pretrained=True,
        num_classes=1000,
        weights=None,
        in_channels: int = 3,
        embed_dim=1024,
        input_size=224,
        patch_size=16,
        version: int = 2,
        layers: int = 3,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        lr: float = 4.8,
        gather_distributed: bool = False,
        size: int = 224,
        warmup_epochs=5,
        mask_ratio=0.75,
        loss_on_all_patches=False,
        input_res=None,
        target_res=None,
        callbacks=None,
        adapter=False,
        adapter_trainable=True,
        norm_trainable=True,
        adapter_type="lora",
        adapter_shared=True,
        adapter_scale=1.0,
        adapter_hidden_dim=8,
        patch_embed_adapter=False,
        patch_embed_adapter_scale=1.0,
        freeze_backbone=True,
        train_patch_embed=False,
        train_cls_mask_tokens=False,
        train_all_params=False,
        fixed_output_size=0,
        use_mask_token=True,
        only_scaler_trainable=False,
        only_bias_trainable=False,
    ) -> None:
        super().__init__()

        assert self.hparams["model"] in ["mae", "sat_mae", "scale_mae"]
        assert self.hparams["model_type"] == "mae"

    def configure_callbacks(self):
        return self.hparams["callbacks"]  # self.callbacks

    def configure_models(self):
        self.model = src.models.get_model(**self.hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.95)
        )
        max_epochs = 200
        if self.trainer and self.trainer.max_epochs:
            max_epochs = self.trainer.max_epochs
        warmup_epochs = self.hparams["warmup_epochs"]

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def forward(self, x):
        x = x.squeeze()
        if self.hparams["model"] == "sat_mae":
            loss, pred, mask = self.model(
                x, self.hparams["mask_ratio"], self.hparams["loss_on_all_patches"]
            )
        elif self.hparams["model"] == "mae":
            loss, pred, mask = self.model(x, self.hparams["mask_ratio"])
        elif self.hparams["model"] == "scale_mae":
            targets = torch.nn.functional.interpolate(x, 448)

            loss, y, mask, mean, var, pos_emb, pos_emb_decoder, samples = self.model(
                x,
                input_res=torch.tensor([self.hparams["input_res"]]),
                targets=targets,
                target_res=torch.tensor([self.hparams["target_res"]]),
                mask_ratio=self.hparams["mask_ratio"],
                source_size=self.hparams["input_size"],
            )
            pred = y[1]

        return loss, pred, mask

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"].squeeze()

        loss, pred, mask = self(x)

        self.log("train_loss", loss)

        if batch_idx % 100 == 0:
            x, y, im_masked, im_paste = self.get_reconstruction_images(
                x,
                pred,
                mask,
                p=self.model.patch_embed.patch_size[0],
                c=self.hparams["in_channels"],
            )
            self.logger.log_image(key="train_imgs", images=x)
            self.logger.log_image(key="train_predictions", images=y)
            self.logger.log_image(key="train_masked_imgs", images=im_masked)
            self.logger.log_image(key="train_imgs_paste", images=im_paste)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        with torch.no_grad():
            x = batch["image"].squeeze()
            loss, pred, mask = self(x)
        self.log("val_loss", loss)

        if batch_idx % 100 == 0:
            x, y, im_masked, im_paste = self.get_reconstruction_images(
                x,
                pred,
                mask,
                p=self.model.patch_embed.patch_size[0],
                c=self.hparams["in_channels"],
            )
            self.logger.log_image(key="val_imgs", images=x)
            self.logger.log_image(key="val_predictions", images=y)
            self.logger.log_image(key="val_masked_imgs", images=im_masked)
            self.logger.log_image(key="val_imgs_paste", images=im_paste)
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def PIL_imgs_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            # img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            if img.shape[-1] not in [3, 1]:
                img = img[:, :, [3, 2, 1]]  # S2 RGB
            img = img.detach().cpu().numpy()
            img /= img.max(axis=(0, 1))
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(PIL.Image.fromarray(img))

        return imgs

    def get_reconstruction_images(self, x, y, mask, p=16, c=3):
        y = self.model.unpatchify(y, p=p, c=c)
        y = torch.einsum("nchw->nhwc", y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(
            1, 1, self.model.patch_embed.patch_size[0] ** 2 * c
        )  # (N, H*W, p*p*3)
        mask = self.model.unpatchify(mask, p=p, c=c)  # 1 is removing, 0 is keeping
        mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

        x = torch.einsum("nchw->nhwc", x).cpu()

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        x = self.PIL_imgs_from_batch(x)
        y = self.PIL_imgs_from_batch(y)
        im_masked = self.PIL_imgs_from_batch(im_masked)
        im_paste = self.PIL_imgs_from_batch(im_paste)

        return x, y, im_masked, im_paste

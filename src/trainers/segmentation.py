"""Trainer for image segmentation."""

from typing import Any, Optional

import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)
import torchgeo
import torchgeo.trainers

import src.models
import src.models_segmentation
import src.utils

src.utils.set_resources(num_threads=4)


class SegmentationTrainer(torchgeo.trainers.SemanticSegmentationTask):
    def __init__(
        self,
        segmentation_model,
        model,
        model_type="",
        weights=None,
        feature_map_indices=(5, 11, 17, 23),
        aux_loss_factor=0.5,
        input_size=224,
        patch_size=16,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_filters: int = 3,
        loss: str = "ce",
        pretrained=True,
        input_res=10,
        adapter=False,
        adapter_trainable=True,
        adapter_shared=False,
        adapter_scale=1.0,
        adapter_type="lora",
        adapter_hidden_dim=16,
        norm_trainable=True,
        fixed_output_size=0,
        use_mask_token=False,
        train_patch_embed=False,
        patch_embed_adapter=False,
        patch_embed_adapter_scale=1.0,
        train_all_params=False,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        lr: float = 1e-3,
        patience: int = 10,
        train_cls_mask_tokens=False,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        callbacks=None,
        only_scaler_trainable=False,
        only_bias_trainable=False,
    ) -> None:
        super().__init__()

    def configure_callbacks(self):
        return self.hparams["callbacks"]  # self.callbacks

    def configure_models(self):
        backbone = src.models.get_model(**self.hparams)

        # add segmentation head
        if self.hparams["segmentation_model"] == "fcn":
            self.model = src.models_segmentation.ViTWithFCNHead(
                backbone,
                num_classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "upernet":
            self.model = src.models_segmentation.UPerNetWrapper(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
            )
        else:
            raise NotImplementedError(
                f"`model` must be in [fcn, upernet], not {self.hparams['model']}"
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_aux_metrics = metrics.clone(prefix="train_aux_")
        self.val_aux_metrics = metrics.clone(prefix="val_aux_")
        self.test_aux_metrics = metrics.clone(prefix="test_aux_")

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux, y)
            self.log("train_aux_loss", loss_aux)
            self.train_aux_metrics(y_aux_hard, y)
            self.log_dict(self.train_aux_metrics)
        else:
            y_hat = self(x)

        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("train_loss", loss)
        self.train_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.train_metrics)

        if batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="train_imgs",
                images=imgs,
            )
            self.logger.log_image(key="train_preds", images=pred_imgs)
            self.logger.log_image(key="train_targets", images=target_imgs)

        if self.model.deepsup:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux, y)
            self.log("val_aux_loss", loss_aux)
            self.val_aux_metrics(y_aux_hard, y)
            self.log_dict(self.val_aux_metrics)
        else:
            y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("val_loss", loss)
        self.val_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.val_metrics)

        if batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="val_imgs",
                images=imgs,
            )
            self.logger.log_image(key="val_preds", images=pred_imgs)
            self.logger.log_image(key="val_targets", images=target_imgs)

        # log some figures
        if False:
            #  (
            #  batch_idx < 10
            #  and hasattr(self.trainer, "datamodule")
            #  and self.logger
            #  and hasattr(self.logger, "experiment")
            #  and hasattr(self.logger.experiment, "add_figure")
            # ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = torchgeo.datasets.utils.unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux, y)
            self.log("test_aux_loss", loss_aux)
            self.test_aux_metrics(y_aux_hard, y)
            self.log_dict(self.test_aux_metrics)
        else:
            y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.test_metrics)

        if False:  # batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="test_imgs",
                images=imgs,
            )
            self.logger.log_image(key="test_preds", images=pred_imgs)
            self.logger.log_image(key="test_targets", images=target_imgs)

        if self.model.deepsup:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        if self.model.deepsup:
            y_hat, _ = self(x)
            y_hat = y_hat.softmax(dim=q)
        else:
            y_hat: torch.Tensor = self(x).softmax(dim=1)
        return y_hat

    def PIL_imgs_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            # assert img.shape[-1] == 3
            if img.shape[-1] not in [3, 1]:
                img = img[:, :, [3, 2, 1]]  # S2 RGB
            # img = img.detach().cpu().numpy()
            img /= img.max(axis=(0, 1))
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(PIL.Image.fromarray(img))

        return imgs

    def PIL_masks_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            # img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            assert len(img.shape) == 2 or img.shape[-1] == 1, f"{img.shape=}"
            assert img.min() >= 0
            assert img.max() <= 255
            img = img.detach().cpu().numpy()
            img = img.astype(np.uint8) * (255 // self.hparams["num_classes"])
            imgs.append(PIL.Image.fromarray(img, mode="P"))

        return imgs

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for image classification linear evaluation."""

from functools import partial
from typing import Any, Optional

import torch
import torch.optim
import torch.optim.lr_scheduler
import torchgeo
import torchgeo.trainers

import src.models
import src.utils

src.utils.set_resources(num_threads=4)


class LinearEvaluationTask(torchgeo.trainers.ClassificationTask):
    def __init__(
        self,
        model: str = "sat_mae",
        model_type: str = "",
        in_channels: int = 3,
        input_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        loss: str = "ce",
        class_weights: Optional[torch.Tensor] = None,
        pretrain_checkpoint=None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        pretrained=False,
        callbacks=None,
        input_res=None,
        adapter=False,
        adapter_scale=1.0,
        adapter_shared=True,
        adapter_type="lora",
        adapter_hidden_dim=8,
        adapter_trainable=True,
        norm_trainable=True,
        patch_embed_adapter=False,
        patch_embed_adapter_scale=1.0,
        train_patch_embed=False,
        train_all_params=False,
        train_cls_mask_tokens=False,
        fixed_output_size=0,
        use_mask_token=0,
        head_lr=None,
        use_lr_scheduler=True,
        monitor_on="val_loss",
        only_scaler_trainable=False,
        only_bias_trainable=False,
    ) -> None:
        # self.callbacks = callbacks
        super().__init__()

    def configure_callbacks(self):
        return self.hparams["callbacks"]  # self.callbacks

    def configure_models(self):
        self.model = src.models.get_model(**self.hparams)
        if self.hparams["model"] in ["sat_mae", "mae"]:
            self.model.head = torch.nn.Linear(1024, self.hparams["num_classes"])

        elif self.hparams["model"] == "scale_mae":
            # fix the input resolution
            self.model.forward = partial(
                self.model.forward, input_res=torch.tensor([self.hparams["input_res"]])
            )

        if hasattr(self.model, "head"):
            for p in self.model.head.parameters():
                p.requires_grad = True

    def configure_optimizers(self):
        "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig"
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        parameters = self.parameters()

        if self.hparams["head_lr"] is not None:
            parameters = [
                {"params": self.model.head.parameters(), "lr": self.hparams["head_lr"]},
                {
                    "params": [
                        v for k, v in self.named_parameters() if not "head" in k
                    ],
                    "lr": self.hparams["lr"],
                },
            ]
            for group in parameters:
                # group["params"] = [p for p in group["params"] if p.requires_grad]
                print(f"{sum([p.numel() for p in group['params']])=:,}")
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams["lr"])
        if self.hparams["use_lr_scheduler"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.hparams["patience"]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
            }
        else:
            return {"optimizer": optimizer}

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
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat_hard, y)
        self.log_dict(self.val_metrics)

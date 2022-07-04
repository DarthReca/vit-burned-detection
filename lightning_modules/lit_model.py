# Copyright 2022 Daniele Rege Cambrin

from itertools import chain
from typing import Any, Dict, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.utils as vutils

import utils


def compute_prec_recall_f1_acc(conf_matr):
    """Copyright Simone Monaco. Copy from https://github.com/dbdmg/rescue."""
    accuracy = np.trace(conf_matr) / conf_matr.sum()

    predicted_sum = conf_matr.sum(axis=0)  # sum along column
    gt_sum = conf_matr.sum(axis=1)  # sum along rows

    diag = np.diag(conf_matr)

    # Take into account possible divisions by zero
    precision = np.true_divide(
        diag, predicted_sum, np.full(diag.shape, np.nan), where=predicted_sum != 0
    )
    recall = np.true_divide(
        diag, gt_sum, np.full(diag.shape, np.nan), where=gt_sum != 0
    )
    num = 2 * (precision * recall)
    den = precision + recall
    f1 = np.true_divide(num, den, np.full(num.shape, np.nan), where=den != 0)
    return precision, recall, f1, accuracy


class LitModel(pl.LightningModule):
    def __init__(self, classes: int, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.model = utils.config_to_object(
            "neural_net",
            self.hparams["model"]["name"],
            self.hparams["model"]["parameters"],
        )
        if "weights" in self.hparams["model"]:
            for k, v in self.hparams["model"]["weights"].items():
                state_dict = torch.load(v)
                getattr(self.model, k).load_state_dict(state_dict, strict=True)

        metrics = torchmetrics.MetricCollection(
            [
                utils.config_to_object("torchmetrics", m, p)
                for m, p in self.hparams["metrics"].items()
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        losses = self.hparams["losses"]
        self.classification_criterion = utils.config_to_object(
            "torch.nn",
            losses["classification"]["name"],
            losses["classification"]["parameters"],
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        configuration = {
            "optimizer": utils.config_to_object(
                "torch.optim",
                self.hparams["optimizer"]["name"],
                {
                    "params": self.model.parameters(),
                    "lr": self.hparams["lr"],
                    **self.hparams["optimizer"]["parameters"],
                },
            )
        }
        if "scheduler" in self.hparams:
            configuration["lr_scheduler"] = utils.config_to_object(
                "torch.optim.lr_scheduler",
                self.hparams["scheduler"]["name"],
                {
                    "optimizer": configuration["optimizer"],
                    **self.hparams["scheduler"]["parameters"],
                },
            )
            configuration["monitor"] = self.hparams["scheduler"].get("monitor", "")

        return configuration

    def forward(
        self, x: torch.Tensor, resize_to: Optional[Tuple[int]] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.model(x, **kwargs)
        logits = result[0] if isinstance(result, tuple) else result
        if resize_to:
            logits = F.interpolate(
                logits, size=resize_to[-2:], mode="bilinear", align_corners=False
            )
        norm_score = torch.sigmoid(logits)
        mask = torch.round(norm_score)
        return norm_score, mask, logits

    def calculate_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> float:
        # Select if normalized scores are needed
        selected_pred = logits
        if (
            not self.hparams["losses"]["classification"]["normalized_scores"]
            and scores is not None
        ):
            selected_pred = scores
        # Squeeze as needed
        return self.classification_criterion(selected_pred.squeeze(1), masks)

    def training_step(self, batch, batch_idx):
        masks, images = (
            batch["mask"].float(),
            batch["image"].float(),
        )
        logits, pred_mask, scores = self.forward(images, resize_to=masks.size())
        masks = masks.squeeze(1)
        loss = self.calculate_loss(logits, masks, scores)
        logits = torch.concat([1 - logits, logits], dim=1)
        self._log_metrics(self.train_metrics(logits, masks.long()))
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["image"].float()
        logits, pred_mask, scores = self.forward(images, resize_to=masks.size())
        masks = masks.squeeze(1)
        self.log("val_loss", self.calculate_loss(logits, masks, scores))
        # Log metrics
        logits = torch.concat([1 - logits, logits], dim=1)
        self._log_metrics(self.val_metrics(logits, masks.long()))
        # Log images
        self._log_images(images, masks, pred_mask, logits, log_limit=5)

    def test_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["image"].float()
        # Crop
        if "crop" in self.hparams and self.hparams["crop"]:
            cropped_images, cropped_positions = [], []
            for image in images:
                img, pos = utils.crop_image(image, 64)
                cropped_images.append(img)
                cropped_positions.append(pos)
            positions = list(chain(*cropped_positions))
            images = torch.concat(cropped_images)
            num_crops = (masks.size()[-1] // 64) ** 2

        confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2)
        logits, pred_mask, scores = self.forward(images, resize_to=images.size())

        # Recompose
        if "crop" in self.hparams and self.hparams["crop"]:
            rec_logits, rec_masks, rec_scores = [], [], []
            for i in range(0, logits.size()[0], num_crops):
                l, p, s = logits[i : i + 64], pred_mask[i : i + 64], scores[i : i + 64]
                pos = positions[i : i + 64]
                rec_logits.append(utils.recompose_image(l, pos))
                rec_masks.append(utils.recompose_image(p, pos))
                rec_scores.append(utils.recompose_image(s, pos))
            logits = torch.stack(rec_logits, 0)
            pred_mask = torch.stack(rec_masks, 0)
            scores = torch.stack(rec_scores, 0)

        masks = masks.squeeze(1)
        self.log("test_loss", self.calculate_loss(logits, masks, scores))
        logits = torch.concat([1 - logits, logits], dim=1)
        # Log metrics
        self._log_metrics(self.test_metrics(logits, masks.squeeze(1).long()))
        # Log images
        self._log_images(
            batch["image"].float(),
            masks,
            pred_mask,
            logits,
            draw_heatmap=True,
            draw_rgb=True,
        )
        return confusion_matrix(logits, masks.long())

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        acc_conf_mat = torch.zeros([2, 2])
        for d in outputs:
            acc_conf_mat += d

        p, r, f, a = compute_prec_recall_f1_acc(acc_conf_mat.numpy())
        self.log("test_prec", p[1])
        self.log("test_rec", r[1])
        self.log("test_f1", f[1])
        self.log("test_a", a)

    def _log_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        pred_mask: torch.Tensor,
        logits: torch.Tensor,
        prefix: str = "",
        draw_rgb: bool = False,
        draw_heatmap: bool = False,
        log_limit: int = 100,
    ):
        images_count = images.size()[0]
        indexes = (
            torch.randperm(images_count)[:log_limit]
            if images_count > log_limit
            else torch.arange(images_count)
        )
        for i in indexes:
            collection = {}
            gt = (masks[i].squeeze() > 0).byte().cpu()
            pr = (pred_mask[i].squeeze() > 0).byte().cpu()
            if draw_rgb:
                img = utils.extract_rgb(images[i], self.hparams["rgb_channels"]).cpu()
                collection["original image"] = (img, None)
                collection["ground truth with image"] = (
                    vutils.draw_segmentation_masks(img, gt.bool(), colors=["red"]),
                    None,
                )
                collection["prediction with image"] = (
                    vutils.draw_segmentation_masks(img, pr.bool(), colors=["red"]),
                    None,
                )
            if draw_heatmap:
                collection["prediction heatmap"] = (
                    logits[i][1].unsqueeze(0).cpu(),
                    "viridis",
                )
            collection["ground truth mask"] = (gt.unsqueeze(0), "gray")
            collection["prediction mask"] = (pr.unsqueeze(0), "gray")
            collection = {
                k: (v.permute(1, 2, 0).numpy(), cmap)
                for k, (v, cmap) in collection.items()
            }
            if hasattr(self.logger.experiment, "log_image"):
                figure, axs = plt.subplots(ncols=len(collection), figsize=(20, 20))
                figure.tight_layout()
                for ax, (k, (v, cmap)) in zip(axs, collection.items()):
                    ax.imshow(v, cmap=cmap)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_title(k, {"fontsize": 15})
                self.logger.experiment.log_figure(
                    figure=figure,
                    figure_name=f"{prefix}S{self.global_step}N{i}",
                    step=self.global_step,
                )
                plt.close()

    def _log_metrics(self, metrics: Dict[str, torch.Tensor]):
        for k, v in metrics.items():
            if v.numel() == 2:
                self.log(k + "_class0", v[0])
                self.log(k + "_class1", v[1])
            else:
                self.log(k, v)

"""Trainer to run KNN evaluation.

code based on https://gist.github.com/calebrob6/912c2509de9d94ad6bc924420eca40bb
"""

from typing import Tuple, Dict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

import src.utils

src.utils.set_resources(num_threads=4)


class KNNEval:
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        k: int = 5,
    ):
        self.feature_extractor = feature_extractor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.k = k
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k)
        self.scaler = StandardScaler()

    def get_features(
        self, loader: torch.utils.data.DataLoader, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        self.feature_extractor.to(device)
        self.feature_extractor.eval()
        print(f"Extracting features...")
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            labels = batch["label"].numpy()

            with torch.inference_mode():
                features = (
                    self.feature_extractor.forward_features(images)
                    .detach()
                    .cpu()
                    .numpy()
                )

            x.append(features)
            y.append(labels)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y

    def fit_eval(self, device: torch.device = torch.device("cuda")) -> Dict[str, float]:
        x_train, y_train = self.get_features(self.train_dataloader, device=device)
        x_eval, y_eval = self.get_features(self.val_dataloader, device=device)

        # get acc with unscaled features
        self.knn_model.fit(x_train, y_train)
        knn_train_acc = self.knn_model.score(x_train, y_train)
        knn_eval_acc = self.knn_model.score(x_eval, y_eval)

        # get acc for scaled features
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_eval_scaled = self.scaler.transform(x_eval)
        self.knn_model_scaled = KNeighborsClassifier(n_neighbors=self.k)
        self.knn_model_scaled.fit(x_train_scaled, y_train)
        knn_train_acc_scaled = self.knn_model_scaled.score(x_train_scaled, y_train)
        knn_eval_acc_scaled = self.knn_model_scaled.score(x_eval_scaled, y_eval)

        return {
            "knn_train_acc": knn_train_acc,
            "knn_eval_acc": knn_eval_acc,
            "knn_train_acc_scaled": knn_train_acc_scaled,
            "knn_eval_acc_scaled": knn_eval_acc_scaled,
        }

    def test(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda"),
    ) -> Dict[str, float]:
        x_test, y_test = self.get_features(dataloader, device=device)
        knn_test_acc = self.knn_model.score(x_test, y_test)

        x_test_scaled = self.scaler.transform(x_test)

        knn_test_acc = self.knn_model.score(x_test, y_test)
        knn_test_acc_scaled = self.knn_model_scaled.score(x_test_scaled, y_test)

        return {
            "knn_test_acc": knn_test_acc,
            "knn_test_acc_scaled": knn_test_acc_scaled,
        }

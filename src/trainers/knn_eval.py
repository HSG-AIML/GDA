# code based on https://gist.github.com/calebrob6/912c2509de9d94ad6bc924420eca40bb

import os


os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNEval:
    def __init__(self, feature_extractor, train_dataloader, val_dataloader, k=5):
        self.feature_extractor = feature_extractor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.k = k
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k)
        self.scaler = StandardScaler()

    def get_features(self, loader, device):
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

    def fit_eval(self, device=torch.device("cuda")):
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

    def test(self, dataloader, device=torch.device("cuda")):
        x_test, y_test = self.get_features(dataloader, device=device)
        knn_test_acc = self.knn_model.score(x_test, y_test)

        x_test_scaled = self.scaler.transform(x_test)

        knn_test_acc = self.knn_model.score(x_test, y_test)
        knn_test_acc_scaled = self.knn_model_scaled.score(x_test_scaled, y_test)

        return {
            "knn_test_acc": knn_test_acc,
            "knn_test_acc_scaled": knn_test_acc_scaled,
        }

import os
from importlib.resources import files
from pathlib import Path

import faiss
import hydra
import rootutils
from omegaconf import DictConfig

root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=True)
import os
from enum import Enum

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.neighbors._base import _get_weights
from sklearn.preprocessing import Normalizer
from sklearn.utils.multiclass import unique_labels


def get_results(ground_truth, pred, pred_proba):
    logits = pred_proba[:, 1]
    result = dict(
        auc=roc_auc_score(ground_truth, logits) * 100,
        apr=average_precision_score(ground_truth, logits) * 100,
        acc=np.mean((ground_truth == pred).astype(float)) * 100.0,
        loss=log_loss(ground_truth, logits),
    )
    return result


class Preprocess_Type(Enum):
    NONE = "NONE"
    NORM_SEPERATLY = "NORM_SEPERATLY"
    NORM_AFTER_CONCAT = "NORM_AFTER_CONCAT"


class KNN_Model(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        modalities: list[str],
        modality_weights: list[float],
        n_neighbors=5,
        algorithm="l2",
        weights="uniform",
        preprocess: Preprocess_Type = Preprocess_Type.NONE,
    ):
        self.modalities = modalities
        self.modality_weights = modality_weights
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.weights = weights
        self.preprocess = preprocess

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def kneighbors(self, X: np.array, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0." " Got {}".format(n_neighbors))
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take {} value, " "enter integer value".format(type(n_neighbors))
                )

        X = np.atleast_2d(X).astype(np.float32)
        dist, idx = self.index.search(X, n_neighbors)

        if return_distance:
            return dist, idx
        else:
            return idx

    def get_uniform_proba(self, idx):
        class_idx = self.y_[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_), axis=1, arr=class_idx.astype(np.int16)
        )
        return counts / self.n_neighbors

    def get_distance_proba(self, neigh_dist, neigh_ind):
        _y = self.y_
        # _y = _y.reshape((-1, 1))
        n_queries = neigh_dist.shape[0]

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(n_queries)
        # probabilities = []

        classes = self.classes_
        pred_labels = _y[neigh_ind]
        proba = np.zeros((n_queries, classes.size))

        # a simple ':' index doesn't work right
        for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
            proba[all_rows, idx] += weights[:, i]

        # normalize 'votes' into real [0,1] probabilities
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        proba /= normalizer
        return proba

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        X = self.transform_preprocess(X)
        if self.weights == "uniform":  # uses closest neighbors and treats all labels equally
            idx = self.kneighbors(X, self.n_neighbors, return_distance=False)
            return self.get_uniform_proba(idx)
        else:  # weights neighbor's labels by inverse distance to the neighbor
            assert self.weights == "distance"
            neigh_dist, neigh_ind = self.kneighbors(X, self.n_neighbors, return_distance=True)
            return self.get_distance_proba(neigh_dist, neigh_ind)

    def concat_data(self, X: list[np.array] | pl.DataFrame):
        embeddings = []
        if isinstance(X, pl.DataFrame):
            for modality in self.modalities:
                # embed_array = np.asarray(X[modality].to_list())
                column_data = X[modality]
                if isinstance(column_data.dtype, pl.List):
                    embed_array = np.asarray(column_data.to_list())
                else:
                    embed_array = np.expand_dims(np.asarray(column_data.to_list()), axis=1)
                embeddings.append(embed_array)
        else:
            embeddings = X
        return np.concatenate(embeddings, axis=1)

    def fit_preprocess(self, X):
        self.get_modality_lengths(X)
        if self.preprocess == Preprocess_Type.NORM_AFTER_CONCAT:
            X = self.concat_data(X)
            self.scaler = Normalizer()
            self.scaler.fit(X)
        elif self.preprocess == Preprocess_Type.NORM_SEPERATLY:
            self.scalers = []
            assert len(self.modalities) == len(self.modality_weights)
            for modality in self.modalities:
                embed_array = np.asarray(X[modality].to_list())
                scaler = Normalizer()
                scaler.fit(embed_array)
                self.scalers.append(scaler)
        else:
            assert self.preprocess == Preprocess_Type.NONE

    def get_modality_lengths(self, X: pl.DataFrame):
        modality_lengths = []
        for modality in self.modalities:
            if isinstance(X.get_column(modality)[0], (int, float)):
                modality_lengths.append(1)
            else:
                modality_lengths.append(X.get_column(modality)[0].len())
        self.modality_lengths = modality_lengths

    def transform_preprocess(self, X: pl.DataFrame):
        reweight_vector = np.repeat(self.modality_weights, self.modality_lengths)[None, :]
        if self.preprocess == Preprocess_Type.NORM_AFTER_CONCAT:
            X = self.concat_data(X)
            self.scaler = Normalizer()
            X = self.scaler.transform(X)
            # return X
        elif self.preprocess == Preprocess_Type.NORM_SEPERATLY:
            self.scalers = []
            transformed_x = []
            for modality in self.modalities:
                embed_array = np.asarray(X[modality].to_list())
                scaler = Normalizer()
                transformed_x.append(scaler.transform(embed_array))
            X = self.concat_data(transformed_x)
        else:
            assert self.preprocess == Preprocess_Type.NONE
            X = self.concat_data(X)

        return X * reweight_vector

    def fit(self, X: pl.DataFrame):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : polars dataframe of shape (n_samples, n_features)
            Data used to fit the model.
            Has modality columns as well as a label column
        """
        y = X.get_column("label")
        self.classes_ = unique_labels(y)
        self.fit_preprocess(X)
        X = self.transform_preprocess(X)
        d = X.shape[1]  # dimensionality of the feature vector
        self._prepare_knn_algorithm(X, d, False)
        self.index.add(X)
        self.y_ = np.array(y)
        self.n_classes_ = np.unique(y).size
        return self

    def _prepare_knn_algorithm(self, X, d, gpu_usage=False):
        """_summary_

        Args:
            X (_type_): _description_
            d (_type_): dimension of the embeddings
            gpu_usage (boolean): boolean for whether or not gpu should be used

        Raises:
            ValueError: _description_
        """
        if self.algorithm == "l2":
            index = faiss.IndexFlatL2(sum(self.modality_lengths))
            self.index = index
            if gpu_usage:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        elif self.algorithm == "ip":
            index = faiss.IndexFlatIP(d)
            self.index = index
            if gpu_usage:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        else:
            raise ValueError(
                "Invalid algorithm option." " Expected ['l2', 'ip'], " "got {}".format(self.algorithm)
            )

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]


config_yaml = files("meds_interp").joinpath("configs/knn.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    modality_weights = [cfg.weights[modality] for modality in cfg.modalities]
    knn = KNN_Model(
        modalities=cfg.modalities,
        modality_weights=modality_weights,
        n_neighbors=cfg.n_neighbors,
        algorithm=cfg.distance_metric,
        weights=cfg.neighbor_weighting,
        preprocess=Preprocess_Type[cfg.preprocess],
    )

    train_df = pl.read_parquet(Path(cfg.input_path) / "train.parquet")
    knn.fit(train_df)
    val_df = pl.read_parquet(Path(cfg.input_path) / "val.parquet")
    pred_labels = knn.predict(val_df)
    return roc_auc_score(val_df.get_column("label").to_numpy(), pred_labels)


if __name__ == "__main__":
    main()

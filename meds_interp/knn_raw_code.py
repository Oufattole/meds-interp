import os

import faiss
import rootutils

root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=True)
import os
from enum import Enum

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors._base import _get_weights
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

# from src.train.ebcl.ebcl_train_pl import EbclPretrainTuneConfig

# cfg = EbclPretrainTuneConfig()
# dataloader_creator = EBCLDataLoaderCreator(cfg.dataloader_config)
# train_dataloader, val_dataloader, test_dataloader = dataloader_creator.get_dataloaders()


def get_results(ground_truth, pred, pred_proba):
    logits = pred_proba[:, 1]
    result = dict(
        auc=roc_auc_score(ground_truth, logits) * 100,
        apr=average_precision_score(ground_truth, logits) * 100,
        acc=np.mean((ground_truth == pred).astype(float)) * 100.0,
        loss=log_loss(ground_truth, logits),
    )
    return result


from enum import Enum

import faiss
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors._base import _get_weights
from sklearn.preprocessing import Normalizer
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class Preprocess_Type(Enum):
    NONE = "NONE"
    NORM_SEPERATLY = "NORM_SEPERATLY"
    NORM_AFTER_CONCAT = "NORM_AFTER_CONCAT"


class DualCombo(Enum):
    AVG = "AVG"
    CONCAT = "CONCAT"


class DualFaissKNNClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn wrapper interface for Faiss KNN.

    Parameters
    ----------
    n_neighbors : int (Default = 5)
                Number of neighbors used in the nearest neighbor search.

    n_jobs : int (Default = None)
             The number of jobs to run in parallel for both fit and predict.
              If -1, then the number of jobs is set to the number of cores.

    algorithm : {'brute', 'voronoi'} (Default = 'brute')

        Algorithm used to compute the nearest neighbors:

            - 'brute' will use the :class: `IndexFlatL2` class from faiss.
            - 'voronoi' will use :class:`IndexIVFFlat` class from faiss.
            - 'hierarchical' will use :class:`IndexHNSWFlat` class from faiss.

        Note that selecting 'voronoi' the system takes more time during
        training, however it can significantly improve the search time
        on inference. 'hierarchical' produce very fast and accurate indexes,
        however it has a higher memory requirement. It's recommended when
        you have a lots of RAM or the dataset is small.

        For more information see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

    n_cells : int (Default = 100)
        Number of voronoi cells. Only used when algorithm=='voronoi'.

    n_probes : int (Default = 1)
        Number of cells that are visited to perform the search. Note that the
        search time roughly increases linearly with the number of probes.
        Only used when algorithm=='voronoi'.

    References
    ----------
    Johnson Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity
    search with gpus." arXiv preprint arXiv:1702.08734 (2017).
    """

    def __init__(
        self,
        modalities: list[str],
        n_neighbors=5,
        n_jobs=None,
        algorithm="l2",
        weights="uniform",
        preprocess: Preprocess_Type = Preprocess_Type.NONE,
        C=2.0,
        d=32,
    ):
        self.modalities = modalities
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.weights = weights
        self.preprocess = preprocess
        self.C = C
        self.d = d

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------

        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : array, shape (n_samples,)
                Class labels for samples in X.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Parameters
        ----------

        X : array of shape (n_samples, n_features)
            The input data.

        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dists : list of shape = [n_samples, k]
            The distances between the query and each sample in the region of
            competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """
        # X, pre, post = self.transform_preprocess(X)
        X = self.transform_preprocess(X)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0." " Got {}".format(n_neighbors))
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take {} value, " "enter integer value".format(type(n_neighbors))
                )

        check_is_fitted(self, "pre_and_post_index_")
        check_is_fitted(self, "pre_index_")
        check_is_fitted(self, "post_index_")

        X = np.atleast_2d(X).astype(np.float32)
        dist, idx = self.search(X, n_neighbors)
        # both_dist, both_idx = self.pre_and_post_index_.search(X, n_neighbors)
        # both_dist /= self.C  # we normalize both_dist since it is the sum of the pre and post distances
        # pre_dist, pre_idx = self.pre_index_.search(pre, n_neighbors)
        # post_dist, post_idx = self.post_index_.search(post, n_neighbors)
        # # all 3
        # if self.combo == DualCombo.CONCAT:
        #     dist = np.concatenate([both_dist, pre_dist, post_dist], axis=1)
        #     idx = np.concatenate([both_idx, pre_idx, post_idx], axis=1)
        # else:
        #     dist = [both_dist, pre_dist, post_dist]
        #     idx = [both_idx, pre_idx, post_idx]

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
                embed_array = np.asarray(X[modality].to_list())
                embeddings.append(embed_array)
        else:
            embeddings = X
        return np.concatenate(embeddings, axis=1)

    def fit_preprocess(self, X: pl.DataFrame):
        if self.preprocess == Preprocess_Type.NORM_AFTER_CONCAT:
            self.scaler = Normalizer()
            self.scaler.fit(X)
        elif self.preprocess == Preprocess_Type.NORM_SEPERATLY:
            self.scalers = []
            assert X.shape[1] == self.d * 2
            for modality in self.modalities:
                embed_array = np.asarray(X[modality].to_list())
                scaler = Normalizer()
                scaler.fit(embed_array)
                self.scalers.append(scaler)
        else:
            assert self.preprocess == Preprocess_Type.NONE

    def transform_preprocess(self, X):
        if self.preprocess == Preprocess_Type.NORM_AFTER_CONCAT:
            self.scaler = Normalizer()
            X = self.scaler.transform(X)
            return X
        elif self.preprocess == Preprocess_Type.NORM_SEPERATLY:
            self.scalers = []
            assert X.shape[1] == self.d * 2
            for modality in self.modalities:
                embed_array = np.asarray(X[modality].to_list())
                # scaler = self.scalers[0]
                scaler = Normalizer()
                scaler.transform(embed_array)
                self.scalers.append(scaler)
            return self.scalers
        else:
            assert self.preprocess == Preprocess_Type.NONE
            return X

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.
        """
        assert X.shape[1] == self.d * 2, f"X.shape[1] = {X.shape[1]}, self.d*2 = {self.d*2}"
        self.classes_ = unique_labels(y)
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        self.fit_preprocess(X)
        X = self.transform_preprocess(X)
        d = X.shape[1]  # dimensionality of the feature vector
        self._prepare_knn_algorithm(X, d)
        self.index.add(X)
        self.y_ = y
        self.n_classes_ = np.unique(y).size
        return self

    def _prepare_knn_algorithm(self, X, d):
        """_summary_

        Args:
            X (_type_): _description_
            d (_type_): dimension of the embeddings

        Raises:
            ValueError: _description_
        """
        for modality in self.modalities:
            if self.algorithm == "l2":
                index = faiss.IndexFlatL2(d * len(self.modalities))
                self.index = index
                # TODO add some argument for allowing gpu usage
                # self = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            elif self.algorithm == "ip":
                index = faiss.IndexFlatIP(d)
                # index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                self.index = index
            else:
                raise ValueError(
                    "Invalid algorithm option." " Expected ['l2', 'ip'], " "got {}".format(self.algorithm)
                )

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]


def train_dual_model(train_features: list[np.array], train_labels, val_features, val_labels, dimension):
    """_summary_

    Args:
        train_features (list[np.Array]): list of 2d array of embeddings (each row is a single embedding) for each modality
            so train_features[0] contains embeddings of modality_0 for all patient events
        train_labels (_type_): corresponding prediction label for this, binary (1/0) for now.
        val_features (_type_): _description_
        val_labels (_type_): _description_
        dimension (_type_): _description_

    Returns:
        _type_: _description_
    """
    LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    pipe = Pipeline([("classifier", DualFaissKNNClassifier(algorithm="l2", d=dimension))])
    input_data = np.concatenate([train_features, val_features])
    input_labels = np.concatenate([train_labels, val_labels])

    pds = PredefinedSplit(test_fold=[-1] * len(train_labels) + [0] * len(val_labels))
    param_grid = [
        {
            "classifier__preprocess": list(Preprocess_Type),
            "classifier__algorithm": ["l2", "ip"],
            "classifier__weights": ["uniform", "distance"],
            "classifier__n_neighbors": [30, 100, 300, 1000],
            "classifier__combo": list(DualCombo),
            "classifier__d": [dimension],
            # TODO: add 'modality_weights' optimization
            # TODO: scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
        },
    ]
    # Create grid search object

    clf = GridSearchCV(pipe, param_grid=param_grid, cv=pds, verbose=2, n_jobs=1, scoring=LogLoss, refit=False)

    # Fit on data

    best_clf = clf.fit(input_data, input_labels)
    best_params = {k[12:]: v for k, v in best_clf.best_params_.items()}
    best_clf = DualFaissKNNClassifier(**best_params)
    best_clf.fit(train_features, train_labels)
    return best_clf, best_params


if __name__ == "__main__":
    dual_model, dual_best_params = train_dual_model()

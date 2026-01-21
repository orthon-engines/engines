"""
PRISM State Layer - State/Transition Classifier
================================================

Classifies transitions based on their signatures.
Unlike static classification, this classifies the TRANSITION TYPE.

Two modes:
1. Supervised: Train on labeled transitions (fault type)
2. Unsupervised: Discover transition types via clustering

Key insight: PRISM classifies DYNAMICS, not snapshots. The signature
captures how the system transitions, not what state it's in.

Usage:
    from orthon._internal.state import StateClassifier

    # Supervised mode (with labels)
    clf = StateClassifier(mode='supervised')
    clf.fit(signatures, labels=[1, 2, 1, 3, ...])
    predictions = clf.predict(new_signatures)

    # Unsupervised mode (discover clusters)
    clf = StateClassifier(mode='unsupervised', n_clusters=5)
    clf.fit(signatures)
    clusters = clf.predict(signatures)
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Dict, Optional, Tuple
import warnings

from .state_signature import StateSignature, signatures_to_features

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class StateClassifier:
    """
    Classifies system state transitions.

    Two modes:
    1. Supervised: Train on labeled transitions (fault type)
       - Uses GradientBoostingClassifier for robust classification
       - Provides feature importance for interpretability

    2. Unsupervised: Discover transition types via clustering
       - Uses KMeans or DBSCAN for cluster discovery
       - No labels required

    Attributes:
        mode: 'supervised' or 'unsupervised'
        model: Underlying sklearn model
        scaler: StandardScaler for feature normalization
        feature_cols: List of feature column names used
        is_fitted: Whether the model has been trained

    Example:
        >>> clf = StateClassifier(mode='supervised')
        >>> clf.fit(train_signatures, train_labels)
        >>> predictions = clf.predict(test_signatures)
        >>> importance = clf.get_feature_importance()
    """

    def __init__(
        self,
        mode: str = 'supervised',
        n_clusters: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the classifier.

        Args:
            mode: 'supervised' or 'unsupervised'
            n_clusters: Number of clusters for unsupervised mode
            random_state: Random seed for reproducibility
        """
        self.mode = mode
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()

        if mode == 'supervised':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
            )

        self.feature_cols: Optional[List[str]] = None
        self.is_fitted = False

    def fit(
        self,
        signatures: List[StateSignature],
        labels: Optional[List[int]] = None,
    ) -> 'StateClassifier':
        """
        Train the classifier.

        Args:
            signatures: List of StateSignature objects
            labels: Class labels (required for supervised mode)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If supervised mode and no labels provided
            ValueError: If not enough signatures for training
        """
        if len(signatures) < 2:
            raise ValueError("Need at least 2 signatures for training")

        features_df = signatures_to_features(signatures)

        # Define feature columns (exclude metadata)
        self.feature_cols = [
            c for c in features_df.columns
            if c not in ['window_end', 'leader_signal']
        ]

        X = features_df.select(self.feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)

        if self.mode == 'supervised':
            if labels is None:
                raise ValueError("Supervised mode requires labels")
            if len(labels) != len(signatures):
                raise ValueError(f"Labels length ({len(labels)}) != signatures length ({len(signatures)})")
            self.model.fit(X, labels)
        else:
            # Unsupervised: use clustering
            self.model.fit(X)

        self.is_fitted = True
        return self

    def predict(self, signatures: List[StateSignature]) -> np.ndarray:
        """
        Predict state/transition type.

        Args:
            signatures: List of StateSignature objects

        Returns:
            Array of predicted labels/clusters

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if len(signatures) == 0:
            return np.array([])

        features_df = signatures_to_features(signatures)
        X = features_df.select(self.feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, signatures: List[StateSignature]) -> np.ndarray:
        """
        Predict class probabilities (supervised mode only).

        Args:
            signatures: List of StateSignature objects

        Returns:
            Array of shape (n_samples, n_classes) with probabilities

        Raises:
            ValueError: If model not fitted or not supervised mode
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.mode != 'supervised':
            raise ValueError("predict_proba only available in supervised mode")

        if len(signatures) == 0:
            return np.array([])

        features_df = signatures_to_features(signatures)
        X = features_df.select(self.feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (supervised mode only).

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If model not fitted or not supervised mode
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.mode != 'supervised':
            raise ValueError("Feature importance only available in supervised mode")

        return dict(zip(self.feature_cols, self.model.feature_importances_))

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centers (unsupervised mode only).

        Returns:
            Array of shape (n_clusters, n_features) with cluster centers

        Raises:
            ValueError: If model not fitted or not unsupervised mode
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.mode != 'unsupervised':
            raise ValueError("Cluster centers only available in unsupervised mode")

        return self.scaler.inverse_transform(self.model.cluster_centers_)

    def evaluate_clustering(self, signatures: List[StateSignature]) -> Dict[str, float]:
        """
        Evaluate clustering quality (unsupervised mode).

        Args:
            signatures: List of StateSignature objects

        Returns:
            Dictionary with clustering metrics:
                - silhouette: Silhouette score (-1 to 1, higher is better)
                - inertia: Within-cluster sum of squares (lower is better)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.mode != 'unsupervised':
            raise ValueError("Clustering evaluation only for unsupervised mode")

        features_df = signatures_to_features(signatures)
        X = features_df.select(self.feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)

        labels = self.model.predict(X)

        # Silhouette score requires at least 2 clusters with samples
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            silhouette = 0.0
        else:
            silhouette = silhouette_score(X, labels)

        return {
            'silhouette': silhouette,
            'inertia': self.model.inertia_,
            'n_clusters_used': len(unique_labels),
        }


def find_optimal_clusters(
    signatures: List[StateSignature],
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        signatures: List of StateSignature objects
        k_range: Range of k values to try (min, max)
        random_state: Random seed

    Returns:
        Dictionary with:
            - optimal_k: Best number of clusters
            - scores: Dict mapping k to silhouette score
            - classifier: Fitted StateClassifier with optimal k
    """
    features_df = signatures_to_features(signatures)
    feature_cols = [c for c in features_df.columns if c not in ['window_end', 'leader_signal']]
    X = features_df.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    best_k = k_range[0]
    best_score = -1

    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(signatures):
            break

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            scores[k] = 0.0
            continue

        score = silhouette_score(X_scaled, labels)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    # Fit classifier with optimal k
    clf = StateClassifier(mode='unsupervised', n_clusters=best_k, random_state=random_state)
    clf.fit(signatures)

    return {
        'optimal_k': best_k,
        'scores': scores,
        'best_score': best_score,
        'classifier': clf,
    }

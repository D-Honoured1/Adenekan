"""
AttentionMLPClassifier — sklearn-compatible MLP with a learnable feature attention layer.

Kept in its own module so joblib can deserialise the class consistently regardless
of which script calls joblib.load() (train_baselines.py, evaluate.py, api, etc.).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset


class _AttentionMLP(nn.Module):
    """
    Learnable sigmoid-gated attention over input features, followed by dense layers.

    Architecture
    ------------
        Input
          └─► sigmoid(attn_weights) * Input   ← per-feature attention gate
              └─► Linear → ReLU → Dropout  (repeated for each hidden size)
                  └─► Linear(h_last, 1)    (logit)
    """

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.attn_weights = nn.Parameter(torch.zeros(n_features))

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.attn_weights)
        return self.mlp(x * gate).squeeze(-1)

    @property
    def feature_weights(self) -> np.ndarray:
        return torch.sigmoid(self.attn_weights).detach().cpu().numpy()


class AttentionMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible binary classifier: MLP with a learnable feature attention layer.

    The attention gate (sigmoid-gated weights) and the dense MLP layers are trained
    jointly end-to-end via Adam + BCEWithLogitsLoss.

    Parameters
    ----------
    hidden_sizes : tuple[int]   Sizes of dense hidden layers.
    epochs       : int          Training epochs.
    batch_size   : int
    lr           : float        Adam learning rate.
    random_state : int

    Attributes
    ----------
    feature_weights_ : ndarray of shape (n_features,)
        Learned sigmoid gate values after training. Higher → feature more attended.
    """

    def __init__(
        self,
        hidden_sizes: tuple = (256, 128, 64),
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttentionMLPClassifier":
        torch.manual_seed(self.random_state)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_ = _AttentionMLP(X.shape[1], self.hidden_sizes)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()

        self.model_.eval()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.feature_weights_ = self.model_.feature_weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

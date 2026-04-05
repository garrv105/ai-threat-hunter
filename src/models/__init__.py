"""Model architectures for threat detection."""

try:
    from .lstm_model import LSTMModel

    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    LSTMModel = None

try:
    from .cnn_model import CNNModel

    HAS_CNN = True
except ImportError:
    HAS_CNN = False
    CNNModel = None

from .ensemble_model import EnsembleModel

__all__ = ["LSTMModel", "CNNModel", "EnsembleModel"]

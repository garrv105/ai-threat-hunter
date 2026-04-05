import numpy as np
import pytest

from src.models.ensemble_model import EnsembleModel

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestEnsembleModel:
    @pytest.fixture
    def ensemble(self):
        return EnsembleModel()

    def test_initialization(self, ensemble):
        assert ensemble.models == {}
        assert "lstm" in ensemble.weights
        assert "cnn" in ensemble.weights
        assert "isolation_forest" in ensemble.weights

    def test_weights_sum(self, ensemble):
        total = sum(ensemble.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_add_model(self, ensemble):
        class DummyModel:
            def predict(self, x):
                return np.ones((1, 5)) * 0.2

        ensemble.add_model("dummy", DummyModel(), weight=0.5)
        assert "dummy" in ensemble.models
        assert ensemble.weights["dummy"] == 0.5

    def test_predict_no_models(self, ensemble):
        result = ensemble.predict(np.random.rand(1, 41))
        assert result.shape == (1, 5)

    def test_predict_with_dummy_model(self, ensemble):
        class DummyModel:
            def predict(self, x):
                return np.ones((1, 5)) * 0.2

        ensemble.add_model("test_model", DummyModel(), weight=1.0)
        result = ensemble.predict(np.random.rand(1, 41))
        assert result.shape == (1, 5)

    def test_predict_model_without_predict_method(self, ensemble):
        class NoPredict:
            pass

        ensemble.add_model("bad", NoPredict(), weight=1.0)
        result = ensemble.predict(np.random.rand(1, 41))
        # Should fall through to dummy prediction
        assert result.shape == (1, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLSTMModel:
    @pytest.fixture
    def model(self):
        from src.models.lstm_model import LSTMModel

        return LSTMModel(input_size=41, hidden_size=64, num_layers=2, num_classes=5)

    def test_initialization(self, model):
        assert model.hidden_size == 64
        assert model.num_layers == 2

    def test_forward_pass(self, model):
        x = torch.randn(2, 10, 41)  # batch=2, seq=10, features=41
        output = model.forward(x)
        assert output.shape == (2, 5)

    def test_forward_output_sums_to_one(self, model):
        x = torch.randn(1, 10, 41)
        output = model.forward(x)
        np.testing.assert_almost_equal(output.sum().item(), 1.0, decimal=5)

    def test_predict(self, model):
        x = np.random.rand(10, 41).astype(np.float32)
        result = model.predict(x)
        assert result.shape == (1, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCNNModel:
    @pytest.fixture
    def model(self):
        from src.models.cnn_model import CNNModel

        return CNNModel(input_size=41, num_classes=5)

    def test_initialization(self, model):
        assert model.conv1 is not None
        assert model.fc2 is not None

    def test_forward_pass(self, model):
        x = torch.randn(2, 41)
        output = model.forward(x)
        assert output.shape == (2, 5)

    def test_forward_single_sample(self, model):
        x = torch.randn(1, 41)
        output = model.forward(x)
        assert output.shape == (1, 5)

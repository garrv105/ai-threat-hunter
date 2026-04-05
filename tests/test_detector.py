import numpy as np
import pytest

from src.threat_detector import ThreatDetector


class TestThreatDetector:
    @pytest.fixture
    def detector(self):
        return ThreatDetector()

    def test_initialization(self, detector):
        assert detector is not None
        assert hasattr(detector, "config")
        assert hasattr(detector, "threat_history")

    def test_initialization_default_config(self, detector):
        assert isinstance(detector.config, dict)
        assert detector.models == {}
        assert detector.threat_history == []

    def test_detect_threat_returns_required_keys(self, detector):
        data = np.random.rand(41)
        result = detector.detect_threat(data)
        required_keys = [
            "is_threat",
            "confidence",
            "threat_type",
            "severity",
            "timestamp",
            "processing_time_ms",
        ]
        for key in required_keys:
            assert key in result

    def test_detect_threat_types(self, detector):
        data = np.random.rand(41)
        result = detector.detect_threat(data)
        assert isinstance(result["is_threat"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["threat_type"], str)
        assert isinstance(result["severity"], str)
        assert isinstance(result["processing_time_ms"], float)

    def test_detect_threat_confidence_range(self, detector):
        data = np.random.rand(41)
        result = detector.detect_threat(data)
        assert 0 <= result["confidence"] <= 1

    def test_severity_calculation_critical_dos(self, detector):
        severity = detector._calculate_severity(0.96, "DoS")
        assert severity == "critical"

    def test_severity_calculation_probe_medium(self, detector):
        severity = detector._calculate_severity(0.87, "Probe")
        assert severity == "medium"

    def test_severity_calculation_normal(self, detector):
        severity = detector._calculate_severity(0.5, "Normal")
        assert severity == "low"

    def test_severity_calculation_high_confidence(self, detector):
        severity = detector._calculate_severity(0.96, "R2L")
        assert severity == "critical"

    def test_severity_calculation_r2l_base(self, detector):
        severity = detector._calculate_severity(0.87, "R2L")
        assert severity == "high"

    def test_severity_calculation_u2r(self, detector):
        severity = detector._calculate_severity(0.87, "U2R")
        assert severity == "critical"

    def test_severity_probe_high_confidence(self, detector):
        severity = detector._calculate_severity(0.92, "Probe")
        assert severity == "high"

    def test_classify_threat_classes(self, detector):
        for i in range(5):
            preds = np.zeros((1, 5))
            preds[0, i] = 1.0
            result = detector._classify_threat(preds)
            if i == 0:
                assert result == "Normal"
            else:
                assert result in ["DoS", "Probe", "R2L", "U2R"]

    def test_extract_features_1d(self, detector):
        data = np.random.rand(41)
        result = detector._extract_features(data)
        assert result.shape == (1, 41)

    def test_extract_features_2d(self, detector):
        data = np.random.rand(5, 41)
        result = detector._extract_features(data)
        assert result.shape == (5, 41)

    def test_ensemble_predict_shape(self, detector):
        features = np.random.rand(3, 41)
        result = detector._ensemble_predict(features)
        assert result.shape == (3, 5)

    def test_ensemble_predict_sums_to_one(self, detector):
        features = np.random.rand(1, 41)
        result = detector._ensemble_predict(features)
        np.testing.assert_almost_equal(result.sum(axis=1), [1.0])

    def test_threat_statistics_empty(self, detector):
        stats = detector.get_threat_statistics()
        assert stats == {"total_threats": 0}

    def test_load_config_missing_file(self, detector):
        config = detector._load_config("nonexistent.yaml")
        assert "detection" in config
        assert config["detection"]["threshold"] == 0.85

    def test_load_models_without_torch(self, detector):
        detector.load_models("./models")
        # Should not raise, just log warning/error


if __name__ == "__main__":
    pytest.main([__file__])

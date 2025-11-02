import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.threat_detector import ThreatDetector


class TestThreatDetector:
    
    @pytest.fixture
    def detector(self):
        return ThreatDetector()
    
    def test_initialization(self, detector):
        assert detector is not None
        assert hasattr(detector, 'config')
        assert hasattr(detector, 'threat_history')
    
    def test_detect_threat(self, detector):
        # Test with random data
        data = np.random.rand(41)
        result = detector.detect_threat(data)
        
        assert 'is_threat' in result
        assert 'confidence' in result
        assert 'threat_type' in result
        assert 'severity' in result
        assert isinstance(result['is_threat'], bool)
        assert 0 <= result['confidence'] <= 1
    
    def test_severity_calculation(self, detector):
        # Test DoS with high confidence
        severity = detector._calculate_severity(0.96, 'DoS')
        assert severity == 'critical'
        
        # Test Probe with medium confidence
        severity = detector._calculate_severity(0.87, 'Probe')
        assert severity == 'medium'


if __name__ == '__main__':
    pytest.main([__file__])
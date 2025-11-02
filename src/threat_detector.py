import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatDetector:
    """
    Main threat detection engine using ensemble deep learning models.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the threat detector with configuration."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.threat_history = []
        
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return {
                'model': {'threshold': 0.85},
                'detection': {'threshold': 0.85}
            }
    
    def load_models(self, model_dir: str = './models'):
        """Load pre-trained models."""
        try:
            import torch
            from src.models.lstm_model import LSTMModel
            from src.models.ensemble_model import EnsembleModel
            
            # Load LSTM model
            self.models['lstm'] = LSTMModel()
            
            # Load ensemble model
            self.models['ensemble'] = EnsembleModel()
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def detect_threat(self, network_data: np.ndarray) -> Dict:
        """
        Detect threats in network traffic data.
        
        Args:
            network_data: Preprocessed network features
            
        Returns:
            Dictionary containing threat detection results
        """
        start_time = datetime.now()
        
        # Feature extraction
        features = self._extract_features(network_data)
        
        # Make predictions using ensemble
        predictions = self._ensemble_predict(features)
        
        # Calculate confidence scores
        confidence = float(np.max(predictions))
        threat_type = self._classify_threat(predictions)
        
        # Determine if threat detected
        threshold = self.config.get('detection', {}).get('threshold', 0.85)
        is_threat = confidence > threshold
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'is_threat': is_threat,
            'confidence': confidence,
            'threat_type': threat_type,
            'severity': self._calculate_severity(confidence, threat_type),
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        if is_threat:
            self.threat_history.append(result)
            logger.warning(f"Threat detected: {threat_type} (confidence: {confidence:.2%})")
        
        return result
    
    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract relevant features from raw network data."""
        # Simulate feature extraction
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        return data
    
    def _ensemble_predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble of models."""
        # Simulate ensemble prediction
        predictions = np.random.rand(len(features), 5)  # 5 attack classes
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        return predictions
    
    def _classify_threat(self, predictions: np.ndarray) -> str:
        """Classify the type of threat based on predictions."""
        threat_classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        class_idx = np.argmax(predictions)
        return threat_classes[class_idx] if class_idx > 0 else 'Normal'
    
    def _calculate_severity(self, confidence: float, threat_type: str) -> str:
        """Calculate threat severity level."""
        severity_map = {
            'DoS': 'critical',
            'R2L': 'high',
            'U2R': 'critical',
            'Probe': 'medium'
        }
        
        if threat_type == 'Normal':
            return 'low'
        
        base_severity = severity_map.get(threat_type, 'medium')
        
        if confidence > 0.95:
            return 'critical'
        elif confidence > 0.90:
            return base_severity if base_severity != 'medium' else 'high'
        else:
            return base_severity
    
    def get_threat_statistics(self) -> Dict:
        """Get statistics about detected threats."""
        if not self.threat_history:
            return {'total_threats': 0}
        
        df = pd.DataFrame(self.threat_history)
        
        stats = {
            'total_threats': len(df),
            'by_type': df['threat_type'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'avg_confidence': float(df['confidence'].mean()),
            'avg_processing_time_ms': float(df['processing_time_ms'].mean())
        }
        
        return stats
class EnsembleModel:
    """
    Ensemble model combining LSTM, CNN, and Isolation Forest.
    """
    
    def __init__(self):
        """Initialize ensemble components."""
        self.models = {}
        self.weights = {'lstm': 0.5, 'cnn': 0.3, 'isolation_forest': 0.2}
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make ensemble prediction."""
        predictions = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(x)
                predictions.append(pred * self.weights.get(name, 1.0))
        
        if predictions:
            ensemble_pred = np.sum(predictions, axis=0) / sum(self.weights.values())
            return ensemble_pred
        
        # Return dummy prediction if no models
        return np.random.rand(1, 5)
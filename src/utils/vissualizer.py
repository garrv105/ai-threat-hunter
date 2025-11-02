import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


class ThreatVisualizer:
    """Visualization tools for threat detection results."""
    
    def __init__(self):
        sns.set_style('darkgrid')
        self.colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    
    def plot_threat_timeline(self, threats: List[Dict]):
        """Plot threats over time."""
        if not threats:
            print("No threats to visualize")
            return
        
        timestamps = [t['timestamp'] for t in threats]
        severities = [t['severity'] for t in threats]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(timestamps)), range(len(timestamps)), 
                   c=[self._severity_to_color(s) for s in severities],
                   s=100, alpha=0.6)
        plt.xlabel('Threat Index')
        plt.ylabel('Detection Order')
        plt.title('Threat Detection Timeline')
        plt.tight_layout()
        plt.show()
    
    def _severity_to_color(self, severity: str) -> str:
        color_map = {
            'low': 'green',
            'medium': 'yellow',
            'high': 'orange',
            'critical': 'red'
        }
        return color_map.get(severity, 'gray')
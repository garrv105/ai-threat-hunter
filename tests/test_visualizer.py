import pytest

from src.utils.visualizer import ThreatVisualizer


class TestThreatVisualizer:
    @pytest.fixture
    def visualizer(self):
        return ThreatVisualizer()

    def test_initialization(self, visualizer):
        assert len(visualizer.colors) == 5

    def test_severity_to_color_low(self, visualizer):
        assert visualizer._severity_to_color("low") == "green"

    def test_severity_to_color_medium(self, visualizer):
        assert visualizer._severity_to_color("medium") == "yellow"

    def test_severity_to_color_high(self, visualizer):
        assert visualizer._severity_to_color("high") == "orange"

    def test_severity_to_color_critical(self, visualizer):
        assert visualizer._severity_to_color("critical") == "red"

    def test_severity_to_color_unknown(self, visualizer):
        assert visualizer._severity_to_color("unknown") == "gray"

    def test_plot_threat_timeline_empty(self, visualizer, capsys):
        visualizer.plot_threat_timeline([])
        captured = capsys.readouterr()
        assert "No threats to visualize" in captured.out

    def test_get_severity_summary_empty(self, visualizer):
        result = visualizer.get_severity_summary([])
        assert result == {}

    def test_get_severity_summary(self, visualizer):
        threats = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "critical"},
            {"severity": "low"},
        ]
        result = visualizer.get_severity_summary(threats)
        assert result == {"critical": 2, "high": 1, "low": 1}

    def test_get_severity_summary_single(self, visualizer):
        threats = [{"severity": "medium"}]
        result = visualizer.get_severity_summary(threats)
        assert result == {"medium": 1}

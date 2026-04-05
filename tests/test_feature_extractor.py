import queue

import numpy as np
import pytest

from src.feature_extractor import HAS_SCAPY, NetworkFeatureExtractor


class TestNetworkFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return NetworkFeatureExtractor()

    def test_initialization_default(self, extractor):
        assert extractor.interface is None
        assert isinstance(extractor.packet_queue, queue.Queue)
        assert extractor.is_running is False

    def test_initialization_with_interface(self):
        ext = NetworkFeatureExtractor(interface="eth0")
        assert ext.interface == "eth0"

    def test_encode_protocol_tcp(self, extractor):
        assert extractor._encode_protocol(6) == 2

    def test_encode_protocol_udp(self, extractor):
        assert extractor._encode_protocol(17) == 3

    def test_encode_protocol_icmp(self, extractor):
        assert extractor._encode_protocol(1) == 1

    def test_encode_protocol_unknown(self, extractor):
        assert extractor._encode_protocol(99) == 0

    def test_stop_capture(self, extractor):
        extractor.is_running = True
        extractor.stop_capture()
        assert extractor.is_running is False

    def test_get_features_batch_empty(self, extractor):
        batch = extractor.get_features_batch(batch_size=10)
        assert batch == []

    def test_get_features_batch_with_data(self, extractor):
        for i in range(5):
            extractor.packet_queue.put(np.zeros(41))
        batch = extractor.get_features_batch(batch_size=3)
        assert len(batch) == 3

    def test_get_features_batch_partial(self, extractor):
        for i in range(2):
            extractor.packet_queue.put(np.ones(41))
        batch = extractor.get_features_batch(batch_size=10)
        assert len(batch) == 2

    def test_flow_stats_default(self, extractor):
        flow = extractor.flow_stats["test_flow"]
        assert flow["packet_count"] == 0
        assert flow["total_bytes"] == 0

    @pytest.mark.skipif(not HAS_SCAPY, reason="scapy not installed")
    def test_extract_packet_features_requires_scapy(self, extractor):
        # Only runs when scapy is available
        pass

    def test_start_capture_without_scapy(self, extractor):
        if not HAS_SCAPY:
            with pytest.raises(RuntimeError, match="scapy is required"):
                extractor.start_capture(packet_count=1)

    def test_extract_features_without_scapy(self, extractor):
        if not HAS_SCAPY:
            with pytest.raises(RuntimeError, match="scapy is required"):
                extractor.extract_packet_features(None)

from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import threading
import queue


class NetworkFeatureExtractor:
    """
    Extract features from network packets for threat detection.
    """
    
    def __init__(self, interface: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            interface: Network interface to sniff (None for default)
        """
        self.interface = interface
        self.packet_queue = queue.Queue()
        self.flow_stats = defaultdict(lambda: {
            'packet_count': 0,
            'total_bytes': 0,
            'protocol_counts': defaultdict(int),
            'flags': set()
        })
        self.is_running = False
        
    def extract_packet_features(self, packet) -> np.ndarray:
        """
        Extract features from a single packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Feature vector as numpy array
        """
        features = np.zeros(41)  # Standard NSL-KDD feature count
        
        try:
            if IP in packet:
                # Basic IP features
                features[0] = len(packet)  # packet length
                features[1] = self._encode_protocol(packet[IP].proto)
                
                # TCP features
                if TCP in packet:
                    features[2] = packet[TCP].sport
                    features[3] = packet[TCP].dport
                    features[4] = packet[TCP].flags
                    features[5] = packet[TCP].window
                
                # UDP features
                elif UDP in packet:
                    features[2] = packet[UDP].sport
                    features[3] = packet[UDP].dport
                
                # Flow-based features
                flow_key = self._get_flow_key(packet)
                flow = self.flow_stats[flow_key]
                
                features[6] = flow['packet_count']
                features[7] = flow['total_bytes']
                
                # Update flow statistics
                flow['packet_count'] += 1
                flow['total_bytes'] += len(packet)
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
        
        return features
    
    def _encode_protocol(self, proto_num: int) -> int:
        """Encode protocol number to categorical value."""
        protocol_map = {1: 1, 6: 2, 17: 3}  # ICMP, TCP, UDP
        return protocol_map.get(proto_num, 0)
    
    def _get_flow_key(self, packet) -> str:
        """Generate unique flow key from packet."""
        if IP in packet:
            src = packet[IP].src
            dst = packet[IP].dst
            if TCP in packet:
                return f"{src}:{packet[TCP].sport}-{dst}:{packet[TCP].dport}"
            elif UDP in packet:
                return f"{src}:{packet[UDP].sport}-{dst}:{packet[UDP].dport}"
            return f"{src}-{dst}"
        return "unknown"
    
    def start_capture(self, packet_count: int = 0):
        """
        Start capturing network packets.
        
        Args:
            packet_count: Number of packets to capture (0 for infinite)
        """
        self.is_running = True
        logger.info(f"Starting packet capture on interface: {self.interface or 'default'}")
        
        try:
            sniff(
                iface=self.interface,
                prn=self._packet_handler,
                count=packet_count,
                store=False
            )
        except Exception as e:
            logger.error(f"Error during packet capture: {e}")
            self.is_running = False
    
    def _packet_handler(self, packet):
        """Handle captured packet."""
        if self.is_running:
            features = self.extract_packet_features(packet)
            self.packet_queue.put(features)
    
    def stop_capture(self):
        """Stop packet capture."""
        self.is_running = False
        logger.info("Packet capture stopped")
    
    def get_features_batch(self, batch_size: int = 64) -> List[np.ndarray]:
        """Get a batch of extracted features."""
        batch = []
        while len(batch) < batch_size and not self.packet_queue.empty():
            batch.append(self.packet_queue.get())
        return batch
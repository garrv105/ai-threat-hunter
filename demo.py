"""
Demo script for AI Threat Hunter
Run this to see the system in action!
"""

import numpy as np
from src.threat_detector import ThreatDetector
import time


def main():
    print("=" * 80)
    print("AI-POWERED NETWORK THREAT HUNTER - Demo")
    print("=" * 80)
    print()
    
    # Initialize detector
    print("🔧 Initializing threat detector...")
    detector = ThreatDetector()
    print("✓ Threat detector initialized")
    print()
    
    # Simulate network traffic
    print("🌐 Simulating network traffic analysis...")
    print()
    
    for i in range(10):
        # Generate random network features
        network_data = np.random.rand(41)
        
        # Detect threats
        print(f"📦 Analyzing Packet {i+1}...")
        result = detector.detect_threat(network_data)
        
        # Display results
        if result['is_threat']:
            print(f"   ⚠️  THREAT DETECTED!")
            print(f"   Type: {result['threat_type']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Severity: {result['severity'].upper()}")
        else:
            print(f"   ✓ Normal traffic")
        
        print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
        print()
        
        time.sleep(0.5)
    
    # Show statistics
    print("=" * 80)
    print("📊 THREAT STATISTICS")
    print("=" * 80)
    stats = detector.get_threat_statistics()
    
    print(f"Total threats detected: {stats['total_threats']}")
    if stats['total_threats'] > 0:
        print(f"Average confidence: {stats['avg_confidence']:.2%}")
        print(f"Average processing time: {stats['avg_processing_time_ms']:.2f}ms")
        print()
        print("Threats by type:")
        for threat_type, count in stats.get('by_type', {}).items():
            print(f"  - {threat_type}: {count}")
        print()
        print("Threats by severity:")
        for severity, count in stats.get('by_severity', {}).items():
            print(f"  - {severity}: {count}")
    
    print()
    print("=" * 80)
    print("✓ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
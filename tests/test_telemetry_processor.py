"""
Unit tests for telemetry processing module
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.telemetry_processor import (
    TelemetryProcessor, TelemetryPoint, TelemetryComparator
)


class TestTelemetryProcessor(unittest.TestCase):
    """Test TelemetryProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = TelemetryProcessor(sample_rate=100)
        
        # Create sample telemetry data
        self.sample_data = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 1000),
            'speed': np.sin(np.linspace(0, 4*np.pi, 1000)) * 50 + 100,
            'throttle': np.clip(np.random.randn(1000) * 20 + 60, 0, 100),
            'brake': np.clip(np.random.randn(1000) * 10 + 5, 0, 100),
            'steering': np.sin(np.linspace(0, 8*np.pi, 1000)) * 30,
            'gear': np.random.randint(1, 7, 1000),
            'rpm': np.random.randint(5000, 9000, 1000)
        })
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        metrics = self.processor.calculate_racing_metrics(self.sample_data)
        
        self.assertIn('avg_speed', metrics)
        self.assertIn('max_speed', metrics)
        self.assertIn('lap_time', metrics)
        self.assertGreater(metrics['avg_speed'], 0)
    
    def test_detect_corners(self):
        """Test corner detection"""
        corners = self.processor.detect_corners(self.sample_data)
        
        self.assertIsInstance(corners, list)
        if len(corners) > 0:
            self.assertIsInstance(corners[0], tuple)
            self.assertEqual(len(corners[0]), 2)
    
    def test_corner_analysis(self):
        """Test corner performance analysis"""
        corners = self.processor.detect_corners(self.sample_data)
        
        if len(corners) > 0:
            analysis = self.processor.analyze_corner_performance(
                self.sample_data, corners
            )
            
            self.assertGreater(len(analysis), 0)
            self.assertIn('entry_speed', analysis[0])
            self.assertIn('exit_speed', analysis[0])
    
    def test_generate_insights(self):
        """Test insight generation"""
        metrics = self.processor.calculate_racing_metrics(self.sample_data)
        corners = self.processor.detect_corners(self.sample_data)
        corner_analysis = self.processor.analyze_corner_performance(
            self.sample_data, corners
        )
        
        insights = self.processor.generate_insights(metrics, corner_analysis)
        
        self.assertIsInstance(insights, list)


class TestTelemetryComparator(unittest.TestCase):
    """Test TelemetryComparator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.lap1 = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'speed': np.random.randn(100) * 10 + 100,
            'throttle': np.random.randn(100) * 15 + 70,
            'brake': np.random.randn(100) * 5 + 10,
            'steering': np.random.randn(100) * 20,
            'gear': np.random.randint(1, 6, 100),
            'rpm': np.random.randint(6000, 8000, 100)
        })
        
        self.lap2 = self.lap1.copy()
        self.lap2['speed'] += 5  # Slightly faster lap
    
    def test_compare_laps(self):
        """Test lap comparison"""
        comparator = TelemetryComparator()
        comparison = comparator.compare_laps(self.lap1, self.lap2)
        
        self.assertIn('lap1_time', comparison)
        self.assertIn('lap2_time', comparison)
        self.assertIn('speed_delta', comparison)
        self.assertGreater(comparison['speed_delta'], 0)


if __name__ == '__main__':
    unittest.main()

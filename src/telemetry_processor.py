"""
Telemetry data processing and analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TelemetryPoint:
    """Single telemetry data point"""
    timestamp: float
    speed: float  # km/h
    throttle: float  # 0-100%
    brake: float  # 0-100%
    steering: float  # -100 to 100 (left to right)
    gear: int
    rpm: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    g_force_lat: Optional[float] = None  # Lateral G-force
    g_force_lon: Optional[float] = None  # Longitudinal G-force


class TelemetryProcessor:
    """Process and analyze racing telemetry data"""
    
    def __init__(self, sample_rate: int = 100):
        """
        Initialize telemetry processor
        
        Args:
            sample_rate: Data sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.data: List[TelemetryPoint] = []
        
    def add_point(self, point: TelemetryPoint):
        """Add a telemetry point"""
        self.data.append(point)
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load telemetry from CSV file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with telemetry data
        """
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} telemetry points from {csv_path}")
        return df
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert telemetry data to DataFrame"""
        if not self.data:
            return pd.DataFrame()
        
        data_dict = {
            'timestamp': [p.timestamp for p in self.data],
            'speed': [p.speed for p in self.data],
            'throttle': [p.throttle for p in self.data],
            'brake': [p.brake for p in self.data],
            'steering': [p.steering for p in self.data],
            'gear': [p.gear for p in self.data],
            'rpm': [p.rpm for p in self.data],
        }
        
        return pd.DataFrame(data_dict)
    
    def calculate_racing_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key racing metrics
        
        Args:
            df: Telemetry DataFrame
            
        Returns:
            Dictionary of racing metrics
        """
        metrics = {}
        
        # Lap time (assuming df contains one lap)
        if len(df) > 0:
            metrics['lap_time'] = df['timestamp'].max() - df['timestamp'].min()
        
        # Speed metrics
        metrics['avg_speed'] = df['speed'].mean()
        metrics['max_speed'] = df['speed'].max()
        metrics['min_speed'] = df['speed'].min()
        
        # Throttle analysis
        metrics['avg_throttle'] = df['throttle'].mean()
        metrics['full_throttle_pct'] = (df['throttle'] > 95).sum() / len(df) * 100
        
        # Braking analysis
        metrics['avg_brake'] = df['brake'].mean()
        metrics['brake_frequency'] = (df['brake'] > 10).sum()
        
        # Smoothness (steering variance as proxy)
        metrics['steering_smoothness'] = 100 - min(df['steering'].std(), 100)
        
        return metrics
    
    def detect_corners(self, df: pd.DataFrame, 
                      speed_threshold: float = 0.85) -> List[Tuple[int, int]]:
        """
        Detect corner entry/exit points based on speed
        
        Args:
            df: Telemetry DataFrame
            speed_threshold: Speed drop threshold (fraction of max speed)
            
        Returns:
            List of (start_idx, end_idx) for each corner
        """
        max_speed = df['speed'].max()
        corner_speed = max_speed * speed_threshold
        
        in_corner = df['speed'] < corner_speed
        corners = []
        
        corner_start = None
        for i, is_corner in enumerate(in_corner):
            if is_corner and corner_start is None:
                corner_start = i
            elif not is_corner and corner_start is not None:
                corners.append((corner_start, i))
                corner_start = None
        
        logger.info(f"Detected {len(corners)} corners")
        return corners
    
    def analyze_corner_performance(self, df: pd.DataFrame, 
                                  corners: List[Tuple[int, int]]) -> List[Dict]:
        """
        Analyze performance in each corner
        
        Args:
            df: Telemetry DataFrame
            corners: List of corner indices
            
        Returns:
            List of corner analysis dictionaries
        """
        corner_analyses = []
        
        for i, (start, end) in enumerate(corners):
            corner_data = df.iloc[start:end]
            
            analysis = {
                'corner_num': i + 1,
                'duration': corner_data['timestamp'].max() - corner_data['timestamp'].min(),
                'entry_speed': corner_data.iloc[0]['speed'],
                'min_speed': corner_data['speed'].min(),
                'exit_speed': corner_data.iloc[-1]['speed'],
                'avg_throttle': corner_data['throttle'].mean(),
                'max_brake': corner_data['brake'].max(),
                'avg_steering': abs(corner_data['steering'].mean()),
            }
            
            # Calculate time lost/gained (simplified)
            speed_loss = analysis['entry_speed'] - analysis['min_speed']
            analysis['time_in_corner'] = len(corner_data) / self.sample_rate
            
            corner_analyses.append(analysis)
        
        return corner_analyses
    
    def find_optimal_racing_line(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate optimal racing line based on telemetry
        
        Args:
            df: Telemetry DataFrame with GPS data
            
        Returns:
            Array of optimal line points
        """
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.warning("GPS data not available")
            return np.array([])
        
        # Remove invalid GPS points
        valid_gps = df.dropna(subset=['latitude', 'longitude'])
        
        # Extract coordinates
        coords = valid_gps[['latitude', 'longitude']].values
        speeds = valid_gps['speed'].values
        
        # Weight by speed (faster = better line)
        speed_weight = speeds / speeds.max()
        
        # Apply smoothing filter
        from scipy.ndimage import gaussian_filter1d
        smoothed_lat = gaussian_filter1d(coords[:, 0] * speed_weight, sigma=5)
        smoothed_lon = gaussian_filter1d(coords[:, 1] * speed_weight, sigma=5)
        
        optimal_line = np.column_stack([smoothed_lat, smoothed_lon])
        
        return optimal_line
    
    def generate_insights(self, metrics: Dict[str, float], 
                         corners: List[Dict]) -> List[str]:
        """
        Generate actionable insights for driver improvement
        
        Args:
            metrics: Overall lap metrics
            corners: Corner performance data
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Speed insights
        if metrics['avg_speed'] < 100:
            insights.append("Average speed is low. Focus on maintaining momentum through corners.")
        
        # Throttle insights
        if metrics['full_throttle_pct'] < 30:
            insights.append(f"Full throttle only {metrics['full_throttle_pct']:.1f}% of lap. "
                          "Look for earlier throttle application points.")
        
        # Braking insights
        if metrics['brake_frequency'] > len(corners) * 2:
            insights.append("Excessive braking detected. Try to brake less frequently and more decisively.")
        
        # Smoothness insights
        if metrics['steering_smoothness'] < 70:
            insights.append("Steering inputs are not smooth. Work on progressive steering application.")
        
        # Corner-specific insights
        for corner in corners:
            if corner['exit_speed'] < corner['entry_speed'] * 0.8:
                insights.append(f"Corner {corner['corner_num']}: Slow exit speed. "
                              "Try earlier throttle application.")
            
            if corner['max_brake'] > 80 and corner['avg_throttle'] < 20:
                insights.append(f"Corner {corner['corner_num']}: Long brake-to-throttle transition. "
                              "Work on faster weight transfer.")
        
        return insights


class TelemetryComparator:
    """Compare telemetry between laps or drivers"""
    
    @staticmethod
    def compare_laps(lap1_df: pd.DataFrame, lap2_df: pd.DataFrame) -> Dict:
        """
        Compare two laps
        
        Args:
            lap1_df: First lap telemetry
            lap2_df: Second lap telemetry
            
        Returns:
            Comparison results
        """
        comparison = {
            'lap1_time': lap1_df['timestamp'].max() - lap1_df['timestamp'].min(),
            'lap2_time': lap2_df['timestamp'].max() - lap2_df['timestamp'].min(),
            'speed_delta': lap2_df['speed'].mean() - lap1_df['speed'].mean(),
            'throttle_delta': lap2_df['throttle'].mean() - lap1_df['throttle'].mean(),
        }
        
        comparison['time_delta'] = comparison['lap2_time'] - comparison['lap1_time']
        comparison['faster_lap'] = 1 if comparison['time_delta'] < 0 else 2
        
        return comparison
    
    @staticmethod
    def find_time_differences(lap1_df: pd.DataFrame, 
                            lap2_df: pd.DataFrame) -> np.ndarray:
        """
        Find where time is gained/lost between laps
        
        Args:
            lap1_df: First lap telemetry
            lap2_df: Second lap telemetry
            
        Returns:
            Array of time differences at each point
        """
        # Normalize to same length via interpolation
        from scipy.interpolate import interp1d
        
        n_points = min(len(lap1_df), len(lap2_df))
        
        x1 = np.linspace(0, 1, len(lap1_df))
        x2 = np.linspace(0, 1, len(lap2_df))
        x_common = np.linspace(0, 1, n_points)
        
        speed1_interp = interp1d(x1, lap1_df['speed'].values)(x_common)
        speed2_interp = interp1d(x2, lap2_df['speed'].values)(x_common)
        
        # Calculate cumulative time difference
        time_diff = np.cumsum(1/speed2_interp - 1/speed1_interp)
        
        return time_diff

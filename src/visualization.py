"""
Visualization utilities for racing analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path

sns.set_style("darkgrid")


class RacingVisualizer:
    """Create visualizations for racing analysis"""
    
    @staticmethod
    def plot_racing_line(track_coords: np.ndarray, 
                        racing_line: np.ndarray,
                        title: str = "Racing Line",
                        save_path: Optional[str] = None):
        """
        Plot racing line on track
        
        Args:
            track_coords: Track boundary coordinates
            racing_line: Racing line coordinates
            title: Plot title
            save_path: Optional save path
        """
        fig = go.Figure()
        
        # Track boundaries
        fig.add_trace(go.Scatter(
            x=track_coords[:, 0],
            y=track_coords[:, 1],
            mode='lines',
            name='Track',
            line=dict(color='gray', width=2)
        ))
        
        # Racing line
        fig.add_trace(go.Scatter(
            x=racing_line[:, 0],
            y=racing_line[:, 1],
            mode='lines+markers',
            name='Racing Line',
            line=dict(color='green', width=3),
            marker=dict(size=5)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X Position",
            yaxis_title="Y Position",
            showlegend=True,
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_telemetry_trace(df: pd.DataFrame,
                            columns: List[str] = ['speed', 'throttle', 'brake'],
                            title: str = "Telemetry Trace",
                            save_path: Optional[str] = None):
        """
        Plot telemetry traces
        
        Args:
            df: Telemetry DataFrame
            columns: Columns to plot
            title: Plot title
            save_path: Optional save path
        """
        fig = make_subplots(
            rows=len(columns), cols=1,
            subplot_titles=columns,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Plotly
        
        for i, col in enumerate(columns):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[col],
                    mode='lines',
                    name=col.capitalize(),
                    line=dict(color=colors[i % len(colors)])
                ),
                row=i+1, col=1
            )
        
        fig.update_xaxes(title_text="Time (s)", row=len(columns), col=1)
        fig.update_layout(
            title=title,
            height=200 * len(columns),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_speed_trace_on_track(coords: np.ndarray, 
                                  speeds: np.ndarray,
                                  title: str = "Speed Trace",
                                  save_path: Optional[str] = None):
        """
        Plot speed as color on track
        
        Args:
            coords: Track coordinates
            speeds: Speed values
            title: Plot title
            save_path: Optional save path
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers+lines',
            marker=dict(
                size=8,
                color=speeds,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Speed (km/h)")
            ),
            line=dict(width=2),
            name='Speed Trace'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_corner_analysis(corner_analyses: List[Dict],
                           title: str = "Corner Performance",
                           save_path: Optional[str] = None):
        """
        Plot corner analysis metrics
        
        Args:
            corner_analyses: List of corner analysis dictionaries
            title: Plot title
            save_path: Optional save path
        """
        df = pd.DataFrame(corner_analyses)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Entry/Min/Exit Speed',
                'Time in Corner',
                'Throttle/Brake Usage',
                'Average Steering'
            )
        )
        
        # Speed plot
        corner_nums = df['corner_num']
        fig.add_trace(
            go.Bar(name='Entry Speed', x=corner_nums, y=df['entry_speed'],
                  marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Min Speed', x=corner_nums, y=df['min_speed'],
                  marker_color='red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Exit Speed', x=corner_nums, y=df['exit_speed'],
                  marker_color='green'),
            row=1, col=1
        )
        
        # Time plot
        fig.add_trace(
            go.Bar(x=corner_nums, y=df['time_in_corner'],
                  marker_color='purple'),
            row=1, col=2
        )
        
        # Throttle/Brake plot
        fig.add_trace(
            go.Bar(name='Avg Throttle', x=corner_nums, y=df['avg_throttle'],
                  marker_color='green'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Max Brake', x=corner_nums, y=df['max_brake'],
                  marker_color='red'),
            row=2, col=1
        )
        
        # Steering plot
        fig.add_trace(
            go.Bar(x=corner_nums, y=df['avg_steering'],
                  marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Corner", row=2, col=1)
        fig.update_xaxes(title_text="Corner", row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_lap_comparison(lap1_df: pd.DataFrame,
                          lap2_df: pd.DataFrame,
                          time_diff: np.ndarray,
                          title: str = "Lap Comparison",
                          save_path: Optional[str] = None):
        """
        Plot comparison between two laps
        
        Args:
            lap1_df: First lap telemetry
            lap2_df: Second lap telemetry
            time_diff: Time difference array
            title: Plot title
            save_path: Optional save path
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Speed Comparison',
                'Throttle Comparison',
                'Time Delta'
            ),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Normalize timestamps to percentage
        lap1_pct = np.linspace(0, 100, len(lap1_df))
        lap2_pct = np.linspace(0, 100, len(lap2_df))
        
        # Speed comparison
        fig.add_trace(
            go.Scatter(x=lap1_pct, y=lap1_df['speed'], name='Lap 1 Speed',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=lap2_pct, y=lap2_df['speed'], name='Lap 2 Speed',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Throttle comparison
        fig.add_trace(
            go.Scatter(x=lap1_pct, y=lap1_df['throttle'], name='Lap 1 Throttle',
                      line=dict(color='blue', dash='dash')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=lap2_pct, y=lap2_df['throttle'], name='Lap 2 Throttle',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Time delta
        delta_pct = np.linspace(0, 100, len(time_diff))
        colors = ['green' if x < 0 else 'red' for x in time_diff]
        
        fig.add_trace(
            go.Scatter(x=delta_pct, y=time_diff, name='Time Delta',
                      fill='tozeroy',
                      line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Lap Progress (%)", row=3, col=1)
        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Throttle (%)", row=2, col=1)
        fig.update_yaxes(title_text="Time Delta (s)", row=3, col=1)
        
        fig.update_layout(
            title=title,
            height=900,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def create_dashboard(telemetry_df: pd.DataFrame,
                        metrics: Dict,
                        corners: List[Dict],
                        insights: List[str],
                        save_path: str):
        """
        Create comprehensive dashboard
        
        Args:
            telemetry_df: Telemetry DataFrame
            metrics: Lap metrics
            corners: Corner analyses
            insights: Insights list
            save_path: Path to save HTML dashboard
        """
        from plotly.subplots import make_subplots
        
        # Create multi-plot dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Speed Trace',
                'Key Metrics',
                'Throttle & Brake',
                'Corner Performance',
                'Steering Input',
                'Insights'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Speed trace
        fig.add_trace(
            go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['speed'],
                      name='Speed', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Key metrics indicators
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics['lap_time'],
                title={"text": "Lap Time (s)"},
                delta={'reference': metrics.get('target_time', metrics['lap_time'])}
            ),
            row=1, col=2
        )
        
        # Throttle & Brake
        fig.add_trace(
            go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['throttle'],
                      name='Throttle', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['brake'],
                      name='Brake', line=dict(color='red')),
            row=2, col=1
        )
        
        # Corner performance
        if corners:
            corner_df = pd.DataFrame(corners)
            fig.add_trace(
                go.Bar(x=corner_df['corner_num'], y=corner_df['exit_speed'],
                      name='Exit Speed', marker_color='green'),
                row=2, col=2
            )
        
        # Steering
        fig.add_trace(
            go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['steering'],
                      name='Steering', line=dict(color='orange')),
            row=3, col=1
        )
        
        # Insights table
        insights_text = "<br>".join([f"• {insight}" for insight in insights])
        fig.add_trace(
            go.Table(
                header=dict(values=["Insights"]),
                cells=dict(values=[[insights_text]])
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title="DriveOS Racing Analysis Dashboard",
            height=1200,
            showlegend=True
        )
        
        fig.write_html(save_path)
        return fig

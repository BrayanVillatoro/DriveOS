"""
Main CLI application for DriveOS
"""
import argparse
import logging
import sys
from pathlib import Path

from .config import config
from .inference import BatchProcessor
from .telemetry_processor import TelemetryProcessor
from .visualization import RacingVisualizer

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='DriveOS - Intelligent Racing Line Analysis System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze video command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze racing video')
    analyze_parser.add_argument('video', type=str, help='Path to video file')
    analyze_parser.add_argument('--telemetry', type=str, help='Path to telemetry CSV file')
    analyze_parser.add_argument('--output', type=str, help='Output video path')
    analyze_parser.add_argument('--model', type=str, default=config.MODEL_PATH,
                               help='Path to model weights')
    
    # Analyze telemetry command
    telemetry_parser = subparsers.add_parser('telemetry', 
                                             help='Analyze telemetry data')
    telemetry_parser.add_argument('file', type=str, help='Path to telemetry CSV file')
    telemetry_parser.add_argument('--output', type=str, help='Output report path')
    
    # Compare laps command
    compare_parser = subparsers.add_parser('compare', help='Compare two laps')
    compare_parser.add_argument('lap1', type=str, help='First lap telemetry CSV')
    compare_parser.add_argument('lap2', type=str, help='Second lap telemetry CSV')
    compare_parser.add_argument('--output', type=str, help='Output comparison report')
    
    # Start API server command
    api_parser = subparsers.add_parser('serve', help='Start API server')
    api_parser.add_argument('--host', type=str, default=config.API_HOST,
                           help='API host')
    api_parser.add_argument('--port', type=int, default=config.API_PORT,
                           help='API port')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_video(args)
    elif args.command == 'telemetry':
        analyze_telemetry(args)
    elif args.command == 'compare':
        compare_laps(args)
    elif args.command == 'serve':
        start_server(args)
    else:
        parser.print_help()


def analyze_video(args):
    """Analyze racing video"""
    logger.info(f"Analyzing video: {args.video}")
    
    # Default output to Videos folder if not specified
    if args.output:
        output_path = args.output
    else:
        videos_folder = Path.home() / "Videos"
        output_path = str(videos_folder / f"analyzed_{Path(args.video).name}")
    
    processor = BatchProcessor(args.model)
    stats = processor.process_video(
        args.video,
        output_path,
        args.telemetry
    )
    
    logger.info(f"Analysis complete!")
    logger.info(f"Processed {stats['total_frames']} frames")
    logger.info(f"Average inference time: {stats['avg_inference_time']:.2f}ms")
    logger.info(f"Output saved to: {output_path}")
    
    # Generate telemetry insights if available
    if args.telemetry:
        tel_processor = TelemetryProcessor()
        df = tel_processor.load_from_csv(args.telemetry)
        metrics = tel_processor.calculate_racing_metrics(df)
        corners = tel_processor.detect_corners(df)
        corner_analysis = tel_processor.analyze_corner_performance(df, corners)
        insights = tel_processor.generate_insights(metrics, corner_analysis)
        
        logger.info("\n=== Racing Insights ===")
        for insight in insights:
            logger.info(f"  • {insight}")
        
        # Create dashboard
        visualizer = RacingVisualizer()
        dashboard_path = str(Path(output_path).parent / "dashboard.html")
        visualizer.create_dashboard(df, metrics, corner_analysis, insights, dashboard_path)
        logger.info(f"Dashboard saved to: {dashboard_path}")


def analyze_telemetry(args):
    """Analyze telemetry data"""
    logger.info(f"Analyzing telemetry: {args.file}")
    
    processor = TelemetryProcessor()
    df = processor.load_from_csv(args.file)
    
    # Calculate metrics
    metrics = processor.calculate_racing_metrics(df)
    corners = processor.detect_corners(df)
    corner_analysis = processor.analyze_corner_performance(df, corners)
    insights = processor.generate_insights(metrics, corner_analysis)
    
    # Print results
    logger.info("\n=== Lap Metrics ===")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")
    
    logger.info(f"\n=== Corner Analysis ({len(corner_analysis)} corners) ===")
    for corner in corner_analysis:
        logger.info(f"  Corner {corner['corner_num']}:")
        logger.info(f"    Entry speed: {corner['entry_speed']:.1f} km/h")
        logger.info(f"    Min speed: {corner['min_speed']:.1f} km/h")
        logger.info(f"    Exit speed: {corner['exit_speed']:.1f} km/h")
    
    logger.info("\n=== Insights ===")
    for insight in insights:
        logger.info(f"  • {insight}")
    
    # Create visualizations
    if args.output:
        visualizer = RacingVisualizer()
        visualizer.create_dashboard(df, metrics, corner_analysis, insights, args.output)
        logger.info(f"\nReport saved to: {args.output}")


def compare_laps(args):
    """Compare two laps"""
    logger.info(f"Comparing laps: {args.lap1} vs {args.lap2}")
    
    from .telemetry_processor import TelemetryComparator
    
    processor = TelemetryProcessor()
    lap1_df = processor.load_from_csv(args.lap1)
    lap2_df = processor.load_from_csv(args.lap2)
    
    comparator = TelemetryComparator()
    comparison = comparator.compare_laps(lap1_df, lap2_df)
    time_diff = comparator.find_time_differences(lap1_df, lap2_df)
    
    # Print results
    logger.info("\n=== Lap Comparison ===")
    logger.info(f"  Lap 1 time: {comparison['lap1_time']:.3f}s")
    logger.info(f"  Lap 2 time: {comparison['lap2_time']:.3f}s")
    logger.info(f"  Time delta: {comparison['time_delta']:.3f}s")
    logger.info(f"  Faster lap: Lap {comparison['faster_lap']}")
    logger.info(f"  Speed delta: {comparison['speed_delta']:.2f} km/h")
    
    # Create comparison visualization
    if args.output:
        visualizer = RacingVisualizer()
        visualizer.plot_lap_comparison(lap1_df, lap2_df, time_diff, 
                                      save_path=args.output)
        logger.info(f"\nComparison report saved to: {args.output}")


def start_server(args):
    """Start API server"""
    logger.info(f"Starting DriveOS API server on {args.host}:{args.port}")
    
    import uvicorn
    from .api import app
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

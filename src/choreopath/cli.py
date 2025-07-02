import click
import pandas as pd

from .video import Video
from .svg_generator import SVGGenerator
from .tracking_data import TrackingData

@click.group()
def cli():
    pass

@cli.command(help='Track body parts from video and save to CSV')
@click.argument('src', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('dst', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option("--min-detection-confidence", type=float, default=0.5)
@click.option("--min-tracking-confidence", type=float, default=0.5)
def track(src, dst, min_detection_confidence, min_tracking_confidence):
    video = Video(src)
    click.echo(f"Tracking poses in {src}")
    click.echo("Found {} frames".format(video.total_frames()))
    click.echo("FPS: {}".format(video.fps()))
    click.echo("Tracking poses with min detection confidence: {} and min tracking confidence: {}".format(min_detection_confidence, min_tracking_confidence))

    tracking_data = video.track_poses(min_detection_confidence, min_tracking_confidence)

    if tracking_data:
        df = pd.DataFrame(tracking_data)
        df.to_csv(dst, index=False)
        click.echo(f"Tracking data saved to: {dst}")
        click.echo(f"Total data points: {len(tracking_data)}")
    else:
        click.echo("No tracking data found. Check if there are people visible in the video.")

@cli.command(help='Generate SVG trajectories from body tracking data')
@click.argument('src', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('dst', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option("--min-visibility", help='Minimum visibility threshold (0.0-1.0, default: 0.5)', type=float, default=0.5)
@click.option("--width", help='SVG width in pixels (default: 1280)', type=int, default=1280)
@click.option("--height", help='SVG height in pixels (default: 720)', type=int, default=720)
@click.option('--no-legend', help='Disable legend display', is_flag=True)
def draw(src, dst, min_visibility, width, height, no_legend):
    click.echo(f"Generating SVG to {dst} from {src}")
    tracking_data = TrackingData(src)
    tracking_data.load()
    tracking_data.reject_invisible(min_visibility)
    tracking_data.sort_by_frame_and_landmark_id()

    generator = SVGGenerator(width=width, height=height, show_legend=not no_legend)
    svg_document = generator.generate(tracking_data)
    svg_document.write(dst, encoding='utf-8', xml_declaration=True)
    click.echo(f"SVG saved to: {dst}")

@cli.command(help='Analyze tracking data')
@click.argument('src', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option('--animation', type=click.Path(exists=False, writable=True, dir_okay=False), default='tracking_data.mp4')
@click.option('--fps', type=int, default=24)
def analyze(src, animation, fps):
    tracking_data = TrackingData(src)
    tracking_data.load()

    audit = tracking_data.audit()

    click.echo(f"{src}:")
    click.echo(f"Total data points: {audit['total_rows']:,}")
    click.echo(f"Frames processed: {audit['unique_frames']:,}")
    click.echo(f"Duration: {audit['timestamp_range'][1]:.2f} seconds")
    click.echo(f"Average visibility: {audit['visibility_stats']['mean']:.3f}")
    
    if audit['anomalies']:
        click.echo(f"\n⚠ Found {len(audit['anomalies'])} potential issues:")
        for anomaly in audit['anomalies']:
            click.echo(f"  - {anomaly}")
    else:
        click.echo("\n✓ No anomalies detected - data looks good!")

    click.echo(f"\nGenerating tracking data animation to: {animation}")
    tracking_data.to_animation(animation, fps)

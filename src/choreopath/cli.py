import click
import pandas as pd

from .video import Video
from .svg_generator import SVGGenerator
from .tracking_data import TrackingData
from .video_overlay_renderer import VideoOverlayRenderer
from .colors import Palette

@click.group()
def cli():
    pass

@cli.command(help='Track body parts from video and save to CSV')
@click.argument('src', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('dst', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option("--min-detection-confidence", type=float, default=0.5)
@click.option("--min-tracking-confidence", type=float, default=0.5)
def track(src, dst, min_detection_confidence, min_tracking_confidence):
    video = Video(src, min_detection_confidence, min_tracking_confidence)
    click.echo(f"Tracking poses in {src}")
    click.echo("Found {} frames".format(video.total_frames()))
    click.echo("FPS: {}".format(video.fps()))
    click.echo("Tracking poses with min detection confidence: {} and min tracking confidence: {}".format(min_detection_confidence, min_tracking_confidence))

    tracking_data = video.track_poses()

    if tracking_data:
        df = pd.DataFrame(tracking_data)
        df.to_csv(dst, index=False)
        click.echo(f"Tracking data saved to: {dst}")
        click.echo(f"Total data points: {len(tracking_data)}")
    else:
        click.echo("No tracking data found. Check if there are people visible in the video.")
    
    video.close()

@cli.command(help='Generate SVG trajectories from body tracking data')
@click.argument('src', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('dst', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option("--palette", help='Palette to use (default: default, white_and_gold)', type=str, default='default')
@click.option("--min-visibility", help='Minimum visibility threshold (0.0-1.0, default: 0.5)', type=float, default=0.5)
@click.option("--width", help='SVG width in pixels (default: 1280)', type=int, default=1280)
@click.option("--height", help='SVG height in pixels (default: 720)', type=int, default=720)
@click.option('--no-legend', help='Disable legend display', is_flag=True)
def draw(src, dst, palette, min_visibility, width, height, no_legend):
    click.echo(f"Generating SVG to {dst} from {src}")
    tracking_data = TrackingData(src)
    tracking_data.load()
    tracking_data.reject_invisible(min_visibility)
    tracking_data.sort_by_frame_and_landmark_id()

    generator = SVGGenerator(width=width, palette=Palette(palette), height=height, show_legend=not no_legend)
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

@cli.command(help='Generate video with progressive pose path overlays')
@click.argument('video', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output', type=click.Path(writable=True, dir_okay=False))
@click.option('--palette', help='Palette to use (default: default, white_and_gold)', type=str, default='default')
@click.option('--min-detection-confidence', type=float, default=0.5, help='Minimum detection confidence')
@click.option('--min-tracking-confidence', type=float, default=0.5, help='Minimum tracking confidence')
@click.option('--min-visibility', type=float, default=0.5, help='Minimum visibility threshold')
@click.option('--line-thickness', type=int, default=1, help='Path line thickness in pixels')
@click.option('--no-current-point', is_flag=True, help='Disable current position marker')
@click.option('--paths-only', is_flag=True, help='Render only paths')
def overlay(video, output, palette, min_detection_confidence, min_tracking_confidence, min_visibility, line_thickness, no_current_point, paths_only):
    click.echo(f"Generating video overlay from {video} to {output}")
    if paths_only:
        click.echo("Mode: Paths only (black background)")

    video = Video(video, min_detection_confidence, min_tracking_confidence)

    renderer = VideoOverlayRenderer(
        palette=Palette(palette),
        line_thickness=line_thickness,
        show_current_point=not no_current_point,
        min_visibility=min_visibility,
        paths_only=paths_only,
    )

    renderer.render_overlay(video=video, output_path=output)

    video.close()

    click.echo(f"Video overlay saved to: {output}")

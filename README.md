# choreopath

Transform human movement into generative art. A Python tool that uses MediaPipe
to track body poses from video and creates SVG visualizations of motion trajectories -
perfect for pen plotters, laser cutters, and digital art.

## Features

- Track 33 body landmarks using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) Pose
- Process video files with [OpenCV](https://opencv.org) to extract tracking data
- Export tracking data to CSV with coordinates and visibility scores
- Generate SVG visualizations with hierarchical body part grouping, ready to be used in Inkscape
- Analyze tracking data quality with detailed reports
- Create animated visualizations of pose tracking
- Command-line interface

## Installation

choreopath is pushed to Pypi at each new release. You can `pip install choreopath` to install it. Beware though, some versions of Python are incompatible with it as MediaPipe isn't currently available for Python >=3.13.

Instead, it is suggested to use [uv](https://docs.astral.sh/uv/) and run choreopath like this:

`uvx --python 3.12 choreopath`

## Usage

### Track poses from video

```bash
choreopath track video.mp4 tracking_data.csv
```

### Track with custom confidence thresholds

```bash
choreopath track video.mp4 tracking_data.csv --min-detection-confidence 0.7 --min-tracking-confidence 0.7
```

### Generate SVG visualization

```bash
choreopath draw tracking_data.csv output.svg
```

### Generate SVG with custom dimensions

```bash
choreopath draw tracking_data.csv output.svg --width 1920 --height 1080 --min-visibility 0.7
```

### Analyze tracking data

```bash
choreopath analyze tracking_data.csv --animation output.mp4 --fps 30
```

## Origin Story

This project was created when working on a generative art project. I used
generated videos of dancers to create SVG that I could then plot using a pen plotter.

See example files:

[original video](examples/fall-recovery-4.mp4)
[animated tracking data](examples/fall-recovery-4-animation.mp4)
![examples/fall-recovery-4.svg](https://github.com/marcw/choreopath/blob/main/examples/fall-recovery-4.svg)

## License

This software is under a MIT license. Please see [LICENSE.md](LICENSE.md)

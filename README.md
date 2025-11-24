# Soccer Goal Detection System

A hybrid ML + computer vision system that automatically detects goals in soccer videos using YOLO for object detection and Hough transforms for goal structure analysis.

## Inspiration & Related Work

### Football Goal Detector

    Project: Open-source football goal detection system

    Relation: Inspiration for goal detection approaches and computer vision techniques

    Source: julianfromano/football-goal-detector

## Technologies Used

    YOLOv8: Player and ball detection via Ultralytics

    OpenCV: Computer vision operations and image processing

    Supervision: Detection utilities and tracking

    Roboflow Inference SDK: Cloud-based goalpost detection

## Quick Setup & Run

### Option 1: Using `uv` (Recommended)

1. **Install uv** (if you don't have it):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or with pip:
pip install uv
```

2. **Create virtual environment and install dependencies**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. **Run the system**:
```bash
python soccer_analysis.py your_video.mp4 --roboflow-key YOUR_API_KEY
```

### Option 2: Using Python's Built-in venv

1. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -e .
```

3. **Run the system**:
```bash
python soccer_analysis.py your_video.mp4 --roboflow-key YOUR_API_KEY
```

## Usage Examples

### Basic usage:
```bash
python soccer_analysis.py match.mp4 --roboflow-key rf_xxxxxxx
```

### Advanced options:
```bash
python soccer_analysis.py match.mp4 \
    --roboflow-key rf_xxxxxxx \
    --max-frames 1000 \
    --start-frame 500 \
    --output-name my_analysis \
    --debug
```

### Command Line Arguments:
- `input_video`: Path to input video file (required)
- `--roboflow-key`: Roboflow API key for goalpost detection (required)
- `--max-frames`: Maximum frames to process (default: 500)
- `--start-frame`: Start frame number (default: 0)
- `--output-name`: Base name for output files
- `--debug`: Enable debug mode

## Output

The system generates:
- `OUTPUT_{name}.mp4` - Processed video with annotations
- `OUTPUT_{name}_goal_moments/` - Folder with goal moment images
- Console logs with detection status

## Requirements

- Python 3.8+
- Roboflow API key (get from [roboflow.com](https://roboflow.com))

## Deactivating Virtual Environment

When done, deactivate the virtual environment:
```bash
deactivate
```

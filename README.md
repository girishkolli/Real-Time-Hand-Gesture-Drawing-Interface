# Real-Time Hand Gesture Drawing Interface

A webcam-based virtual whiteboard that lets you draw in the air using hand gestures tracked with MediaPipe.

## Features

- Real-time hand tracking with webcam input
- Gesture-based mode switching:
  - **Selection mode** (index + middle fingers up)
  - **Drawing mode** (index finger up)
- Color tools: **Red**, **Green**, **Blue**
- **Eraser** tool with thicker stroke
- Clear canvas and quit shortcuts

## Requirements

- Python 3.8+
- Webcam
- Python packages:
  - `opencv-python`
  - `mediapipe`
  - `numpy`

## Installation

```bash
pip install opencv-python mediapipe numpy
```

## Run the project

From the repository root:

```bash
python real_time_hand_gesture_drawing_interface.py
```

## Controls

- **Two fingers up (index + middle):** Selection mode
- **One finger up (index only):** Drawing mode
- Move your index finger into the top toolbar to select:
  - Red
  - Green
  - Blue
  - Eraser
- Press **`c`** to clear the canvas
- Press **`q`** to quit

## Project Structure

- `real_time_hand_gesture_drawing_interface.py` - main application
- `selection menu/` - toolbar header images for tools

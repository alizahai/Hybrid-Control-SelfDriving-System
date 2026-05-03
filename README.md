# Hybrid Control Self-Driving System

A real-time autonomous car perception system that combines classical image 
processing with deep learning to detect road boundaries, identify obstacles, 
and make driving decisions from video footage.

The system supports two modes — a single front camera pipeline and a 
three-camera multi-stream pipeline with a fused decision engine.

---

## What it does

- Detects road lane boundaries and physical road edges using Canny edge 
  detection and Hough line transform — works on both painted lane markings 
  and unmarked roads separated by grass, kerb, or concrete
- Detects obstacles in real time using YOLOv8s — persons, vehicles, and animals
- Classifies each detection by distance (VERY NEAR, NEAR, FAR, VERY FAR) 
  and whether it is on the road or roadside
- Makes driving decisions: STOP, SLOW DOWN, or MOVE FORWARD
- Computes steering corrections: STRAIGHT, SLIGHT LEFT/RIGHT, TURN LEFT/RIGHT
- In multi-stream mode, combines front, left, and right camera views into 
  one fused driving command

---

## Project structure

```
project/
│
├── LaneDetection.py        # Road boundary detection (front camera)
├── object_detection.py     # YOLOv8s obstacle detection and road zone check
├── decision_making.py      # Rule-based driving decisions and steering logic
├── autonomous_car.py       # Single-stream pipeline — main entry point
│
└── multi_stream/
    ├── main.py             # Entry point — choose single or multi-stream mode
    ├── config.py           # Video paths, frame sizes, stream assignments
    ├── lane_detector.py    # Class-based lane detector for front camera
    ├── side_detector.py    # Road boundary detector for left/right cameras
    ├── stream_manager.py   # Orchestrates 3 streams, YOLO thread, grid display
    ├── walkthrough.md      # Detailed technical walkthrough of multi-stream
    └── README.md           # Multi-stream module documentation
```

## How to run

### Single-stream mode

Processes one video at a time from the front camera:

```bash
python autonomous_car.py
```

### Multi-stream mode

Processes front, left, and right camera videos simultaneously:

```bash
python multi_stream/main.py
# select [2] Multi-Stream
```

Or run `main.py` from the multi_stream folder and select either mode from 
the same menu.

---

## Controls

| Key | Action             |
|-----|--------------------|
| Q   | Quit               |
| P   | Pause / Resume     |
| N   | Skip to next video |

---

## Display

**Single-stream** opens two windows — the final annotated frame and an edge map showing detected boundaries alongside Canny edges.

**Multi-stream** opens a single 2×2 grid window:

```
┌─────────────┬─────────────┐
│    FRONT    │    STATS    │
│             │ fused cmd   │
├─────────────┼─────────────┤
│    LEFT     │    RIGHT    │
│             │             │
└─────────────┴─────────────┘
```

Each cell shows the video frame with lane overlay, YOLO bounding boxes
color-coded by distance, and per-stream decision text. The stats panel
shows the final fused driving command and per-camera status.

## Lane detection approach

Rather than relying on painted lane markings, the system detects the 
physical boundary separating the road surface from its surroundings — 
grass, kerb, concrete pavement, or gravel. This makes it work across 
all road types in the dataset.

Pipeline per frame:

1. Grayscale conversion and Gaussian blur
2. Canny edge detection
3. Trapezoidal ROI mask isolating the road region
4. Hough probabilistic line transform
5. Lines filtered by slope and position to separate left and right boundaries
6. Weighted polynomial fit — points lower in the frame (closer to camera) 
   weighted more heavily
7. EMA temporal smoothing (alpha = 0.82) to reduce frame-to-frame jitter

---

## Object detection approach

YOLOv8s runs in a background thread to avoid blocking the main loop. 
Only road-relevant classes are kept — persons, cars, motorcycles, buses, 
trucks, and animals. Each detection is checked against a fixed trapezoidal 
road zone using a pixel mask test on the bottom-center of its bounding box.

Distance is estimated from apparent size — the bounding box height as a 
fraction of frame height. Objects closer to the camera appear larger and 
take up more of the frame.

---

## Decision logic

Decisions follow a priority cascade applied to on-road detections only:

| Priority | Condition                | Decision     |
|----------|--------------------------|--------------|
| 1        | Person NEAR or VERY NEAR | STOP         |
| 2        | Any object VERY NEAR     | STOP         |
| 3        | Any object NEAR          | SLOW DOWN    |
| 4        | Person FAR               | SLOW DOWN    |
| 5        | All objects FAR or none  | MOVE FORWARD |

Steering is computed from the offset between the detected lane center and 
the frame center. Offsets under 30px give STRAIGHT, 30–80px give a mild 
correction, and over 80px give a sharp turn command.

In multi-stream mode, `fused_decide()` combines front, left, and right 
camera results. The front camera determines STOP or SLOW DOWN. Side cameras 
influence steering — if one side is blocked, the system steers toward the 
clear side. If all sides are blocked, the system stops.

---

## Logging

Both modes write a CSV log every 30 frames:

**Single-stream** → `detection_log.txt`  
**Multi-stream** → `multi_stream/multi_stream_log.txt`

Columns: `frame, fps, lane_center, left_detected, right_detected, decision, steering, on_road_count`

---

## Requirements
opencv-python
numpy
ultralytics

Install with:

```bash
pip install opencv-python numpy ultralytics
```

YOLOv8s weights (`yolov8s.pt`) download automatically on first run via 
the ultralytics library. Place your video files in the path specified in 
`autonomous_car.py` or `multi_stream/config.py`.

---

## Techniques used

| Technique            | Purpose                                     |
|----------------------|---------------------------------------------|
| Grayscale conversion | Prepares frame for edge detection           |
| Gaussian blur        | Reduces noise before Canny                  |
| Canny edge detection | Finds road boundary edges                   |
| ROI masking          | Focuses processing on the road region       |
| Hough line transform | Detects straight line segments from edges   |
| Weighted polyfit     | Fits one best-fit boundary line per side    |
| EMA smoothing        | Stabilizes detections across frames         |
| YOLOv8s inference    | Real-time obstacle detection                |
| Polygon mask test    | Determines if an object is on the road      |
| Multi-threading      | Runs YOLO without blocking the display loop |
| Image blending       | Overlays lane lines onto the original frame |

---

## Limitations

- Lane detection relies on edge contrast between road and surroundings. 
  Roads where both sides have similar texture to the road surface produce 
  weak edges and may cause missed detections.
- Distance estimation uses apparent bounding box size, not a depth sensor. 
  Results are approximate and may be less accurate for very tall objects 
  like trucks.
- The system was tested under daytime, dry conditions. Performance under 
  rain, night, or harsh lighting has not been evaluated.
- Wide roads where boundaries appear at shallow angles may fall outside 
  the slope filter and go undetected.

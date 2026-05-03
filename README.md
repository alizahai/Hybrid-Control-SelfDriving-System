# Lane Detection Module

Detects road boundaries from a front-facing camera using classical image processing.
Works on both painted lane markings and unmarked roads — it detects the physical
edge separating the road surface from its surroundings (grass, kerb, or pavement).

## Pipeline
1. **Grayscale + blur** — reduces noise before edge detection
2. **Canny edge detection** — finds intensity changes at road boundaries
3. **Trapezoidal ROI mask** — isolates the road region, ignores sky and surroundings
4. **Hough line transform** — detects straight line segments within the ROI
5. **Slope filtering** — separates left boundary (negative slope) from right (positive slope)
6. **Weighted polyfit** — fits one best-fit line per side, points lower in frame weighted more
7. **EMA smoothing** — blends previous and current detections to reduce jitter

## Output

Each frame returns a `lane_info` dictionary:

```python
lane_info = {
    "left_line":   (x_bottom, y_bottom, x_top, y_top),  # or None
    "right_line":  (x_bottom, y_bottom, x_top, y_top),  # or None
    "lane_center": 320   # x-coordinate of road center at bottom of frame
}
```

## Usage

```python
import LaneDetection

LaneDetection.reset_state()  # call between videos

edge_map, result, lane_info = LaneDetection.process(frame)
```

## Key parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `SMOOTH_ALPHA` | 0.82 | Higher = smoother but slower to react |
| `MISS_LIMIT` | 15 | Frames before resetting a lost lane |
| `ROI_TOP_RATIO` | 0.35 | How high the ROI reaches (fraction of frame height) |
| `Y_TOP_FLOOR` | 0.38 | Maximum height a detected line can reach |
| `HOUGH_THRESHOLD` | 35 | Minimum votes for a valid Hough line |

## Limitations

- Weak edges on roads where boundary texture matches the road surface
- Wide roads where boundaries appear at shallow angles may be missed
- Tested under daytime only

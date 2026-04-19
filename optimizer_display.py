import cv2
import time
import numpy as np
import os
import glob

# ─────────────────────────────────────────────────────────────
# DIP Project — Member D (Aleena)
# Module: Optimization & Display
# ─────────────────────────────────────────────────────────────

# Global variable for FPS tracking
prev_time = time.time()


def compute_fps():
    """
    Calculate real-time frames per second.
    Returns: float FPS value
    """
    global prev_time
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    return round(fps, 2)


def resize_frame(frame, width=640):
    """
    Resize frame to target width while maintaining aspect ratio.
    Args: frame = input image, width = target width in pixels
    Returns: resized frame
    """
    h, w = frame.shape[:2]
    scale = width / w
    new_size = (width, int(h * scale))
    return cv2.resize(frame, new_size)


def should_run_yolo(frame_count, every_n=3):
    """
    Control YOLO execution frequency to save processing time.
    Args: frame_count = current frame number, every_n = run every Nth frame
    Returns: True if YOLO should run this frame
    """
    return frame_count % every_n == 0


def benchmark_stage(func, *args):
    """
    Measure how long a single pipeline stage takes in milliseconds.
    Args: func = function to measure, *args = its arguments
    Returns: (result, time_taken_in_ms)
    """
    start = time.time()
    result = func(*args)
    elapsed = (time.time() - start) * 1000
    return result, round(elapsed, 2)


def adaptive_optimizer(current_fps):
    """
    Automatically adjust optimization level based on current FPS.
    Args: current_fps = measured FPS
    Returns: (skip_rate, resize_width, status_message)
    """
    if current_fps < 10:
        return 5, 480, "CRITICAL — Aggressive optimization ON"
    elif current_fps < 15:
        return 3, 640, "WARNING — Moderate optimization ON"
    else:
        return 2, 640, "GOOD — Normal operation"


def display_debug(original, fps, frame_count, video_name="",
                  edges=None, lane_overlay=None, decision="FORWARD"):
    """
    Build and show the full debug dashboard window.
    Args:
        original     = raw resized frame
        fps          = current FPS float
        frame_count  = current frame number
        video_name   = name of current video being processed
        edges        = edge detected frame — from Member A (placeholder for now)
        lane_overlay = lane lines drawn frame — from Member A (placeholder for now)
        decision     = driving decision string — from Member C (placeholder for now)
    """

    h, w = original.shape[:2]

    # ── If teammate outputs not ready yet, use placeholders ──
    if edges is None:
        edges = np.zeros((h, w), dtype=np.uint8)
    if lane_overlay is None:
        lane_overlay = original.copy()

    # Convert grayscale edges to BGR for stacking
    if len(edges.shape) == 2:
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        edges_bgr = edges

    # ── Decision color coding ──
    color_map = {
        "FORWARD": (0, 255, 0),
        "STOP":    (0, 0, 255),
        "LEFT":    (0, 255, 255),
        "RIGHT":   (255, 165, 0),
    }
    dec_color = color_map.get(decision, (255, 255, 255))

    # ── Info Panel (bottom right) ──
    info_panel = np.zeros((h, w, 3), dtype=np.uint8)

    # System Info title
    cv2.putText(info_panel, "SYSTEM INFO", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

    # Decision
    cv2.putText(info_panel, "DECISION:", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(info_panel, decision, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, dec_color, 3)

    # FPS
    fps_color = (0, 255, 0) if fps >= 15 else (0, 0, 255)
    cv2.putText(info_panel, f"FPS: {fps}", (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

    # Frame counter
    cv2.putText(info_panel, f"Frame: {frame_count}", (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Video name
    short_name = video_name[:30] + "..." if len(video_name) > 30 else video_name
    cv2.putText(info_panel, f"Video: {short_name}", (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # FPS health bar
    bar_length = int((min(fps, 30) / 30) * (w - 40))
    bar_color = (0, 255, 0) if fps >= 15 else (0, 0, 255)
    cv2.rectangle(info_panel, (20, 285), (20 + bar_length, 305),
                  bar_color, -1)
    cv2.rectangle(info_panel, (20, 285), (w - 20, 305),
                  (255, 255, 255), 1)
    cv2.putText(info_panel, "FPS Health", (20, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Add labels to each panel ──
    cv2.putText(original,     "Original Feed",  (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(edges_bgr,    "Edge Detection", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(lane_overlay, "Lane Detection", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Stack 4 panels into 2x2 grid ──
    top_row    = cv2.hconcat([original, edges_bgr])
    bottom_row = cv2.hconcat([lane_overlay, info_panel])
    dashboard  = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("Self-Driving Debug Dashboard — Member D (Aleena)", dashboard)


# ─────────────────────────────────────────────────────────────
# MAIN — Automatically process ALL videos in data folder
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Find all videos automatically ──
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(BASE_DIR, "data")
    video_files = glob.glob(os.path.join(DATA_DIR, "*.mp4"))

    if not video_files:
        print("ERROR: No videos found in data folder.")
        print(f"Looking in: {DATA_DIR}")
        exit()

    # Sort videos by name for consistent order
    video_files.sort()

    print(f"Found {len(video_files)} videos:")
    for i, v in enumerate(video_files):
        print(f"  [{i + 1}] {os.path.basename(v)}")

    print("\nPress Q to skip to next video.")
    print("Starting in 2 seconds...\n")
    time.sleep(2)

    # ── Store summary results for all videos ──
    all_summaries = []

    # ── Process each video one by one ──
    for video_index, video_path in enumerate(video_files):

        video_name = os.path.basename(video_path)
        print(f"\n{'=' * 55}")
        print(f"Video [{video_index + 1}/{len(video_files)}]: {video_name}")
        print(f"{'=' * 55}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"SKIPPING — Could not open: {video_name}")
            continue

        frame_count = 0
        fps_log     = []
        skipped     = False

        # Reset FPS timer for each new video
        prev_time = time.time()

        while True:
            ret, frame = cap.read()

            # Video ended naturally
            if not ret:
                print(f"Finished: {video_name}")
                break

            frame_count += 1

            # ── Optimization steps ──
            frame = resize_frame(frame, width=640)
            fps   = compute_fps()
            fps_log.append(fps)

            skip_rate, _, status = adaptive_optimizer(fps)
            run_yolo = should_run_yolo(frame_count, every_n=skip_rate)

            # ── Pipeline placeholders ──
            # These will be replaced with real teammate outputs later
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges, edge_time = benchmark_stage(
                cv2.Canny, gray, 50, 150
            )

            decisions = ["FORWARD", "LEFT", "RIGHT", "STOP"]
            decision  = decisions[(frame_count // 30) % 4]

            # ── Show dashboard ──
            display_debug(
                frame.copy(), fps, frame_count,
                video_name=video_name,
                edges=edges,
                lane_overlay=frame.copy(),
                decision=decision
            )

            # ── Print to terminal every 30 frames ──
            if frame_count % 30 == 0:
                avg_fps = round(sum(fps_log[-30:]) / 30, 2)
                print(f"  Frame {frame_count:4d} | "
                      f"FPS: {fps:5.1f} | "
                      f"Avg: {avg_fps:5.1f} | "
                      f"YOLO: {'YES' if run_yolo else 'NO ':3s} | "
                      f"{status}")

            # ── Q to skip to next video ──
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Manually skipped: {video_name}")
                skipped = True
                break

        # ── Summary for this video ──
        if fps_log:
            avg = round(sum(fps_log) / len(fps_log), 2)
            summary = {
                "video":   video_name,
                "frames":  frame_count,
                "avg_fps": avg,
                "min_fps": min(fps_log),
                "max_fps": max(fps_log),
                "skipped": skipped
            }
            all_summaries.append(summary)

            print(f"\n  ── Summary ──")
            print(f"  Total Frames : {frame_count}")
            print(f"  Average FPS  : {avg}")
            print(f"  Min FPS      : {min(fps_log)}")
            print(f"  Max FPS      : {max(fps_log)}")

        cap.release()

    # ── Final report for ALL videos ──
    cv2.destroyAllWindows()

    print(f"\n{'=' * 55}")
    print("ALL VIDEOS PROCESSED — FULL REPORT")
    print(f"{'=' * 55}")
    print(f"{'Video':<35} {'Frames':>7} {'Avg FPS':>8} {'Min':>6} {'Max':>6}")
    print(f"{'-' * 55}")

    for s in all_summaries:
        name = s['video'][:33]
        print(f"{name:<35} {s['frames']:>7} "
              f"{s['avg_fps']:>8} {s['min_fps']:>6} {s['max_fps']:>6}")

    print(f"{'=' * 55}")
    print("Done. All videos complete.")
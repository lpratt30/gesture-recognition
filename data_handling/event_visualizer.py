"""Generate heatmaps, summaries, and MP4 previews for raw event streams."""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
from data_handling.event_core import (
    EVENT_HEIGHT,
    EVENT_WIDTH,
    FRAME_WINDOW_MS,
    infer_timestamps_ms,
    infer_duration_ms,
    iter_input_files,
    load_events,
    normalize_timestamps_ms,
    render_event_frame,
    split_event_windows,
)
from data_handling.project_paths import EVENT_VISUALIZATIONS_DIR, RAW_EVENT_STREAMS_DIR, ensure_dir


DEFAULT_INPUT = RAW_EVENT_STREAMS_DIR
DEFAULT_OUTPUT_DIR = EVENT_VISUALIZATIONS_DIR
VIDEO_BIN_SIZE_MS = FRAME_WINDOW_MS


def build_heatmap(events, width, height, polarity=None):
    """Accumulate raw events into a dense 2D count map."""

    if polarity is None:
        mask = np.ones(len(events), dtype=bool)
    else:
        mask = events[:, 0] == polarity

    heatmap = np.zeros((height, width), dtype=np.float32)
    xs = np.clip(events[mask, 4].astype(np.int32), 0, width - 1)
    ys = np.clip(events[mask, 5].astype(np.int32), 0, height - 1)
    np.add.at(heatmap, (ys, xs), 1.0)
    return heatmap


def colorize_heatmap(heatmap):
    """Compress dynamic range and map the heatmap into RGB for display."""

    if np.max(heatmap) <= 0:
        return np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    scaled = np.log1p(heatmap)
    scaled = scaled / scaled.max()
    scaled = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def write_summary_images(events, output_dir, width, height):
    """Write overall, positive-polarity, and negative-polarity heatmaps."""

    all_heat = build_heatmap(events, width, height, polarity=None)
    on_heat = build_heatmap(events, width, height, polarity=1)
    off_heat = build_heatmap(events, width, height, polarity=0)

    cv2.imwrite(str(output_dir / "heatmap_all.png"), colorize_heatmap(all_heat))
    cv2.imwrite(str(output_dir / "heatmap_on.png"), colorize_heatmap(on_heat))
    cv2.imwrite(str(output_dir / "heatmap_off.png"), colorize_heatmap(off_heat))


def write_event_video(frames, output_dir, width, height, bin_size_ms=VIDEO_BIN_SIZE_MS):
    """Write one MP4 where each frame corresponds to one event window."""

    if not frames:
        return None

    fps = max(1, round(1000 / bin_size_ms))
    video_path = output_dir / "events.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer.")

    for frame in frames:
        image = render_event_frame(
            frame,
            annotate_lines=[
                frame.stream_name,
                "t={}..{}ms events={}".format(frame.start_ms, frame.end_ms, int(frame.signal)),
            ],
        )
        writer.write(image)

    writer.release()
    return video_path


def write_summary_json(events, timestamps, output_dir, source_path):
    """Write basic bounds and counts for one visualized stream."""

    summary = {
        "source": str(source_path),
        "event_count": int(len(events)),
        "polarity_on_count": int(np.sum(events[:, 0] == 1)),
        "polarity_off_count": int(np.sum(events[:, 0] == 0)),
        "x_min": int(events[:, 4].min()),
        "x_max": int(events[:, 4].max()),
        "y_min": int(events[:, 5].min()),
        "y_max": int(events[:, 5].max()),
        "time_min": int(timestamps.min()),
        "time_max": int(timestamps.max()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def visualize_file(input_path, output_dir):
    """Generate the full visualization bundle for one raw event stream."""

    ensure_dir(output_dir)

    events = load_events(input_path)
    ordering_signal = infer_timestamps_ms(events)
    duration_ms = infer_duration_ms(input_path, len(events))
    timestamps = normalize_timestamps_ms(ordering_signal, duration_ms)
    frames = split_event_windows(input_path, frame_window_ms=VIDEO_BIN_SIZE_MS)

    write_summary_images(events, output_dir, EVENT_WIDTH, EVENT_HEIGHT)
    video_path = write_event_video(frames, output_dir, EVENT_WIDTH, EVENT_HEIGHT)
    write_summary_json(events, timestamps, output_dir, input_path)

    print("Loaded {} events from {}".format(len(events), input_path))
    print("Wrote {}".format(output_dir / "heatmap_all.png"))
    print("Wrote {}".format(output_dir / "heatmap_on.png"))
    print("Wrote {}".format(output_dir / "heatmap_off.png"))
    if video_path is not None:
        print("Wrote {}".format(video_path))
    print("Wrote {}".format(output_dir / "summary.json"))


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INPUT)
    output_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUTPUT_DIR)
    ensure_dir(output_root)

    input_files = iter_input_files(input_path)
    if not input_files:
        print("No .bin files found in {}".format(input_path))
        sys.exit(1)

    for file_path in input_files:
        target_dir = output_root / file_path.stem
        visualize_file(file_path, target_dir)


if __name__ == "__main__":
    main()

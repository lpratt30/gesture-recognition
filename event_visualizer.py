import sys
import json
import re
from pathlib import Path

import cv2
import numpy as np
from project_paths import EVENT_VISUALIZATIONS_DIR, RAW_EVENT_STREAMS_DIR, ensure_dir


DEFAULT_INPUT = RAW_EVENT_STREAMS_DIR
DEFAULT_OUTPUT_DIR = EVENT_VISUALIZATIONS_DIR
EVENT_DTYPE = np.uint16
EVENT_WIDTH = 320
EVENT_HEIGHT = 320
ROW_WIDTH = 6
VIDEO_BIN_SIZE_MS = 75


def load_events(path):
    raw = np.fromfile(path, dtype=EVENT_DTYPE)
    if raw.size % ROW_WIDTH != 0:
        raise ValueError("Event file does not contain a whole number of 6-value rows.")
    events = raw.reshape(-1, ROW_WIDTH)
    return events


def infer_timestamps_ms(events):
    # Empirical layout from your file:
    # [polarity, coarse_tick, mid_tick, fine_tick, x, y]
    # This produces an ordering signal, but not a trustworthy wall-clock time.
    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


def infer_duration_ms(source_path, event_count):
    match = re.search(r"_(\d+)s\.bin$", source_path.name)
    if match:
        return int(match.group(1)) * 1000
    # Fallback: assume ~30 fps output bins and keep duration bounded.
    return max(1000, int(np.ceil(event_count / 100.0) * 33))


def normalize_timestamps_ms(ordering_signal, duration_ms):
    if len(ordering_signal) == 0:
        return ordering_signal

    order = np.argsort(ordering_signal, kind="stable")
    normalized = np.empty(len(ordering_signal), dtype=np.int64)
    if len(ordering_signal) == 1:
        normalized[order[0]] = 0
        return normalized

    normalized_values = np.linspace(0, duration_ms, len(ordering_signal), endpoint=False)
    normalized[order] = normalized_values.astype(np.int64)
    return normalized


def build_heatmap(events, width, height, polarity=None):
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
    if np.max(heatmap) <= 0:
        return np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    scaled = np.log1p(heatmap)
    scaled = scaled / scaled.max()
    scaled = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def write_summary_images(events, output_dir, width, height):
    all_heat = build_heatmap(events, width, height, polarity=None)
    on_heat = build_heatmap(events, width, height, polarity=1)
    off_heat = build_heatmap(events, width, height, polarity=0)

    cv2.imwrite(str(output_dir / "heatmap_all.png"), colorize_heatmap(all_heat))
    cv2.imwrite(str(output_dir / "heatmap_on.png"), colorize_heatmap(on_heat))
    cv2.imwrite(str(output_dir / "heatmap_off.png"), colorize_heatmap(off_heat))


def write_event_video(events, timestamps, output_dir, width, height, bin_size_ms=VIDEO_BIN_SIZE_MS):
    if len(events) == 0:
        return None

    t0 = int(timestamps.min())
    t1 = int(timestamps.max())
    if t1 <= t0:
        t1 = t0 + 1

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

    edges = np.arange(t0, t1 + bin_size_ms, bin_size_ms, dtype=np.int64)
    if len(edges) < 2:
        edges = np.array([t0, t0 + bin_size_ms], dtype=np.int64)

    for start, end in zip(edges[:-1], edges[1:]):
        mask = (timestamps >= start) & (timestamps < end)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if np.any(mask):
            batch = events[mask]
            xs = np.clip(batch[:, 4].astype(np.int32), 0, width - 1)
            ys = np.clip(batch[:, 5].astype(np.int32), 0, height - 1)
            pos = batch[:, 0] == 1
            neg = ~pos
            frame[ys[pos], xs[pos]] = (0, 255, 0)
            frame[ys[neg], xs[neg]] = (0, 0, 255)

        label = "t={}..{}  events={}".format(start - t0, end - t0, int(mask.sum()))
        cv2.putText(
            frame,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    return video_path


def write_summary_json(events, timestamps, output_dir, source_path):
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
    ensure_dir(output_dir)

    events = load_events(input_path)
    ordering_signal = infer_timestamps_ms(events)
    duration_ms = infer_duration_ms(input_path, len(events))
    timestamps = normalize_timestamps_ms(ordering_signal, duration_ms)

    write_summary_images(events, output_dir, EVENT_WIDTH, EVENT_HEIGHT)
    video_path = write_event_video(events, timestamps, output_dir, EVENT_WIDTH, EVENT_HEIGHT)
    write_summary_json(events, timestamps, output_dir, input_path)

    print("Loaded {} events from {}".format(len(events), input_path))
    print("Wrote {}".format(output_dir / "heatmap_all.png"))
    print("Wrote {}".format(output_dir / "heatmap_on.png"))
    print("Wrote {}".format(output_dir / "heatmap_off.png"))
    if video_path is not None:
        print("Wrote {}".format(video_path))
    print("Wrote {}".format(output_dir / "summary.json"))


def iter_input_files(input_path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".bin")
    raise FileNotFoundError("Input path not found: {}".format(input_path))


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

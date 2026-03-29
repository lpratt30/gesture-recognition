import json
from pathlib import Path

import cv2
import numpy as np

from event_visualizer import (
    EVENT_HEIGHT,
    EVENT_WIDTH,
    VIDEO_BIN_SIZE_MS,
    infer_duration_ms,
    infer_timestamps_ms,
    iter_input_files,
    load_events,
    normalize_timestamps_ms,
)
from project_paths import EVENT_VISUALIZATIONS_DIR, RAW_EVENT_STREAMS_DIR, ensure_dir


GRID_SIZE = 16
FRAME_WINDOW_MS = VIDEO_BIN_SIZE_MS
OUTPUT_DIR = EVENT_VISUALIZATIONS_DIR / "snr_sorted_frames"
OUTPUT_VIDEO = OUTPUT_DIR / "snr_sorted_frames.mp4"
OUTPUT_METADATA = OUTPUT_DIR / "snr_sorted_frames.json"


def estimate_frame_snr(events, width=EVENT_WIDTH, height=EVENT_HEIGHT, grid_size=GRID_SIZE):
    event_count = int(len(events))
    if event_count == 0:
        return {
            "signal": 0.0,
            "randomness": 1.0,
            "noise": 1.0,
            "snr": 0.0,
        }

    xs = np.clip(events[:, 4].astype(np.int32), 0, width - 1)
    ys = np.clip(events[:, 5].astype(np.int32), 0, height - 1)

    grid_x = np.minimum((xs * grid_size) // width, grid_size - 1)
    grid_y = np.minimum((ys * grid_size) // height, grid_size - 1)

    hist = np.zeros((grid_size, grid_size), dtype=np.float64)
    np.add.at(hist, (grid_y, grid_x), 1.0)

    probs = hist.ravel() / event_count
    probs = probs[probs > 0]

    if len(probs) <= 1:
        randomness = 0.0
    else:
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(grid_size * grid_size)
        randomness = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    occupied_cells = int(np.count_nonzero(hist))
    noise = 1.0 + (randomness * occupied_cells)
    snr = float(event_count / noise)

    return {
        "signal": float(event_count),
        "randomness": randomness,
        "noise": float(noise),
        "snr": snr,
    }


def frame_batches_for_file(file_path):
    events = load_events(file_path)
    ordering_signal = infer_timestamps_ms(events)
    duration_ms = infer_duration_ms(file_path, len(events))
    timestamps = normalize_timestamps_ms(ordering_signal, duration_ms)

    if len(events) == 0:
        return []

    t0 = int(timestamps.min())
    t1 = int(timestamps.max())
    if t1 <= t0:
        t1 = t0 + FRAME_WINDOW_MS

    edges = np.arange(t0, t1 + FRAME_WINDOW_MS, FRAME_WINDOW_MS, dtype=np.int64)
    if len(edges) < 2:
        edges = np.array([t0, t0 + FRAME_WINDOW_MS], dtype=np.int64)

    batches = []
    for frame_idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (timestamps >= start) & (timestamps < end)
        batch = events[mask]
        metrics = estimate_frame_snr(batch)
        if metrics["signal"] <= 0:
            continue

        batches.append(
            {
                "file_path": file_path,
                "file_name": file_path.name,
                "frame_idx": frame_idx,
                "start_ms": int(start - t0),
                "end_ms": int(end - t0),
                "events": batch,
                "metrics": metrics,
            }
        )

    return batches


def estimate_stream_snr(batches):
    if not batches:
        return {
            "signal": 0.0,
            "noise": 1.0,
            "randomness": 1.0,
            "snr": 0.0,
            "frame_count": 0,
        }

    total_signal = sum(batch["metrics"]["signal"] for batch in batches)
    total_noise = sum(batch["metrics"]["noise"] for batch in batches)
    weighted_randomness = 0.0
    if total_signal > 0:
        weighted_randomness = sum(
            batch["metrics"]["randomness"] * batch["metrics"]["signal"]
            for batch in batches
        ) / total_signal

    return {
        "signal": float(total_signal),
        "noise": float(total_noise),
        "randomness": float(weighted_randomness),
        "snr": float(total_signal / total_noise) if total_noise > 0 else 0.0,
        "frame_count": len(batches),
    }


def render_frame(batch):
    frame = np.zeros((EVENT_HEIGHT, EVENT_WIDTH, 3), dtype=np.uint8)
    events = batch["events"]

    xs = np.clip(events[:, 4].astype(np.int32), 0, EVENT_WIDTH - 1)
    ys = np.clip(events[:, 5].astype(np.int32), 0, EVENT_HEIGHT - 1)
    pos = events[:, 0] == 1
    neg = ~pos

    frame[ys[pos], xs[pos]] = (0, 255, 0)
    frame[ys[neg], xs[neg]] = (0, 0, 255)

    metrics = batch["metrics"]
    cv2.putText(
        frame,
        batch["file_name"],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "t={}..{}ms events={} snr={:.2f}".format(
            batch["start_ms"],
            batch["end_ms"],
            int(metrics["signal"]),
            metrics["snr"],
        ),
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "randomness={:.3f} noise={:.2f}".format(
            metrics["randomness"], metrics["noise"]
        ),
        (8, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def main():
    ensure_dir(OUTPUT_DIR)

    input_files = iter_input_files(RAW_EVENT_STREAMS_DIR)
    ranked_streams = []
    for file_path in input_files:
        batches = frame_batches_for_file(file_path)
        if not batches:
            continue
        ranked_streams.append(
            {
                "file_path": file_path,
                "file_name": file_path.name,
                "batches": batches,
                "metrics": estimate_stream_snr(batches),
            }
        )

    if not ranked_streams:
        print("No non-empty event frames found in {}".format(RAW_EVENT_STREAMS_DIR))
        return

    ranked_streams.sort(key=lambda stream: stream["metrics"]["snr"], reverse=True)

    fps = max(1, round(1000 / FRAME_WINDOW_MS))
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (EVENT_WIDTH, EVENT_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer: {}".format(OUTPUT_VIDEO))

    metadata = []
    total_frames = 0
    for rank, stream in enumerate(ranked_streams, start=1):
        for batch in stream["batches"]:
            writer.write(render_frame(batch))
            total_frames += 1

        metadata.append(
            {
                "rank": rank,
                "file_name": stream["file_name"],
                **stream["metrics"],
                "frames": [
                    {
                        "frame_idx": batch["frame_idx"],
                        "start_ms": batch["start_ms"],
                        "end_ms": batch["end_ms"],
                        **batch["metrics"],
                    }
                    for batch in stream["batches"]
                ],
            }
        )

    writer.release()

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Wrote {}".format(OUTPUT_VIDEO))
    print("Wrote {}".format(OUTPUT_METADATA))
    print(
        "Sorted {} streams by estimated SNR and stitched {} frames in stream order.".format(
            len(ranked_streams), total_frames
        )
    )


if __name__ == "__main__":
    main()

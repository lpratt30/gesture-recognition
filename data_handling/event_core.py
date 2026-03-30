"""Shared event-stream parsing, windowing, rendering, and scoring utilities."""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


EVENT_DTYPE = np.uint16
EVENT_WIDTH = 320
EVENT_HEIGHT = 320
ROW_WIDTH = 6
FRAME_WINDOW_MS = 75
SNR_GRID_SIZE = 16


@dataclass
class EventWindow:
    """A single fixed-duration slice of one raw event stream."""

    source_path: Path
    stream_name: str
    frame_index: int
    start_ms: int
    end_ms: int
    events: np.ndarray
    signal: float
    randomness: float
    noise: float
    snr: float


def load_events(path: Path) -> np.ndarray:
    """Load a raw `.bin` stream as an `N x 6` uint16 event matrix."""

    raw = np.fromfile(path, dtype=EVENT_DTYPE)
    if raw.size % ROW_WIDTH != 0:
        raise ValueError("Event file does not contain a whole number of 6-value rows: {}".format(path))
    return raw.reshape(-1, ROW_WIDTH)


def infer_timestamps_ms(events: np.ndarray) -> np.ndarray:
    """Build a sortable timestamp signal from the event time columns."""

    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


def infer_duration_ms(source_path: Path, event_count: int) -> int:
    """Infer stream duration from the filename, with a count-based fallback."""

    match = re.search(r"_(\d+)s\.bin$", source_path.name)
    if match:
        return int(match.group(1)) * 1000
    return max(1000, int(np.ceil(event_count / 100.0) * 33))


def normalize_timestamps_ms(ordering_signal: np.ndarray, duration_ms: int) -> np.ndarray:
    """Map event ordering onto a normalized timeline with the target duration."""

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


def iter_input_files(input_path: Path):
    """Yield `.bin` inputs from a file path or a directory."""

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".bin")
    raise FileNotFoundError("Input path not found: {}".format(input_path))


def estimate_frame_snr(events: np.ndarray, width: int = EVENT_WIDTH, height: int = EVENT_HEIGHT, grid_size: int = SNR_GRID_SIZE):
    """Estimate a simple signal-to-noise score from event density and spread."""

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


def split_event_windows(file_path: Path, frame_window_ms: int = FRAME_WINDOW_MS):
    """Split one raw stream into non-empty fixed-duration windows."""

    events = load_events(file_path)
    if len(events) == 0:
        return []

    ordering_signal = infer_timestamps_ms(events)
    duration_ms = infer_duration_ms(file_path, len(events))
    timestamps = normalize_timestamps_ms(ordering_signal, duration_ms)

    t0 = int(timestamps.min())
    t1 = int(timestamps.max())
    if t1 <= t0:
        t1 = t0 + frame_window_ms

    edges = np.arange(t0, t1 + frame_window_ms, frame_window_ms, dtype=np.int64)
    if len(edges) < 2:
        edges = np.array([t0, t0 + frame_window_ms], dtype=np.int64)

    windows = []
    for frame_idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (timestamps >= start) & (timestamps < end)
        batch = events[mask]
        metrics = estimate_frame_snr(batch)
        if metrics["signal"] <= 0:
            continue

        windows.append(
            EventWindow(
                source_path=file_path,
                stream_name=file_path.name,
                frame_index=frame_idx,
                start_ms=int(start - t0),
                end_ms=int(end - t0),
                events=batch,
                signal=metrics["signal"],
                randomness=metrics["randomness"],
                noise=metrics["noise"],
                snr=metrics["snr"],
            )
        )

    return windows


EventFrame = EventWindow
split_event_frames = split_event_windows


def estimate_stream_snr(frames):
    """Aggregate per-window SNR values into one stream-level score."""

    if not frames:
        return {
            "signal": 0.0,
            "noise": 1.0,
            "randomness": 1.0,
            "snr": 0.0,
            "frame_count": 0,
        }

    total_signal = sum(frame.signal for frame in frames)
    total_noise = sum(frame.noise for frame in frames)
    weighted_randomness = 0.0
    if total_signal > 0:
        weighted_randomness = sum(frame.randomness * frame.signal for frame in frames) / total_signal

    return {
        "signal": float(total_signal),
        "noise": float(total_noise),
        "randomness": float(weighted_randomness),
        "snr": float(total_signal / total_noise) if total_noise > 0 else 0.0,
        "frame_count": len(frames),
    }


def build_polarity_count_tensor(events: np.ndarray, width: int = EVENT_WIDTH, height: int = EVENT_HEIGHT) -> np.ndarray:
    """Convert events into a two-channel polarity count tensor."""

    xs = np.clip(events[:, 4].astype(np.int32), 0, width - 1)
    ys = np.clip(events[:, 5].astype(np.int32), 0, height - 1)
    pos = events[:, 0] == 1
    neg = ~pos

    tensor = np.zeros((2, height, width), dtype=np.float32)
    np.add.at(tensor[0], (ys[pos], xs[pos]), 1.0)
    np.add.at(tensor[1], (ys[neg], xs[neg]), 1.0)
    return tensor


def render_event_frame(frame: EventFrame, annotate_lines=None) -> np.ndarray:
    """Render one event window as a BGR image for previews or videos."""

    image = np.zeros((EVENT_HEIGHT, EVENT_WIDTH, 3), dtype=np.uint8)
    events = frame.events

    xs = np.clip(events[:, 4].astype(np.int32), 0, EVENT_WIDTH - 1)
    ys = np.clip(events[:, 5].astype(np.int32), 0, EVENT_HEIGHT - 1)
    pos = events[:, 0] == 1
    neg = ~pos

    image[ys[pos], xs[pos]] = (0, 255, 0)
    image[ys[neg], xs[neg]] = (0, 0, 255)

    if annotate_lines:
        for idx, text in enumerate(annotate_lines):
            cv2.putText(
                image,
                text,
                (8, 20 + (idx * 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return image

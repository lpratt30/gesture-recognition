from pathlib import Path

import cv2
import torch

from event_core import (
    FRAME_WINDOW_MS,
    build_polarity_count_tensor,
    iter_input_files,
    render_event_frame,
    split_event_windows,
)
from event_tensor_dataset import make_event_window_dataloader
from project_paths import EVENT_WINDOW_TENSORS_DIR, RAW_EVENT_STREAMS_DIR, ensure_dir


DEFAULT_LABEL = "positive"


def encode_float(value, decimals=3):
    return ("{0:0." + str(decimals) + "f}").format(value).replace(".", "p")


def build_tensor_filename(window, label):
    stream_stem = Path(window.stream_name).stem
    return (
        "label-{label}__stream-{stream}__window-{window_idx:04d}__win-{win:03d}ms__"
        "events-{events:05d}__snr-{snr}__rnd-{rnd}.pt"
    ).format(
        label=label,
        stream=stream_stem,
        window_idx=window.frame_index,
        win=FRAME_WINDOW_MS,
        events=int(window.signal),
        snr=encode_float(window.snr, decimals=2),
        rnd=encode_float(window.randomness, decimals=3),
    )


def iter_matching_assets(output_root, stream_stem, window_index, suffix):
    root = Path(output_root)
    for token in ("window", "frame"):
        pattern = "*__stream-{}__{}-{:04d}__*{}".format(stream_stem, token, window_index, suffix)
        for path in root.glob(pattern):
            yield path


def find_existing_label(output_root, stream_stem, window_index):
    for suffix in (".pt", ".png"):
        for path in iter_matching_assets(output_root, stream_stem, window_index, suffix):
            if path.name.startswith("label-positive__"):
                return "positive"
            if path.name.startswith("label-negative__"):
                return "negative"
    return None


def remove_existing_assets(output_root, stream_stem, window_index):
    for suffix in (".pt", ".png"):
        for path in iter_matching_assets(output_root, stream_stem, window_index, suffix):
            path.unlink()


def export_event_window_tensors(input_root=RAW_EVENT_STREAMS_DIR, output_root=EVENT_WINDOW_TENSORS_DIR, label=DEFAULT_LABEL):
    ensure_dir(output_root)

    written = 0
    for file_path in iter_input_files(Path(input_root)):
        for window in split_event_windows(file_path, frame_window_ms=FRAME_WINDOW_MS):
            stream_stem = Path(window.stream_name).stem
            existing_label = find_existing_label(output_root, stream_stem, window.frame_index)
            effective_label = existing_label or label
            remove_existing_assets(output_root, stream_stem, window.frame_index)

            tensor = torch.from_numpy(build_polarity_count_tensor(window.events))
            metadata = {
                "label": effective_label,
                "stream_name": window.stream_name,
                "window_index": window.frame_index,
                "window_start_ms": window.start_ms,
                "window_end_ms": window.end_ms,
                "event_count": int(window.signal),
                "frame_window_ms": FRAME_WINDOW_MS,
                "signal": window.signal,
                "randomness": window.randomness,
                "noise": window.noise,
                "snr": window.snr,
                "source_path": str(window.source_path),
            }
            metadata["frame_index"] = metadata["window_index"]
            metadata["start_ms"] = metadata["window_start_ms"]
            metadata["end_ms"] = metadata["window_end_ms"]

            output_path = Path(output_root) / build_tensor_filename(window, effective_label)
            preview_path = output_path.with_suffix(".png")
            torch.save(
                {
                    "tensor": tensor,
                    "label": effective_label,
                    "metadata": metadata,
                },
                output_path,
            )
            preview = render_event_frame(window)
            cv2.imwrite(str(preview_path), preview)
            written += 1

    return written


export_event_frame_tensors = export_event_window_tensors


def main():
    written = export_event_window_tensors()
    print("Wrote {} tensor windows to {}".format(written, EVENT_WINDOW_TENSORS_DIR))

    loader = make_event_window_dataloader(batch_size=8, shuffle=False)
    first_batch = next(iter(loader), None)
    if first_batch is not None:
        print(
            "Loaded sample batch: tensors={} targets={}".format(
                tuple(first_batch["tensors"].shape),
                tuple(first_batch["targets"].shape),
            )
        )


if __name__ == "__main__":
    main()

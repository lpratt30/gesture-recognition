import json

import cv2

from event_core import (
    EVENT_HEIGHT,
    EVENT_WIDTH,
    FRAME_WINDOW_MS,
    estimate_stream_snr,
    iter_input_files,
    render_event_frame,
    split_event_frames,
)
from project_paths import EVENT_VISUALIZATIONS_DIR, RAW_EVENT_STREAMS_DIR, ensure_dir


OUTPUT_DIR = EVENT_VISUALIZATIONS_DIR / "snr_sorted_frames"
OUTPUT_VIDEO = OUTPUT_DIR / "snr_sorted_frames.mp4"
OUTPUT_METADATA = OUTPUT_DIR / "snr_sorted_frames.json"


def main():
    ensure_dir(OUTPUT_DIR)

    input_files = iter_input_files(RAW_EVENT_STREAMS_DIR)
    ranked_streams = []
    for file_path in input_files:
        frames = split_event_frames(file_path, frame_window_ms=FRAME_WINDOW_MS)
        if not frames:
            continue
        ranked_streams.append(
            {
                "file_path": file_path,
                "file_name": file_path.name,
                "frames": frames,
                "metrics": estimate_stream_snr(frames),
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
        for frame in stream["frames"]:
            writer.write(
                render_event_frame(
                    frame,
                    annotate_lines=[
                        stream["file_name"],
                        "t={}..{}ms events={} snr={:.2f}".format(
                            frame.start_ms, frame.end_ms, int(frame.signal), frame.snr
                        ),
                        "randomness={:.3f} noise={:.2f}".format(frame.randomness, frame.noise),
                    ],
                )
            )
            total_frames += 1

        metadata.append(
            {
                "rank": rank,
                "file_name": stream["file_name"],
                **stream["metrics"],
                "frames": [
                    {
                        "frame_idx": frame.frame_index,
                        "start_ms": frame.start_ms,
                        "end_ms": frame.end_ms,
                        "signal": frame.signal,
                        "randomness": frame.randomness,
                        "noise": frame.noise,
                        "snr": frame.snr,
                    }
                    for frame in stream["frames"]
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

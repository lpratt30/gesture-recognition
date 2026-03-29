import os
import sys
import json
import struct
from project_paths import RAW_CAPTURE_DIR


FRAME_MAGIC = b"FRM1"
FRAME_HEADER_FMT = "<4sIIHHI"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)

DEFAULT_CHUNKS_DIR = os.fspath(RAW_CAPTURE_DIR / "chunks")
DEFAULT_OUTPUT_DIR = os.fspath(RAW_CAPTURE_DIR / "unpacked_frames")


def unpack_chunk_file(chunk_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "frames.ndjson")

    frame_count = 0
    with open(chunk_path, "rb") as src, open(metadata_path, "a", encoding="utf-8") as meta:
        while True:
            header = src.read(FRAME_HEADER_SIZE)
            if not header:
                break
            if len(header) != FRAME_HEADER_SIZE:
                raise ValueError("Truncated frame header in {}".format(chunk_path))

            magic, ts_ms, frame_id, width, height, jpeg_size = struct.unpack(
                FRAME_HEADER_FMT, header
            )
            if magic != FRAME_MAGIC:
                raise ValueError(
                    "Unexpected magic {} in {}".format(magic, chunk_path)
                )

            jpeg_bytes = src.read(jpeg_size)
            if len(jpeg_bytes) != jpeg_size:
                raise ValueError("Truncated JPEG payload in {}".format(chunk_path))

            frame_name = "frame_{:08d}.jpg".format(frame_id)
            frame_path = os.path.join(output_dir, frame_name)
            with open(frame_path, "wb") as frame_file:
                frame_file.write(jpeg_bytes)

            meta.write(
                json.dumps(
                    {
                        "chunk_file": chunk_path,
                        "frame_id": frame_id,
                        "ts_ms": ts_ms,
                        "width": width,
                        "height": height,
                        "jpeg_size": jpeg_size,
                        "file": frame_path,
                    }
                )
                + "\n"
            )
            frame_count += 1

    return frame_count


def main():
    chunks_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHUNKS_DIR
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    if not os.path.isdir(chunks_dir):
        print("Chunks directory not found: {}".format(chunks_dir))
        sys.exit(1)

    chunk_files = sorted(
        os.path.join(chunks_dir, name)
        for name in os.listdir(chunks_dir)
        if name.lower().endswith(".bin")
    )
    if not chunk_files:
        print("No chunk files found in {}".format(chunks_dir))
        sys.exit(1)

    total_frames = 0
    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        chunk_output_dir = os.path.join(output_dir, os.path.splitext(chunk_name)[0])
        frame_count = unpack_chunk_file(chunk_file, chunk_output_dir)
        total_frames += frame_count
        print("Unpacked {} frames from {}".format(frame_count, chunk_name))

    print("Done. Unpacked {} frames to {}".format(total_frames, output_dir))


if __name__ == "__main__":
    main()

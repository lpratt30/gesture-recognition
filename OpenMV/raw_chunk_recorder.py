"""Capture every frame and either stream JPEGs or transfer chunked recordings."""

import sensor
import image
import time
import os
import sys
import ustruct

FRAME_SIZE = sensor.B320X320
PIXFORMAT = sensor.GRAYSCALE
JPEG_QUALITY = 85
CHUNK_TARGET_BYTES = 512 * 1024
CHUNK_DIR = "raw_chunks"
CAPTURE_MODE = "chunk_store_and_transfer"
STATUS_EVERY_N_FRAMES = 25


def ensure_dir(path):
    try:
        if path not in os.listdir():
            os.mkdir(path)
    except Exception as exc:
        print("dir error: {}".format(exc))
        raise


def chunk_path(chunk_id):
    return "{}/chunk_{:05d}.bin".format(CHUNK_DIR, chunk_id)


def write_frame_record(handle, ts_ms, frame_id, width, height, jpeg_bytes):
    """Append one JPEG frame record to a chunk file."""

    header = ustruct.pack(
        "<4sIIHHI",
        b"FRM1",
        ts_ms & 0xFFFFFFFF,
        frame_id & 0xFFFFFFFF,
        width,
        height,
        len(jpeg_bytes),
    )
    handle.write(header)
    handle.write(jpeg_bytes)
    return len(header) + len(jpeg_bytes)


def transfer_chunk_file(path, chunk_id):
    """Send one chunk file to the host with a binary header."""

    payload_size = os.stat(path)[6]
    sys.stdout.write(ustruct.pack("<4sII", b"CHK1", chunk_id & 0xFFFFFFFF, payload_size))

    with open(path, "rb") as src:
        while True:
            block = src.read(4096)
            if not block:
                break
            sys.stdout.write(block)

    print("chunk_sent id={} bytes={}".format(chunk_id, payload_size))


def stream_single_frame(ts_ms, frame_id, width, height, jpeg_bytes):
    """Send one JPEG-compressed frame directly to the host."""

    sys.stdout.write(
        ustruct.pack(
            "<4sIIHHI",
            b"FRM1",
            ts_ms & 0xFFFFFFFF,
            frame_id & 0xFFFFFFFF,
            width,
            height,
            len(jpeg_bytes),
        )
    )
    sys.stdout.write(jpeg_bytes)


ensure_dir(CHUNK_DIR)

sensor.reset()
sensor.set_pixformat(PIXFORMAT)
sensor.set_framesize(FRAME_SIZE)
sensor.set_color_palette(image.PALETTE_EVT_DARK)
sensor.skip_frames(time=1500)

clock = time.clock()
frame_id = 0
chunk_id = 0
chunk_bytes = 0
frames_in_chunk = 0
chunk_handle = None


def open_new_chunk():
    global chunk_handle

    path = chunk_path(chunk_id)
    chunk_handle = open(path, "wb")
    return path


current_chunk_path = open_new_chunk()
print("raw_chunk_recorder ready mode={}".format(CAPTURE_MODE))

while True:
    clock.tick()
    img = sensor.snapshot()
    ts_ms = time.ticks_ms()
    frame_id += 1

    compressed = img.compress(quality=JPEG_QUALITY)
    jpeg_bytes = compressed.bytearray()
    width = img.width()
    height = img.height()

    if CAPTURE_MODE == "stream":
        stream_single_frame(ts_ms, frame_id, width, height, jpeg_bytes)
        if (frame_id % STATUS_EVERY_N_FRAMES) == 0:
            print("streamed_frames={} fps={:.2f}".format(frame_id, clock.fps()))
        continue

    chunk_bytes += write_frame_record(
        chunk_handle, ts_ms, frame_id, width, height, jpeg_bytes
    )
    frames_in_chunk += 1

    if (frame_id % STATUS_EVERY_N_FRAMES) == 0:
        print(
            "captured_frames={} chunk_id={} chunk_bytes={} fps={:.2f}".format(
                frame_id, chunk_id, chunk_bytes, clock.fps()
            )
        )

    if chunk_bytes >= CHUNK_TARGET_BYTES:
        chunk_handle.close()
        print(
            "chunk_ready id={} frames={} bytes={}".format(
                chunk_id, frames_in_chunk, chunk_bytes
            )
        )

        transfer_chunk_file(current_chunk_path, chunk_id)
        os.remove(current_chunk_path)

        chunk_id += 1
        chunk_bytes = 0
        frames_in_chunk = 0
        current_chunk_path = chunk_path(chunk_id)
        chunk_handle = open(current_chunk_path, "wb")

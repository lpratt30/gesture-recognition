import os
import sys
import json
import time
import struct
import serial
from project_paths import RAW_CAPTURE_DIR


SERIAL_PORT = "COM4"
BAUD_RATE = 115200
SER_TIMEOUT = 0.25

OUTPUT_ROOT = os.fspath(RAW_CAPTURE_DIR)
FRAMES_DIR = os.path.join(OUTPUT_ROOT, "frames")
CHUNKS_DIR = os.path.join(OUTPUT_ROOT, "chunks")
METADATA_PATH = os.path.join(OUTPUT_ROOT, "frames.ndjson")

SAVE_FRAMES = True
SAVE_CHUNKS = True
PRINT_CAMERA_LOGS = True

FRAME_MAGIC = b"FRM1"
CHUNK_MAGIC = b"CHK1"
FRAME_HEADER_SIZE = struct.calcsize("<4sIIHHI")
CHUNK_HEADER_SIZE = struct.calcsize("<4sII")


def ensure_dirs():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)


def read_exact_packet(buffer, total_size):
    if len(buffer) < total_size:
        return None
    packet = bytes(buffer[:total_size])
    del buffer[:total_size]
    return packet


def write_frame_packet(packet):
    _, ts_ms, frame_id, width, height, jpeg_size = struct.unpack(
        "<4sIIHHI", packet[:FRAME_HEADER_SIZE]
    )
    jpeg_bytes = packet[FRAME_HEADER_SIZE:]

    frame_name = "frame_{:08d}.jpg".format(frame_id)
    frame_path = os.path.join(FRAMES_DIR, frame_name)
    if SAVE_FRAMES:
        with open(frame_path, "wb") as frame_file:
            frame_file.write(jpeg_bytes)

    record = {
        "frame_id": frame_id,
        "ts_ms": ts_ms,
        "width": width,
        "height": height,
        "jpeg_size": jpeg_size,
        "file": frame_path if SAVE_FRAMES else None,
    }
    with open(METADATA_PATH, "a", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps(record) + "\n")

    print(
        "frame_saved id={} bytes={} size={}x{}".format(
            frame_id, jpeg_size, width, height
        )
    )


def write_chunk_packet(packet):
    _, chunk_id, payload_size = struct.unpack("<4sII", packet[:CHUNK_HEADER_SIZE])
    chunk_bytes = packet[CHUNK_HEADER_SIZE:]

    chunk_path = os.path.join(CHUNKS_DIR, "chunk_{:05d}.bin".format(chunk_id))
    if SAVE_CHUNKS:
        with open(chunk_path, "wb") as chunk_file:
            chunk_file.write(chunk_bytes)

    print("chunk_saved id={} bytes={}".format(chunk_id, payload_size))


def flush_text_prefix(buffer, next_magic_index):
    if next_magic_index <= 0:
        return

    prefix = bytes(buffer[:next_magic_index])
    last_newline = prefix.rfind(b"\n")
    if last_newline == -1:
        return

    text_block = bytes(buffer[: last_newline + 1])
    del buffer[: last_newline + 1]

    if PRINT_CAMERA_LOGS:
        for raw_line in text_block.splitlines():
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                print("[CAM] {}".format(line))


def flush_text_lines_without_magic(buffer):
    last_newline = buffer.rfind(b"\n")
    if last_newline == -1:
        return

    text_block = bytes(buffer[: last_newline + 1])
    del buffer[: last_newline + 1]

    if PRINT_CAMERA_LOGS:
        for raw_line in text_block.splitlines():
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                print("[CAM] {}".format(line))


def find_next_magic(buffer):
    frame_index = buffer.find(FRAME_MAGIC)
    chunk_index = buffer.find(CHUNK_MAGIC)

    candidates = [idx for idx in (frame_index, chunk_index) if idx != -1]
    if not candidates:
        return -1
    return min(candidates)


def parse_buffer(buffer):
    while True:
        magic_index = find_next_magic(buffer)
        if magic_index == -1:
            flush_text_lines_without_magic(buffer)
            if len(buffer) > 8192:
                del buffer[:-1024]
            return

        flush_text_prefix(buffer, magic_index)
        if magic_index > 0:
            if bytes(buffer[:magic_index]).strip():
                del buffer[:magic_index]
            continue

        if buffer.startswith(FRAME_MAGIC):
            if len(buffer) < FRAME_HEADER_SIZE:
                return
            _, ts_ms, frame_id, width, height, jpeg_size = struct.unpack(
                "<4sIIHHI", bytes(buffer[:FRAME_HEADER_SIZE])
            )
            total_size = FRAME_HEADER_SIZE + jpeg_size
            packet = read_exact_packet(buffer, total_size)
            if packet is None:
                return
            write_frame_packet(packet)
            continue

        if buffer.startswith(CHUNK_MAGIC):
            if len(buffer) < CHUNK_HEADER_SIZE:
                return
            _, chunk_id, payload_size = struct.unpack(
                "<4sII", bytes(buffer[:CHUNK_HEADER_SIZE])
            )
            total_size = CHUNK_HEADER_SIZE + payload_size
            packet = read_exact_packet(buffer, total_size)
            if packet is None:
                return
            write_chunk_packet(packet)
            continue


def main():
    ensure_dirs()
    print("Opening {} at {} baud".format(SERIAL_PORT, BAUD_RATE))

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SER_TIMEOUT)
    except serial.SerialException as exc:
        print("Serial open failed: {}".format(exc))
        sys.exit(1)

    buffer = bytearray()
    print("Listening for FRM1/CHK1 packets...")

    try:
        while True:
            incoming = ser.read(4096)
            if not incoming:
                continue
            buffer.extend(incoming)
            parse_buffer(buffer)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()

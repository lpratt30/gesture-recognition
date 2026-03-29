import os
import sys
import json
import struct
import serial
from project_paths import EVENT_CAPTURE_DIR


SERIAL_PORT = "COM4"
BAUD_RATE = 115200
SER_TIMEOUT = 0.25

OUTPUT_ROOT = os.fspath(EVENT_CAPTURE_DIR)
BATCH_DIR = os.path.join(OUTPUT_ROOT, "batches")
INDEX_PATH = os.path.join(OUTPUT_ROOT, "batches.ndjson")

PRINT_CAMERA_LOGS = True
EVENT_MAGIC = b"EVT1"
EVENT_HEADER_FMT = "<4sIIII"
EVENT_HEADER_SIZE = struct.calcsize(EVENT_HEADER_FMT)


def ensure_dirs():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(BATCH_DIR, exist_ok=True)


def read_exact_packet(buffer, total_size):
    if len(buffer) < total_size:
        return None
    packet = bytes(buffer[:total_size])
    del buffer[:total_size]
    return packet


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


def write_event_packet(packet):
    _, ts_ms, batch_id, event_count, payload_size = struct.unpack(
        EVENT_HEADER_FMT, packet[:EVENT_HEADER_SIZE]
    )
    payload = packet[EVENT_HEADER_SIZE:]
    batch_path = os.path.join(BATCH_DIR, "batch_{:08d}.bin".format(batch_id))

    with open(batch_path, "wb") as batch_file:
        batch_file.write(payload)

    with open(INDEX_PATH, "a", encoding="utf-8") as index_file:
        index_file.write(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "ts_ms": ts_ms,
                    "event_count": event_count,
                    "payload_size": payload_size,
                    "file": batch_path,
                }
            )
            + "\n"
        )

    print(
        "batch_saved id={} events={} bytes={}".format(
            batch_id, event_count, payload_size
        )
    )


def parse_buffer(buffer):
    while True:
        magic_index = buffer.find(EVENT_MAGIC)
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

        if len(buffer) < EVENT_HEADER_SIZE:
            return

        _, ts_ms, batch_id, event_count, payload_size = struct.unpack(
            EVENT_HEADER_FMT, bytes(buffer[:EVENT_HEADER_SIZE])
        )
        total_size = EVENT_HEADER_SIZE + payload_size
        packet = read_exact_packet(buffer, total_size)
        if packet is None:
            return
        write_event_packet(packet)


def main():
    ensure_dirs()
    print("Opening {} at {} baud".format(SERIAL_PORT, BAUD_RATE))
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SER_TIMEOUT)
    except serial.SerialException as exc:
        print("Serial open failed: {}".format(exc))
        sys.exit(1)

    print("Listening for EVT1 packets...")
    buffer = bytearray()
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

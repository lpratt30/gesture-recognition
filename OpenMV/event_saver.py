"""Record raw GENX320 event streams to SD card or on-board storage."""

import os
import time
import csi
from ulab import numpy as np

RUNID = "A"
REC_DURATION_S = 10
REPEAT_COUNT = 5
SENSOR_BUFFER = 65536
START_TIME = 1

BIASES = {"DIFF_OFF": 44, "DIFF_ON": 60, "HPF": 55, "FO": 19, "REFR": 0}


if "sdcard" in os.listdir("/"):
    STORAGE_DIR = "/sdcard"
else:
    STORAGE_DIR = "raw_events"
    if STORAGE_DIR not in os.listdir():
        os.mkdir(STORAGE_DIR)
    print("No SD card found. Falling back to on-board storage.")

base_count = sum(1 for f in os.listdir(STORAGE_DIR) if RUNID in f) + 1

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()

def alloc_events(n):
    """Allocate the largest event buffer the board can hold."""

    while True:
        try:
            return np.zeros((n, 6), dtype=np.uint16)
        except MemoryError:
            if n <= 1024:
                raise
            n //= 2
            print("MemoryError: reducing SENSOR_BUFFER to", n)

events = alloc_events(int(SENSOR_BUFFER))

csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_DIFF_OFF, BIASES["DIFF_OFF"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_DIFF_ON,  BIASES["DIFF_ON"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_HPF,      BIASES["HPF"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_FO,       BIASES["FO"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_REFR,     BIASES["REFR"])

time.sleep(START_TIME)

print("Starting event recording...")
print("Sensor buffer (events):", events.shape[0])
print("Repeat count:", REPEAT_COUNT)

for repeat_idx in range(REPEAT_COUNT):
    count = base_count + repeat_idx
    filename = "{}/{}_{}_{}_{}s.bin".format(
        STORAGE_DIR, SENSOR_BUFFER, RUNID, count, REC_DURATION_S
    )
    file = open(filename, "wb")

    event_count = 0
    start_ms = time.ticks_ms()
    deadline = start_ms + REC_DURATION_S * 1000

    print("Recording file {}/{} -> {}".format(repeat_idx + 1, REPEAT_COUNT, filename))

    try:
        while time.ticks_diff(time.ticks_ms(), deadline) < 0:
            n = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)
            if n > 0:
                file.write(memoryview(events)[: n * 6])
                event_count += n
    finally:
        file.flush()
        try:
            os.sync()
        except OSError as e:
            print("os.sync skipped:", e)
        file.close()

    elapsed_ms = time.ticks_diff(time.ticks_ms(), start_ms)
    eps = (event_count * 1000) // elapsed_ms if elapsed_ms > 0 else 0
    print("Done.")
    print(f"Events saved to {filename}")
    print(f"Duration: {elapsed_ms/1000:.3f}s | Total events: {event_count} | ~{eps} ev/s")

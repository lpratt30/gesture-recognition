"""Stream raw GENX320 event batches to the host over USB serial."""

import time
import sys
import csi
from ulab import numpy as np
import ustruct

SENSOR_BUFFER = 65536
START_TIME_S = 1
STATUS_EVERY_BATCHES = 20

BIASES = {
    "DIFF_OFF": 44,
    "DIFF_ON": 60,
    "HPF": 55,
    "FO": 19,
    "REFR": 0,
}


def alloc_events(n):
    """Allocate the largest batch buffer the board can hold."""

    while True:
        try:
            return np.zeros((n, 6), dtype=np.uint16)
        except MemoryError:
            if n <= 1024:
                raise
            n //= 2
            print("MemoryError: reducing SENSOR_BUFFER to", n)


events = alloc_events(int(SENSOR_BUFFER))

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_DIFF_OFF, BIASES["DIFF_OFF"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_DIFF_ON, BIASES["DIFF_ON"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_HPF, BIASES["HPF"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_FO, BIASES["FO"])
csi0.ioctl(csi.IOCTL_GENX320_SET_BIAS, csi.GENX320_BIAS_REFR, BIASES["REFR"])

print("event_streamer ready")
print("Sensor buffer (events):", events.shape[0])
time.sleep(START_TIME_S)

batch_id = 0
total_events = 0
start_ms = time.ticks_ms()

while True:
    n = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)
    if n <= 0:
        continue

    ts_ms = time.ticks_ms()
    payload = memoryview(events)[: n * 6]
    payload_size = n * 12

    header = ustruct.pack(
        "<4sIIII",
        b"EVT1",
        ts_ms & 0xFFFFFFFF,
        batch_id & 0xFFFFFFFF,
        n & 0xFFFFFFFF,
        payload_size & 0xFFFFFFFF,
    )
    sys.stdout.write(header)
    sys.stdout.write(payload)

    batch_id += 1
    total_events += n

    if (batch_id % STATUS_EVERY_BATCHES) == 0:
        elapsed_ms = time.ticks_diff(time.ticks_ms(), start_ms)
        eps = (total_events * 1000) // elapsed_ms if elapsed_ms > 0 else 0
        print("batches={} total_events={} eps={}".format(batch_id, total_events, eps))

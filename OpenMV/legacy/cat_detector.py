"""Simple blob-detection sandbox for quick threshold experiments."""

import sensor
import image
import time

THRESHOLD = [(10, 20, -10, 10, -20, 0)]


sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.set_color_palette(image.PALETTE_EVT_DARK)

clock = time.clock()

while True:
    clock.tick()

    img = sensor.snapshot()

    blobs = img.find_blobs(
        THRESHOLD,
        invert=True,
        pixels_threshold=30,
        area_threshold=1000,
        merge=True,
    )

    for blob in blobs:
        img.draw_rectangle(blob.rect(), color=(255, 0, 0))
        img.draw_cross(blob.cx(), blob.cy(), color=(0, 255, 0))

    print(clock.fps())

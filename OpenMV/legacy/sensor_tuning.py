"""Live blob-tuning script plus JSON archiving for the current settings."""

import image
import json
import os
import sensor
import time as utime


SETTINGS_DIR = "settings_archive"

PIX_FORMAT = sensor.GRAYSCALE
FRAME_SIZE = sensor.B320X320
COLOR_PALETTE = image.PALETTE_EVT_DARK
BLOB_THRESHOLD = [(20, 47)]

PIXELS_THRESHOLD = 15
AREA_THRESHOLD = 1200
LOWERBOUND_ASPECT_RATIO = 0.4
UPPERBOUND_ASPECT_RATIO = 1.1
MARGIN = 100
MERGE = True
INVERT = False

DISPLAY_FPS = False
FPS_PRINT_INTERVAL = 500


if SETTINGS_DIR not in os.listdir():
    os.mkdir(SETTINGS_DIR)


sensor.reset()
sensor.set_pixformat(PIX_FORMAT)
sensor.set_framesize(FRAME_SIZE)
sensor.set_color_palette(COLOR_PALETTE)


def save_settings():
    """Archive the current tuning parameters as JSON."""

    try:
        dt = utime.localtime()
        timestamp = "%04d%02d%02d_%02d%02d%02d" % (
            dt[0],
            dt[1],
            dt[2],
            dt[3],
            dt[4],
            dt[5],
        )
        filename = "{}/settings_{}.json".format(SETTINGS_DIR, timestamp)

        data = {
            "timestamp": timestamp,
            "sensor": {
                "pixformat": "GRAYSCALE",
                "framesize": "B320X320",
                "color_palette": "PALETTE_EVT_DARK",
            },
            "blob_detection": {
                "threshold": BLOB_THRESHOLD,
                "invert": INVERT,
                "pixels_threshold": PIXELS_THRESHOLD,
                "area_threshold": AREA_THRESHOLD,
                "merge": MERGE,
                "margin": MARGIN,
                "lowerbound_aspect_ratio": LOWERBOUND_ASPECT_RATIO,
                "upperbound_aspect_ratio": UPPERBOUND_ASPECT_RATIO,
            },
        }

        with open(filename, "w") as handle:
            json.dump(data, handle)

        print("Saved settings to {}".format(filename))
        return filename
    except Exception as exc:
        print("Error saving settings: {}".format(exc))
        return None


save_settings()

clock = utime.clock()
fps_skipped = 0

while True:
    clock.tick()
    img = sensor.snapshot()

    if DISPLAY_FPS:
        if fps_skipped < FPS_PRINT_INTERVAL:
            fps_skipped += 1
        else:
            print("FPS: {:.2f}".format(clock.fps()))
            fps_skipped = 0

    blobs = img.find_blobs(
        BLOB_THRESHOLD,
        invert=INVERT,
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=MERGE,
        margin=MARGIN,
    )

    for blob in blobs:
        aspect_ratio = blob.w() / blob.h()

        if LOWERBOUND_ASPECT_RATIO <= aspect_ratio <= UPPERBOUND_ASPECT_RATIO:
            img.draw_rectangle(blob.rect(), color=(0, 255, 0))
        else:
            img.draw_rectangle(blob.rect(), color=(255, 0, 0))

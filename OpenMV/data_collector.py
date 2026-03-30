"""Capture filtered blob crops and their feature vectors for dataset building."""

import sensor
import image
import time
import os
import sys
import json
from machine import LED

led_blue = LED("LED_BLUE")
led_blue.off()

led_red = LED("LED_RED")
led_red.off()

BLOB_THRESHOLD = [(20, 47)]
PIXELS_THRESHOLD = 15
AREA_THRESHOLD = 1200
LOWERBOUND_ASPECT_RATIO = 0.4
UPPERBOUND_ASPECT_RATIO = 1.1
MARGIN = 100

BLOB_COUNTER_START = 1500

MERGE = True
INVERT = False

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.set_color_palette(image.PALETTE_EVT_DARK)

# Create a directory for blobs if it doesn't exist
DIRECTORY_NAME = "raw_data"
try:
    if DIRECTORY_NAME not in os.listdir():
        os.mkdir(DIRECTORY_NAME)
except Exception as exc:
    print("Error creating directory: {}".format(exc))

existing_files = os.listdir(DIRECTORY_NAME)

if len(existing_files) > 50:
    print("CRITICAL: {} files found (Limit 50). Stopping.".format(len(existing_files)))

    while True:
        led_red.on()
        time.sleep(0.1)
        led_red.off()
        time.sleep(0.1)
else:
    print("Found {} files. Cleaning directory...".format(len(existing_files)))

    for f in existing_files:
        try:
            full_path = "{}/{}".format(DIRECTORY_NAME, f)
            os.remove(full_path)
        except OSError as exc:
            print("Error deleting {}: {}".format(f, exc))

    print("Directory Cleaned. Starting main loop...")

def get_features_list(blob, roi_img):
    """Extract the hand-crafted feature vector used by the desktop models."""

    x, y, w, h = blob.rect()
    aspect_ratio_hw = h / float(w)
    extent = blob.extent()
    solidity = blob.solidity()
    circularity = blob.roundness()
    angle = blob.rotation()

    w = roi_img.width()
    h = roi_img.height()

    w_third = w // 3
    h_third = h // 3
    h_half = h // 2

    roi_top_mid = (w_third, 0, w_third, h_third)
    roi_top_left = (0, 0, w_third, h_third)
    roi_top_right = (2 * w_third, 0, w_third, h_third)
    roi_vert_mid = (w_third, 0, w_third, h)
    roi_top_half = (0, 0, w, h_half)
    roi_bot_half = (0, h_half, w, h - h_half)

    # Helper
    def get_norm_density(roi):
        if roi[2] <= 0 or roi[3] <= 0:
            return 0.0
        stats = roi_img.get_statistics(roi=roi)
        return stats.mean() / 255.0

    d_top_mid = get_norm_density(roi_top_mid)
    d_top_left = get_norm_density(roi_top_left)
    d_top_right = get_norm_density(roi_top_right)
    d_top_sides = (d_top_left + d_top_right) / 2.0
    den_vert_mid = get_norm_density(roi_vert_mid)

    den_vert_sides = (
        get_norm_density((0, 0, w_third, h))
        + get_norm_density((2 * w_third, 0, w_third, h))
    ) / 2.0
    v_contrast = den_vert_sides - den_vert_mid
    d_top_half = get_norm_density(roi_top_half)
    d_bot_half = get_norm_density(roi_bot_half)
    palm_contrast = d_bot_half - d_top_half

    return [
        aspect_ratio_hw, extent, solidity, circularity, angle,
        d_top_mid, d_top_left, d_top_right, v_contrast, palm_contrast
    ]
clock = time.clock()
blob_counter = 0
last_save_time = 0
maximum_blobs = 400
SAVE_INTERVAL = 200

print("Starting. Will save {} filtered blobs to /{}".format(maximum_blobs, DIRECTORY_NAME))

while True:
    clock.tick()
    img = sensor.snapshot()

    blobs = img.find_blobs(
        BLOB_THRESHOLD,
        invert=INVERT,
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=MERGE,
        margin=MARGIN,
    )

    current_time = time.ticks_ms()
    did_save_in_this_frame = False

    for blob in blobs:
        aspect_ratio = blob.w() / blob.h()

        if LOWERBOUND_ASPECT_RATIO <= aspect_ratio <= UPPERBOUND_ASPECT_RATIO:
            led_blue.on()

            if time.ticks_diff(current_time, last_save_time) > SAVE_INTERVAL:
                cropped_img = img.copy(roi=blob.rect())
                cropped_img.binary(BLOB_THRESHOLD)
                features = get_features_list(blob, cropped_img)
                base_filename = "{}/blob_{}".format(DIRECTORY_NAME, blob_counter + BLOB_COUNTER_START)
                cropped_img.save(base_filename + ".jpg", quality=90)
                json_data = {
                    "id": blob_counter,
                    "features": features,
                    "timestamp": current_time
                }

                with open(base_filename + ".json", "w") as f:
                    f.write(json.dumps(json_data))

                print("Saved: {} (Ratio: {:.2f})".format(base_filename, aspect_ratio))

                blob_counter += 1
                if blob_counter >= maximum_blobs:
                    print("Finished saving {} blobs.".format(maximum_blobs))
                    sys.exit()

                did_save_in_this_frame = True

            img.draw_rectangle(blob.rect(), color=(0, 255, 0))
            led_blue.off()

        else:
            img.draw_rectangle(blob.rect(), color=(255, 0, 0))

    if did_save_in_this_frame:
        last_save_time = current_time

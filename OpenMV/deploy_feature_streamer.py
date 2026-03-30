"""Detect candidate hand blobs and stream feature vectors as JSON."""

import sensor
import image
import json
import time

BLOB_THRESHOLD = [(20, 47)]
PIXELS_THRESHOLD = 15
AREA_THRESHOLD = 1200
MERGE = True
INVERT = False
LOWERBOUND_RATIO = 0.4
UPPERBOUND_RATIO = 1.1

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.set_color_palette(image.PALETTE_EVT_DARK)
sensor.skip_frames(time=1000)

def get_features_list(blob, roi_img):
    """Extract the same feature vector used during training."""

    aspect_ratio_hw = blob.h() / float(blob.w())
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

    def get_norm_density(roi):
        if roi[2] <= 0 or roi[3] <= 0:
            return 0.0
        stats = roi_img.get_statistics(roi=roi)
        return stats.mean() / 255.0

    d_top_mid = get_norm_density(roi_top_mid)
    d_top_left = get_norm_density(roi_top_left)
    d_top_right = get_norm_density(roi_top_right)

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

while True:
    img = sensor.snapshot()

    blobs = img.find_blobs(BLOB_THRESHOLD, invert=INVERT, pixels_threshold=PIXELS_THRESHOLD, area_threshold=AREA_THRESHOLD, merge=MERGE)

    for blob in blobs:
        ar = blob.w() / blob.h()

        if LOWERBOUND_RATIO <= ar <= UPPERBOUND_RATIO:
            cropped_img = img.copy(roi=blob.rect())
            cropped_img.binary(BLOB_THRESHOLD)
            feats = get_features_list(blob, cropped_img)
            print(json.dumps({"d": feats}))

            img.draw_rectangle(blob.rect(), color=(0, 255, 0))
            img.draw_cross(blob.cx(), blob.cy(), color=(0, 255, 0))

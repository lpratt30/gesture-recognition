"""Extract hand-crafted features on-device and stream them as JSON."""

import sensor
import image
import json

BLOB_THRESHOLD = [(17, 47, -39, 39, -30, 15)]
PIXELS_THRESHOLD = 100
AREA_THRESHOLD = 1000
MERGE = True
INVERT = False
LOWERBOUND_RATIO = 0.4
UPPERBOUND_RATIO = 1.1

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.set_color_palette(image.PALETTE_EVT_DARK)
sensor.skip_frames(time=1000)

def get_features_list(img, blob):
    """Extract the feature vector expected by the deployed model."""

    x, y, w, h = blob.rect()

    aspect_ratio_hw = h / w if w > 0 else 0
    extent = blob.extent()
    solidity = blob.solidity()
    circularity = blob.roundness()

    x_third = w // 3
    y_third = h // 3

    roi_top_mid = (x + x_third, y, x_third, y_third)
    roi_top_left = (x, y, x_third, y_third)
    roi_top_right = (x + 2*x_third, y, x_third, y_third)

    roi_vert_mid = (x + x_third, y, x_third, h)
    roi_vert_left = (x, y, x_third, h)
    roi_vert_right = (x + 2*x_third, y, x_third, h)

    def get_density(roi):
        stats = img.get_statistics(roi=roi, thresholds=BLOB_THRESHOLD)
        return stats.mean() / 255.0

    d_top_mid = get_density(roi_top_mid)
    d_top_sides = (get_density(roi_top_left) + get_density(roi_top_right)) / 2.0

    den_vert_mid = get_density(roi_vert_mid)
    den_vert_sides = (get_density(roi_vert_left) + get_density(roi_vert_right)) / 2.0
    gap_signal = den_vert_sides / (den_vert_mid + 0.00001)

    return [aspect_ratio_hw, extent, solidity, circularity, d_top_mid, d_top_sides, gap_signal]

while True:
    img = sensor.snapshot()

    blobs = img.find_blobs(
        BLOB_THRESHOLD,
        invert=INVERT,
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=MERGE
    )

    for blob in blobs:
        ar = blob.w() / blob.h()

        if LOWERBOUND_RATIO <= ar <= UPPERBOUND_RATIO:
            feats = get_features_list(img, blob)
            packet = {"d": feats}
            print(json.dumps(packet))
            img.draw_rectangle(blob.rect(), color=(0, 255, 0))
        else:
            img.draw_rectangle(blob.rect(), color=(255, 0, 0))

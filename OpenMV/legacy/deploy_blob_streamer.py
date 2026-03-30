"""Detect candidate blobs and stream cropped JPEGs over USB serial."""

import image
import sensor
import sys
import ustruct

BLOB_THRESHOLD = [(17, 47, -39, 39, -30, 15)]
PIXELS_THRESHOLD = 100
AREA_THRESHOLD = 1000
MERGE = True
INVERT = False

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.set_color_palette(image.PALETTE_EVT_DARK)
sensor.skip_frames(time=2000)

while(True):
    img = sensor.snapshot()

    blobs = img.find_blobs(
        BLOB_THRESHOLD,
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=MERGE,
        invert=INVERT
    )

    for blob in blobs:
        ar = blob.w() / blob.h()
        if 0.4 <= ar <= 1.1:
            blob_img = img.copy(roi=blob.rect())
            cimg = blob_img.compress(quality=100)
            sys.stdout.write(ustruct.pack("<3sI", b"IMG", cimg.size()))
            sys.stdout.write(cimg)
            img.draw_rectangle(blob.rect(), color=127)

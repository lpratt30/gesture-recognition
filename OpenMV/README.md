# OpenMV Scripts

This folder mirrors the camera-side scripts from `C:\Users\pratt\OneDrive\Documents\OpenMV` so the OpenMV code lives in the main repo alongside the desktop tooling.

## Event Camera

- `event_saver.py`: record repeated raw event streams to SD card or on-board storage
- `event_streamer.py`: stream raw event batches over USB serial
- `sensor_tuning.py`: tune blob thresholds and archive the current settings as JSON

## Frame / Blob Capture

- `raw_chunk_recorder.py`: capture every frame and either stream JPEGs or transfer chunked recordings
- `data_collector.py`: save filtered blob crops and feature JSON sidecars
- `deploy_blob_streamer.py`: stream cropped blob JPEGs to the desktop
- `deploy_feature_streamer.py`: stream hand-crafted features as JSON
- `deploy_model.py`: extract the deployed feature vector on-device and stream it

## Other

- `cat_detector.py`: small thresholding sandbox
- `peace_model.py`: generated model code used by the OpenMV-side classifier

# OpenMV Scripts

This folder mirrors the camera-side scripts from `C:\Users\pratt\OneDrive\Documents\OpenMV` so the OpenMV code lives in the main repo alongside the desktop tooling.

## Event Camera

- `event_saver.py`: record repeated raw event streams to SD card or on-board storage
- `event_streamer.py`: stream raw event batches over USB serial

- `raw_chunk_recorder.py`: capture every frame and either stream JPEGs or transfer chunked recordings

Older blob/feature-model scripts now live under `OpenMV/legacy/`.

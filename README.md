# Gesture Recognition

Status: `v0.0.1-alpha`

Early event-vision gesture recognition experiments, data tooling, and visualization scripts.

## Overview

The implementation is split into two packages:

- `data_handling/`: ingestion, transfers, visualization, tensor export, labeling, and dataset utilities
- `modeling/`: training and experiment scripts

The current workflow is organized around raw event streams:

- raw recordings live in `gesture_data/raw_event_streams`
- visualizations live in `gesture_data/event_visualizations`
- fixed-duration event-window tensors live in `gesture_data/event_frame_tensors`

The shared window size is defined once in `data_handling/event_core.py` as `FRAME_WINDOW_MS`. Visualization, tensor export, and SNR ranking all use that same value.

## Highest-SNR Event Stream

The GIF below is generated from the current highest-ranked stream by the SNR sorter (`65536_A_19_10s.bin` at the time of generation).

It is meant as a quick preview of the event collection and ranking pipeline.

![Highest-SNR event stream](assets/readme/highest_snr_stream.gif)

## Relevant Scripts

- `python -m data_handling.transfer_raw_event_streams`: move new raw event streams into `gesture_data/raw_event_streams` and visualize them
- `python -m data_handling.event_visualizer`: build per-stream heatmaps, summaries, and MP4 previews
- `python -m data_handling.event_tensor_pipeline`: export fixed-duration event windows as `.pt` tensors plus preview images
- `python -m data_handling.event_frame_labeler`: review saved window previews and label them interactively
- `python -m data_handling.snr_sort_event_frames`: estimate per-stream SNR and create an SNR-ranked stitched MP4
- `modeling/baseline_randomforest.py` and `modeling/xg_randomforest.py`: train the tabular models on processed features

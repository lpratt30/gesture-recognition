# Gesture Recognition

Status: `v0.0.1-alpha`

Early event-vision gesture recognition experiments, data tooling, and visualization scripts.

## Overview

The current workflow is organized around raw event streams:

- raw recordings live in `gesture_data/raw_event_streams`
- visualizations live in `gesture_data/event_visualizations`
- fixed-duration event-window tensors live in `gesture_data/event_frame_tensors`

The shared window size is defined once in `event_core.py` as `FRAME_WINDOW_MS`. Visualization, tensor export, and SNR ranking all use that same value.

## Highest-SNR Event Stream

The GIF below is generated from the current highest-ranked stream by the SNR sorter (`65536_A_19_10s.bin` at the time of generation).

It is meant as a quick preview of the event collection and ranking pipeline.

![Highest-SNR event stream](assets/readme/highest_snr_stream.gif)

## Relevant Scripts

- `transfer_raw_event_streams.py`: move new raw event streams into `gesture_data/raw_event_streams` and visualize them
- `event_visualizer.py`: build per-stream heatmaps, summaries, and MP4 previews
- `event_tensor_pipeline.py`: export fixed-duration event windows as `.pt` tensors plus preview images
- `event_tensor_dataset.py`: PyTorch dataset and dataloader helpers for saved window tensors
- `event_frame_labeler.py`: review saved window previews and label them interactively
- `snr_sort_event_frames.py`: estimate per-stream SNR and create an SNR-ranked stitched MP4

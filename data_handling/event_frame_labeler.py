"""Interactive labeling UI for saved event-window previews and tensors."""

from pathlib import Path

import cv2
import torch

from data_handling.project_paths import EVENT_WINDOW_TENSORS_DIR


WINDOW_NAME = "Event Window Labeler"
POSITIVE_KEY = ord("w")
NEGATIVE_KEY = ord("s")
PREV_KEY = ord("a")
NEXT_KEY = ord("d")
QUIT_KEY = ord("q")


def tensor_to_bgr_image(tensor):
    """Turn a two-channel polarity tensor into a viewable color image."""

    tensor = tensor.detach().cpu().numpy()
    pos = tensor[0]
    neg = tensor[1]

    if pos.max() > 0:
        pos = pos / pos.max()
    if neg.max() > 0:
        neg = neg / neg.max()

    image = cv2.merge(
        [
            (neg * 255).astype("uint8"),
            (pos * 255).astype("uint8"),
            ((pos + neg) * 48).clip(0, 255).astype("uint8"),
        ]
    )
    return image


def swap_label_prefix(name, new_label):
    """Rewrite the label token at the start of an exported sample name."""

    current_name = name
    if current_name.startswith("label-positive__"):
        return current_name.replace("label-positive__", "label-{}__".format(new_label), 1)
    if current_name.startswith("label-negative__"):
        return current_name.replace("label-negative__", "label-{}__".format(new_label), 1)
    raise ValueError("Unexpected event-frame filename: {}".format(name))


def load_sample_payload(sample):
    """Load one saved tensor payload when it exists."""

    pt_path = sample.get("pt")
    if pt_path is None:
        return None
    return torch.load(pt_path, map_location="cpu")


def get_sample_sort_key(sample):
    """Review dense windows first, then break ties by SNR."""

    payload = load_sample_payload(sample)
    if payload is None:
        return (0, 0.0, sample["stem"])

    metadata = payload["metadata"]
    event_count = int(metadata.get("event_count", metadata.get("signal", 0)))
    snr = float(metadata.get("snr", 0.0))
    return (-event_count, -snr, sample["stem"])


def collect_samples():
    """Collect paired `.pt` and `.png` assets for each labeled sample."""

    samples = {}
    for path in EVENT_WINDOW_TENSORS_DIR.iterdir():
        if not path.is_file() or path.suffix not in {".pt", ".png"}:
            continue
        stem = path.stem
        if stem not in samples:
            samples[stem] = {"stem": stem, "pt": None, "png": None}
        samples[stem][path.suffix[1:]] = path
    ordered_samples = [samples[key] for key in sorted(samples.keys())]
    ordered_samples.sort(key=get_sample_sort_key)
    return ordered_samples


def relabel_sample(sample, new_label):
    """Rename both assets and update the saved payload label in place."""

    updated = dict(sample)

    pt_path = sample.get("pt")
    if pt_path is not None:
        payload = torch.load(pt_path, map_location="cpu")
        payload["label"] = new_label
        payload["metadata"]["label"] = new_label
        new_pt_path = pt_path.with_name(swap_label_prefix(pt_path.name, new_label))
        torch.save(payload, new_pt_path)
        if new_pt_path != pt_path:
            pt_path.unlink()
        updated["pt"] = new_pt_path

    png_path = sample.get("png")
    if png_path is not None:
        new_png_path = png_path.with_name(swap_label_prefix(png_path.name, new_label))
        if new_png_path != png_path:
            png_path.rename(new_png_path)
        updated["png"] = new_png_path

    updated["stem"] = (updated.get("pt") or updated.get("png")).stem
    return updated


def load_preview_image(sample):
    """Load a preview image, falling back to rendering from the tensor."""

    if sample.get("png") is not None:
        image = cv2.imread(str(sample["png"]))
        if image is not None:
            return image

    if sample.get("pt") is not None:
        payload = torch.load(sample["pt"], map_location="cpu")
        return tensor_to_bgr_image(payload["tensor"])

    raise RuntimeError("Sample has neither .png nor .pt: {}".format(sample["stem"]))


def main():
    samples = collect_samples()
    if not samples:
        print("No tensor windows found in {}".format(EVENT_WINDOW_TENSORS_DIR))
        return

    print("Controls: w=positive, s=negative, a=previous, d=next, q=quit")
    index = 0

    while True:
        sample = samples[index]
        payload = load_sample_payload(sample)
        metadata = payload["metadata"]
        label = payload.get("label", metadata.get("label", "unknown"))

        image = load_preview_image(sample)
        overlay = image.copy()
        cv2.putText(overlay, "{} / {}".format(index + 1, len(samples)), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, sample["stem"], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "label={} events={} snr={:.2f} window={}..{}ms".format(
                label,
                int(metadata["signal"]),
                metadata["snr"],
                metadata.get("window_start_ms", metadata.get("start_ms", 0)),
                metadata.get("window_end_ms", metadata.get("end_ms", 0)),
            ),
            (8, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, overlay)

        key = cv2.waitKey(0) & 0xFF
        if key == QUIT_KEY:
            break
        if key == PREV_KEY:
            index = max(0, index - 1)
            continue
        if key == NEXT_KEY:
            index = min(len(samples) - 1, index + 1)
            continue
        if key == POSITIVE_KEY:
            samples[index] = relabel_sample(sample, "positive")
            index = min(len(samples) - 1, index + 1)
            continue
        if key == NEGATIVE_KEY:
            samples[index] = relabel_sample(sample, "negative")
            index = min(len(samples) - 1, index + 1)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

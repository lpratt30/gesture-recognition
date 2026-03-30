import re
import shutil
from pathlib import Path
from data_handling.project_paths import RAW_EVENT_STREAMS_DIR
from data_handling.event_visualizer import visualize_file
from data_handling.project_paths import EVENT_VISUALIZATIONS_DIR


DEST_DIR = RAW_EVENT_STREAMS_DIR
SOURCE_DIR = Path(r"E:\raw_event_streams")
FILENAME_RE = re.compile(r"^(?P<prefix>\d+_[A-Za-z]+)_(?P<index>\d+)_(?P<suffix>\d+s\.bin)$")


def parse_filename(path: Path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return {
        "prefix": match.group("prefix"),
        "index": int(match.group("index")),
        "suffix": match.group("suffix"),
    }


def find_max_existing_index(directory: Path) -> int:
    max_index = 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        parsed = parse_filename(path)
        if parsed:
            max_index = max(max_index, parsed["index"])
    return max_index


def collect_source_files(directory: Path):
    files = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        parsed = parse_filename(path)
        if parsed:
            files.append((path, parsed))
    files.sort(key=lambda item: item[1]["index"])
    return files


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError("Source directory not found: {}".format(SOURCE_DIR))

    max_existing = find_max_existing_index(DEST_DIR)
    source_files = collect_source_files(SOURCE_DIR)

    if not source_files:
        print("No matching raw event stream files found in {}".format(SOURCE_DIR))
        return

    next_index = max_existing + 1
    moved_count = 0
    visualized_count = 0

    print("Max existing raw stream index: {}".format(max_existing))

    for source_path, parsed in source_files:
        target_name = "{}_{}_{}".format(parsed["prefix"], next_index, parsed["suffix"])
        target_path = DEST_DIR / target_name

        while target_path.exists():
            next_index += 1
            target_name = "{}_{}_{}".format(parsed["prefix"], next_index, parsed["suffix"])
            target_path = DEST_DIR / target_name

        shutil.move(str(source_path), str(target_path))
        print("Moved {} -> {}".format(source_path.name, target_name))
        visualize_file(target_path, EVENT_VISUALIZATIONS_DIR / target_path.stem)
        visualized_count += 1

        next_index += 1
        moved_count += 1

    print("Transferred {} file(s) to {}".format(moved_count, DEST_DIR))
    print("Visualized {} file(s) in {}".format(visualized_count, EVENT_VISUALIZATIONS_DIR))


if __name__ == "__main__":
    main()

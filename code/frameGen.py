import os
import cv2
from pathlib import Path

os.umask(0o002)

def extract_frames(root_dir):
    root = Path(root_dir)

    # Prompt user to select first-level directories
    lvl1_dirs = [d for d in root.iterdir() if d.is_dir()]
    print("Available first-level directories:")
    for i, d in enumerate(lvl1_dirs):
        print(f"  {i+1}. {d.name}")
    print("  0. All labelled directories")

    choice = input("Select a directory to process (number), or 0 for all: ").strip()
    selected = []

    if choice == "0":
        selected = lvl1_dirs
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(lvl1_dirs):
                selected = [lvl1_dirs[idx]]
            else:
                print("Invalid selection. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return

    # Iterate selected first-level directories
    for lvl1 in selected:
        if not lvl1.is_dir():
            continue

        if not lvl1.name.startswith("labelled - "):
            print(f"Skipping first-level folder (does not meet naming criteria): {lvl1}")
            continue

        # Iterate second-level directories
        for lvl2 in lvl1.iterdir():
            if not lvl2.is_dir():
                continue

            if not lvl2.name.startswith("labelled - "):
                print(f"Skipping second-level folder (does not meet naming criteria): {lvl2}")
                continue

            # Deep search for mp4 files containing 'rgb'
            mp4_files = []
            for path, _, files in os.walk(lvl2):
                for f in files:
                    if f.lower().endswith('.mp4') and 'rgb' in f.lower():
                        mp4_files.append(Path(path) / f)

            if not mp4_files:
                print(f"No RGB .mp4 files found in: {lvl2}")
                continue

            for mp4 in mp4_files:
                print(f"RGB video found: {mp4}")

                output_folder = mp4.parent / f"{mp4.stem}_rgb_video_frames_extracted"

                if output_folder.exists():
                    print(f"Folder exists — skipping extraction: {output_folder}")
                    continue

                output_folder.mkdir(parents=True, exist_ok=True, mode=0o2775)

                cap = cv2.VideoCapture(str(mp4))
                frame_idx = 1

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_name = f"frame{frame_idx:03d}.png"
                    cv2.imwrite(str(output_folder / frame_name), frame)
                    frame_idx += 1

                cap.release()
                print(f"Extracted {frame_idx - 1} frames to {output_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract RGB frames from dataset.")
    parser.add_argument("directory", nargs="?", help="Root directory to scan.")
    args = parser.parse_args()

    if not args.directory:
        user_dir = input("Enter the working directory path: ").strip()
        extract_frames(user_dir)
    else:
        extract_frames(args.directory)

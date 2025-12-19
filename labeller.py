import os
import cv2
from tkinter import Tk, filedialog
from tqdm import tqdm
from datetime import datetime


def choose_main_folder():
    """Select main dataset folder."""
    Tk().withdraw()
    folder = filedialog.askdirectory(title="Select main dataset folder")
    return folder


def choose_dataset_folder(main_folder):
    """Select an object folder (e.g., Coffeecup, Cola, etc.)"""
    datasets = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
    if not datasets:
        print("No dataset folders found.")
        return None

    print("\nAvailable datasets:")
    for i, d in enumerate(datasets):
        print(f"{i}: {d}")

    while True:
        try:
            idx = int(input("\nSelect dataset index: "))
            if 0 <= idx < len(datasets):
                return os.path.join(main_folder, datasets[idx])
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def play_video(video_path, speed=1.0):
    """
    Play the video and wait for user input after playback.
    Replays automatically until a valid input (S/F/N/Q) is received.
    """
    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}")
        return None

    print("\nControls: [S]uccess | [F]ail | [N]eutral/Skip | [Q]uit")

    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None

        delay = int(30 / speed)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("front_rgb", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key in [ord('s'), ord('f'), ord('n'), ord('q')]:
                cap.release()
                cv2.destroyAllWindows()
                if key == ord('s'):
                    return 'S'
                elif key == ord('f'):
                    return 'F'
                elif key == ord('n'):
                    return 'N'
                elif key == ord('q'):
                    print("Exiting...")
                    exit()

        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo ended — replaying until labeled (press S/F/N).")


def rename_folder(folder_path, label):
    """Rename folder with prefix based on label and print full new path."""
    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)

    # Skip if any label (any hyphen) is already present
    if "-" in folder_name:
        return

    new_name = f"{label.lower()} - {folder_name}"
    new_path = os.path.join(parent_dir, new_name)
    os.rename(folder_path, new_path)
    print(f"Renamed folder: {new_path}")


def log_review_action(dataset_path, video_path, label):
    """Append a review log entry to review_log.txt in the dataset root."""
    log_path = os.path.join(dataset_path, "review_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_path = os.path.relpath(video_path, dataset_path)
    entry = f"[{timestamp}] {label} | {rel_path}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"[LOGGED] {rel_path} → {label}")


def review_dataset(dataset_path, speed, review_only=False):
    """
    Traverse the dataset hierarchy:
    dataset → testcase → viewpoint → subfolders (0, 1, 2, ...)
    """
    testcases = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for testcase in testcases:
        testcase_path = os.path.join(dataset_path, testcase)
        print(f"\n--- Reviewing testcase: {testcase} ---")

        viewpoints = [v for v in os.listdir(testcase_path)
                      if os.path.isdir(os.path.join(testcase_path, v))]

        for view in viewpoints:
            view_path = os.path.join(testcase_path, view)
            subfolders = [os.path.join(view_path, s) for s in os.listdir(view_path)
                          if os.path.isdir(os.path.join(view_path, s))]

            for sub in tqdm(subfolders, desc=f"{testcase}/{view}"):
                folder_name = os.path.basename(sub)

                # Skip already labeled folders (any containing "-")
                if "-" in folder_name:
                    continue

                video_path = os.path.join(sub, "front_rgb.mp4")

                if not os.path.exists(video_path):
                    print(f"Missing video: {video_path}")
                    input("Press Enter to continue...")
                    continue

                label = play_video(video_path, speed)

                if review_only:
                    log_review_action(dataset_path, video_path, label)
                else:
                    if label == 'S':
                        rename_folder(sub, "success")
                    elif label == 'F':
                        rename_folder(sub, "fail")
                    elif label == 'N':
                        print("Skipped.")

        print(f"\n Finished reviewing testcase {testcase}.")
        input("Press Enter to continue to next testcase...")


def main():
    print("=== Robot Dataset Labeling Tool ===")
    main_folder = choose_main_folder()
    if not main_folder:
        print("No folder selected. Exiting.")
        return

    while True:
        dataset_path = choose_dataset_folder(main_folder)
        if not dataset_path:
            break

        try:
            speed = float(input("Enter playback speed (e.g., 1.0 = normal, 0.5 = slow): "))
        except ValueError:
            speed = 1.0

        mode = input("Run in [R]eview-only or [L]abeling mode? ").strip().lower()
        review_only = (mode == 'r')

        review_dataset(dataset_path, speed, review_only)

        again = input("\nReview another dataset? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\nAll done!")


if __name__ == "__main__":
    main()
import cv2
import os
import sys
import glob
import re


def natural_sort_key(path: str):
    """Split filename into text and integer chunks for natural ordering."""
    name = os.path.basename(path)
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def join_mp4s(input_dir: str, output_dir: str, output_name: str = "joined.mp4"):
    mp4_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")), key=natural_sort_key)

    if not mp4_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    print(f"Found {len(mp4_files)} files:")
    for f in mp4_files:
        print(f"  {os.path.basename(f)}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    first = cv2.VideoCapture(mp4_files[0])
    if not first.isOpened():
        print(f"Failed to open {mp4_files[0]}")
        return

    fps    = first.get(cv2.CAP_PROP_FPS)
    width  = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()

    print(f"\nOutput: {width}x{height} @ {fps:.2f}fps → {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Failed to open VideoWriter")
        return

    total_frames = 0

    for mp4_path in mp4_files:
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"  Skipping {os.path.basename(mp4_path)} — could not open")
            continue

        file_frames = 0
        fname = os.path.basename(mp4_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            file_frames += 1

        cap.release()
        total_frames += file_frames
        print(f"  {fname}: {file_frames} frames")

    writer.release()
    print(f"\nDone — {total_frames} total frames written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python join_mp4s.py <input_dir> <output_dir> [output_name.mp4]")
        sys.exit(1)

    input_dir   = sys.argv[1]
    output_dir  = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else "joined.mp4"

    join_mp4s(input_dir, output_dir, output_name)
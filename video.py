import cv2
import os

def frames_to_video(input_folder, output_path, fps=30, codec='mp4v'):
    """
    Reads all PNG frames in `input_folder` named frame_XXXXXX.png and writes them as a video.

    Args:
        input_folder (str): Path to the folder containing the frames.
        output_path (str): Filename (with path) for the output video (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
        codec (str): FourCC code for the video codec (e.g., 'mp4v', 'XVID', 'H264').
    """
    # List and sort all frame files
    frames = sorted([
        f for f in os.listdir(input_folder)
        if f.startswith("frame_") and f.endswith(".png")
    ])

    if not frames:
        raise RuntimeError(f"No frames found in {input_folder}")

    # Read the first frame to get size
    first_frame = cv2.imread(os.path.join(input_folder, frames[0]))
    height, width, _ = first_frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fname in frames:
        frame_path = os.path.join(input_folder, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: could not read {frame_path}, skipping.")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Modify these paths as needed
    input_folder = "navigation_dataset/rgb"
    output_path = "video.mp4"
    fps = 30
    codec = "mp4v"  # 'mp4v' for .mp4, 'XVID' for .avi, 'H264' if supported

    frames_to_video(input_folder, output_path, fps, codec)

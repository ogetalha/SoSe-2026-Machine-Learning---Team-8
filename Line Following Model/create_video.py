import cv2
import os

def make_video(images_dir, output_video_path, fps=30):
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist!")
        return

    images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not images:
        print("No images found!")
        return

    first_image_path = os.path.join(images_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Compiling {len(images)} frames into video: {output_video_path}")
    for idx, image_name in enumerate(images):
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        video.write(img)
        if (idx + 1) % 500 == 0:
            print(f"Written {idx + 1}/{len(images)} frames...")

    video.release()
    print("Video compilation done successfully!")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_frames = os.path.join(script_dir, "visualization_output", "frames")
    local_video = os.path.join(script_dir, "visualization_output", "steering_preview.mp4")

    desktop_frames = "C:/Users/G7/Desktop/SoSe-2026-Machine-Learning---Team-8/visualization_output/frames"
    desktop_video = "C:/Users/G7/Desktop/SoSe-2026-Machine-Learning---Team-8/visualization_output/steering_preview.mp4"

    if os.path.exists(local_frames) and len([f for f in os.listdir(local_frames) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) > 0:
        make_video(local_frames, local_video)
    else:
        make_video(desktop_frames, desktop_video)

import cv2
import os

def extract_frames(video_path, output_dir="frames"):
    """
    Extracts all frames from a video and saves them as images in an output directory.

    Args:
        video_path (str): The file path to the input video.
        output_dir (str): The directory where the extracted frames will be saved.
    """
    # 1. Check and Create Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Initialize VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    
    print(f"Starting frame extraction from: {video_path}")

    # 3. Read and Save Frames in a Loop
    while True:
        # cap.read() returns a tuple: (success_flag, frame)
        success, frame = cap.read()

        if success:
            # Construct the output filename (e.g., frames/frame_0000.jpg)
            # '{:04d}' formats the number to be 4 digits, padded with leading zeros
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            
            # Save the frame as a JPEG image
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            
            # Optional: Print progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        else:
            # 'success' is False, meaning no more frames to read (end of video)
            break

    # 4. Release the VideoCapture object
    cap.release()
    print("-" * 30)
    print(f"✅ Finished! Extracted {frame_count} frames to the '{output_dir}' directory.")

# --- Usage Example ---
if __name__ == "__main__":
    # !!! CHANGE THIS TO YOUR VIDEO FILE PATH !!!
    INPUT_VIDEO = 'op.mp4' 
    
    # You can change the name of the folder where frames will be saved
    OUTPUT_FOLDER = 'video_frames_output' 

    extract_frames(INPUT_VIDEO, OUTPUT_FOLDER)
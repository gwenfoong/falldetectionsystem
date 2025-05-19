import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import concurrent.futures
import logging
from functools import partial
import datetime

# Configuration
INPUT_FOLDER      = "/Users/gwen/ITSS_Project/data/"
OUTPUT_FOLDER     = "/Users/gwen/ITSS_Project/processed_data"
LOG_FOLDER        = "/Users/gwen/ITSS_Project/logs"  
COMMON_FORMAT     = ".mp4"
FRAME_RATE        = 10
RESIZE_WIDTH      = 640
RESIZE_HEIGHT     = 480
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
MAX_WORKERS       = 4

# Setup logging
os.makedirs(LOG_FOLDER, exist_ok=True)
log_filename = f"data_processing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(LOG_FOLDER, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),  # Save to file
        logging.StreamHandler()        # Print to console
    ]
)
logger = logging.getLogger(__name__)

def log_startup_info():
    """Log initial configuration and system info."""
    logger.info("===== Data Processing Started =====")
    logger.info(f"Input Folder: {INPUT_FOLDER}")
    logger.info(f"Output Folder: {OUTPUT_FOLDER}")
    logger.info(f"Log File: {log_path}")
    logger.info(f"Target Frame Rate: {FRAME_RATE} FPS")
    logger.info(f"Target Resolution: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    logger.info(f"Max Workers: {MAX_WORKERS}")

def is_video_file(filename):
    """Check if file has a supported video extension."""
    is_video = any(filename.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)
    logger.debug(f"Checking {filename}: {'Valid' if is_video else 'Invalid'}")
    return is_video

def convert_video_format(input_path, output_path):
    """Convert video to MP4 format using OpenCV with detailed logging."""
    try:
        logger.info(f"Converting {os.path.basename(input_path)} to MP4")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open: {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.debug(f"Original: {width}x{height} @ {fps:.2f}FPS, {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        
        logger.info(f"Successfully converted: {frame_count} frames processed")
        return True

    except Exception as e:
        logger.error(f"Conversion failed for {input_path}: {str(e)}", exc_info=True)
        return False

def process_single_frame(frame, output_path):
    """Resize, normalize, and randomly augment a single frame."""
    try:
        # ----- AUGMENTATIONS -----
        # 1) Random horizontal flip
        if random.random() < 0.5:
            frame = cv2.flip(frame, 1)

        # 2) Random rotation ±15°
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), angle, 1)
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        # 3) Random brightness/contrast jitter
        alpha = random.uniform(0.8, 1.2)    # contrast
        beta  = random.uniform(-10, 10)     # brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # ----- RESIZE & SAVE -----
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        # normalizing to [0,1] float isn't strictly needed for disk output, but kept:
        frame = (frame / 255.0).astype(np.float32)
        cv2.imwrite(output_path, (frame * 255).astype(np.uint8))
        return True

    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        return False

def extract_frames(video_path, output_folder, video_name):
    """Extract frames with detailed progress logging."""
    try:
        logger.info(f"Extracting frames from {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FPS) / FRAME_RATE))
        expected_frames = total_frames // frame_interval
        
        logger.debug(f"Total frames: {total_frames}, Extracting every {frame_interval} frames")

        frame_count = extracted_count = 0
        with tqdm(total=expected_frames, desc=f"Extracting {video_name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(
                        output_folder,
                        f"{video_name}_frame_{extracted_count:04d}.jpg"
                    )
                    if process_single_frame(frame, frame_filename):
                        extracted_count += 1
                        pbar.update(1)

                frame_count += 1

        cap.release()
        logger.info(f"Extracted {extracted_count}/{expected_frames} frames from {video_name}")
        return extracted_count
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}", exc_info=True)
        return 0

def process_video(video_file, subfolder_path, output_subfolder):
    """Process a single video file with comprehensive logging."""
    try:
        if not is_video_file(video_file):
            logger.warning(f"Skipping non-video file: {video_file}")
            return

        video_path = os.path.join(subfolder_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        logger.info(f"Processing video: {video_name}")

        # Convert video format
        converted_path = os.path.join(output_subfolder, f"{video_name}{COMMON_FORMAT}")
        if not convert_video_format(video_path, converted_path):
            logger.error(f"Failed to process video: {video_name}")
            return

        # Extract frames
        frames_folder = os.path.join(output_subfolder, video_name)
        os.makedirs(frames_folder, exist_ok=True)
        
        frame_count = extract_frames(converted_path, frames_folder, video_name)
        logger.info(f"Completed processing {video_name}: {frame_count} frames")
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}", exc_info=True)

def preprocess_dataset(dataset_folder):
    """
    Process all videos in a category folder (falls or non-falls).
    Expects dataset_folder to contain video files directly.
    """
    dataset_name = os.path.basename(dataset_folder)
    logger.info(f"Starting dataset: {dataset_name}")

    # Prepare output directory for this category
    output_dataset_folder = os.path.join(OUTPUT_FOLDER, dataset_name)
    os.makedirs(output_dataset_folder, exist_ok=True)

    # Gather all video files directly under this folder
    video_files = [
        f for f in os.listdir(dataset_folder)
        if is_video_file(f)
    ]
    logger.info(f"Found {len(video_files)} videos in {dataset_name}")

    # Process each video in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_fn = partial(
            process_video,
            subfolder_path=dataset_folder,
            output_subfolder=output_dataset_folder
        )
        list(tqdm(
            executor.map(process_fn, video_files),
            total=len(video_files),
            desc=f"Processing {dataset_name}"
        ))

    logger.info(f"Completed dataset: {dataset_name}")

def main():
    """Main function with full logging support."""
    log_startup_info()
    
    try:
        datasets = [d for d in os.listdir(INPUT_FOLDER) 
                   if os.path.isdir(os.path.join(INPUT_FOLDER, d))]
        
        if not datasets:
            logger.error("No datasets found in input folder!")
            return

        logger.info(f"Datasets to process: {datasets}")
        
        for dataset in datasets:
            dataset_path = os.path.join(INPUT_FOLDER, dataset)
            preprocess_dataset(dataset_path)

        logger.info("===== Processing Completed Successfully =====")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    import numpy as np
    main()
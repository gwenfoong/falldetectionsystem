import argparse

from pipeline.frame      import evaluate_frame_model
from pipeline.temporal   import evaluate_temporal_model
from pipeline.two_stream import run_two_stream

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fall Detection Inference Pipeline")
    p.add_argument("--mode", choices=["frame","temporal","two_stream"], required=True)
    p.add_argument("--frame-model",    help="path to frame model .pth")
    p.add_argument("--temporal-model", help="path to temporal model .pth")
    p.add_argument("--input-video",    help="path to input .avi/.mp4")
    p.add_argument("--data-root",      help="processed_data root for eval scripts")
    p.add_argument("--output-dir",     required=True)
    args = p.parse_args()

    if args.mode == "frame":
        evaluate_frame_model(
            model_path = args.frame_model,
            data_root  = args.data_root,
            output_dir = args.output_dir
        )
    elif args.mode == "temporal":
        evaluate_temporal_model(
            model_path = args.temporal_model,
            data_root  = args.data_root,
            output_dir = args.output_dir
        )
    else:  # two_stream
        run_two_stream(
            frame_model_path    = args.frame_model,
            temporal_model_path = args.temporal_model,
            input_video         = args.input_video,
            output_dir          = args.output_dir
        )

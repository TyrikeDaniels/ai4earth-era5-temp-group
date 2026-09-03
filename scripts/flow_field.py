"""
Visualize TrajGRU learned flows overlaid on animated ERA5 data.
Aggregates across 5 trajectories and overlays on input/output comparison.
"""

import torch

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader_utils import aggregate_flows, correlate_flows, animated_visual, get_timeframe, print_scores
from models.trajgru_unet import UNetTrajGRU
from utils.train_utils import load_params
from utils.data_loader import get_data_loader

CHECKPOINT_DIR = "./models/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./visualization_outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load checkpoint
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model_b_1.pt")
    if not os.path.exists(best_ckpt_path):
        raise Exception(f"Checkpoint not found at {best_ckpt_path}.")
    
    model_dict = torch.load(best_ckpt_path, map_location=DEVICE)
    print("Checkpoint loaded")
    
    # Load params and data
    params = load_params(1)
    _, dataset = get_data_loader(params, train=False, shuffle=False)
    print("Dataset loaded")
    
    # Load model
    model = UNetTrajGRU(
        input_channels=len(params.era5_channel_input),
        hidden_channels=[16, 32, 64],
        output_channels=1,
    ).to(DEVICE)
    model.load_state_dict(model_dict["model_state_dict"])
    model.eval()
    print("Model loaded")
    
    # Get samples for timeframe
    time_params = {'month': 1, 'start_day': 15, 'end_day': 16}
    samples, _ = get_timeframe(dataset, time_params)
    if not samples:
        print("ERROR: No samples found for timeframe")
        return
    print(f"Found {len(samples)} samples")
    
    # Run model on all samples to get flows
    print("Running model inference...")
    flows_per_sample = []
    with torch.no_grad():
        for sample_x, sample_y in samples:
            sample_x = sample_x.to(DEVICE).unsqueeze(0) 
            _, _, flows = model(sample_x)  # flows = [flow1, flow2, flow3]
            flows_per_sample.append(flows)
    print(f"Computed flows for {len(flows_per_sample)} samples")
    """ NOTE: check experiments.txt file for explanation ^^^^ """

    # Get cosine similarity flows
    input_data = samples[0][0]
    u_agg, v_agg, _ = aggregate_flows(flows_per_sample[0][0])
    scores = correlate_flows((u_agg, v_agg),input_data,dataset.input_channels,channel_indices=None) # Compute cosine similarity
    print_scores(scores, top_n=10)

    # Create animation
    output_path = os.path.join(OUTPUT_DIR, "flows_animated.gif")
    animated_visual(
        dataset=dataset,
        samples=samples,
        flows_per_sample=flows_per_sample,
        stride=3,
        num_frames=len(samples),
        save_path=output_path,
        interval=500,
        input_channels_to_show=["z_600", "z_1000"]
    )

if __name__ == "__main__":
    main()
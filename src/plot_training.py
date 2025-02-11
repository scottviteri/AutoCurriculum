import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import argparse
from datetime import datetime

def compute_window_stats(values: List[float], window_size: int) -> Tuple[List[float], List[float], List[int]]:
    """Compute mean and std over windows of values."""
    means, stds, episodes = [], [], []
    
    for i in range(0, len(values), window_size):
        window = values[i:i+window_size]
        if window:  # Ensure window is not empty
            means.append(np.mean(window))
            stds.append(np.std(window))
            episodes.append(i + window_size//2)  # Center of window
            
    return means, stds, episodes

def get_default_window_size(num_vars: int) -> int:
    """Return appropriate window size based on number of variables."""
    if num_vars == 2:
        return 20
    elif num_vars == 3:
        return 10
    elif num_vars == 4:
        return 5
    else:
        return 1

def plot_training_progress(output_dir: Path):
    """Plot training progress from stats.jsonl file."""
    stats_file = output_dir / 'stats.jsonl'
    trajectories_file = output_dir / 'trajectories.jsonl'
    
    # Read data
    episodes = []
    thresholds = []
    rewards = []
    
    with open(trajectories_file) as f:
        for line in f:
            data = json.loads(line)
            rewards.append(data['avg_reward'])
    
    with open(stats_file) as f:
        for line in f:
            data = json.loads(line)
            episodes.append(data['episode'])
            thresholds.append(data['threshold'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards and threshold
    ax1.plot(episodes, rewards, label='Average Reward')
    ax1.plot(episodes, thresholds, label='Training Threshold', linestyle='--')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training vs non-training rewards
    trained_rewards = []
    untrained_rewards = []
    with open(trajectories_file) as f:
        for line in f:
            data = json.loads(line)
            if data['was_trained']:
                trained_rewards.append(data['avg_reward'])
            else:
                untrained_rewards.append(data['avg_reward'])
    
    if trained_rewards:
        ax2.hist(trained_rewards, alpha=0.5, label='Trained', bins=20)
    if untrained_rewards:
        ax2.hist(untrained_rewards, alpha=0.5, label='Not Trained', bins=20)
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Rewards')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png')
    plt.close()

def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory based on timestamp."""
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise ValueError(f"No run directories found in {runs_dir}")
    
    latest_dir = max(run_dirs, key=lambda d: datetime.strptime(d.name, '%Y%m%d_%H%M%S'))
    return latest_dir

def main():
    parser = argparse.ArgumentParser(description='Plot training progress from logs')
    parser.add_argument('--run-dir', type=str, help='Path to specific run directory')
    parser.add_argument('--runs-dir', type=str, default='runs', 
                       help='Path to directory containing all runs (default: runs)')
    parser.add_argument('--window-size', type=int, 
                       help='Size of window for computing statistics (default: based on num_vars)')
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs_dir = Path(args.runs_dir)
        run_dir = find_latest_run(runs_dir)
        print(f"Plotting most recent run: {run_dir}")
    
    try:
        plot_training_progress(run_dir, window_size=args.window_size)
        print(f"Plot saved to {run_dir}/training_progress.png")
    except Exception as e:
        print(f"Error creating plot: {e}")
        raise

if __name__ == '__main__':
    main() 
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

def plot_training_progress(output_dir: Path) -> None:
    """Plot training progress from stats.jsonl."""
    stats_file = output_dir / 'stats.jsonl'
    if not stats_file.exists():
        return
    
    # Read config to get sd_factor and num_vars
    config_file = output_dir / 'config.json'
    sd_factor = 1.0
    num_vars = 4  # default
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        sd_factor = config.get('sd_factor', 1.0)
        num_vars = config.get('num_vars', 4)
    max_steps = 2 ** num_vars
    
    episodes = []
    means = []
    stds = []
    thresholds = []
    training_ratios = []
    normalized_rewards = []
    avg_rewards = []
    total_episodes = []
    avg_repeats = []
    
    with open(stats_file) as f:
        for line in f:
            data = json.loads(line)
            episodes.append(data['episode'])
            means.append(data['mean'])
            stds.append(data['std'])
            thresholds.append(data['training_threshold'])
            training_ratios.append(data['training_ratio'])
            normalized_rewards.append(data['normalized_reward'])
            avg_rewards.append(data['avg_reward'])
            total_episodes.append(data.get('episode_count', data['episode'] + 1))  # Handle legacy data
            avg_repeats.append(data['avg_repeats'])
    
    # Convert to fraction of max_steps
    repeated_frac = np.array(avg_repeats) / max_steps
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize rewards to 0-1 scale (assuming rewards are already in this range)
    ax.set_ylim(0, 1)
    
    # Plot actual episode averages
    ax.plot(episodes, avg_rewards, label='Episode Rewards', color='blue')
    ax.fill_between(
        episodes,
        np.array(avg_rewards) - np.array(stds)*sd_factor,
        np.array(avg_rewards) + np.array(stds)*sd_factor,
        alpha=0.2,
        color='blue',
        label=f'±{sd_factor}σ Range'
    )
    
    # Plot training ratio
    ax.plot(episodes, training_ratios, label='Training Ratio', color='green', linestyle=':')
    
    # Plot repeated actions as a fraction of max_steps
    ax.plot(episodes, repeated_frac, 
            label='Repeated Actions (frac)', 
            color='purple', 
            linestyle='--',
            alpha=0.7)
    
    # Configure labels and legend
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value (0-1 Scale)")
    ax.set_title("Training Progress")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True)
    
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png')
    plt.close(fig)

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
        plot_training_progress(run_dir)
        print(f"Plot saved to {run_dir}/training_progress.png")
    except Exception as e:
        print(f"Error creating plot: {e}")
        raise

if __name__ == '__main__':
    main() 
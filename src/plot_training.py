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

def plot_training_progress(run_dir: Path, window_size: Optional[int] = None):
    """Create plots from training logs and save to run directory."""
    # Read config to get num_vars
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)
        num_vars = config['num_vars']
    
    # Set window size if not provided
    if window_size is None:
        window_size = get_default_window_size(num_vars)
    
    # Read trajectories and compute average rewards
    avg_rewards = []
    training_flags = []
    thresholds = []  # Add collection of thresholds
    
    with open(run_dir / 'trajectories.jsonl', 'r') as f:
        for line in f:
            traj = json.loads(line)
            if traj['trajectory']['rewards']:
                avg_reward = np.mean(traj['trajectory']['rewards'])
                avg_rewards.append(avg_reward)
                training_flags.append(traj['loss'] is not None)
    
    # Read thresholds from stats file
    with open(run_dir / 'stats.jsonl', 'r') as f:
        for line in f:
            stats = json.loads(line)
            thresholds.append(stats['threshold'])
    
    # Compute statistics over windows
    reward_means, reward_stds, window_episodes = compute_window_stats(avg_rewards, window_size)
    threshold_means, _, _ = compute_window_stats(thresholds[:len(avg_rewards)], window_size)
    
    # Compute training ratios for same windows
    training_ratios = []
    for i in range(0, len(training_flags), window_size):
        window = training_flags[i:i+window_size]
        ratio = sum(window) / len(window)
        training_ratios.append(ratio)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot rewards with error bars
    color = 'blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward / Threshold', color=color)
    ax1.errorbar(window_episodes, reward_means, yerr=reward_stds, 
                color=color, ecolor=color, alpha=0.3, 
                fmt='o-', capsize=5, label='Reward (mean ± std)')
    
    # Plot threshold on same axis
    ax1.plot(window_episodes, threshold_means, color='red', 
            linestyle=':', label='Training Threshold')
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)
    
    # Plot training ratio on second axis
    ax2 = ax1.twinx()
    color = 'green'
    ax2.set_ylabel('Training Ratio', color=color)
    ax2.plot(window_episodes, training_ratios, color=color, 
             linestyle='--', label='Training Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Training Progress')
    plt.tight_layout()
    plt.savefig(run_dir / 'training_progress.png')
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
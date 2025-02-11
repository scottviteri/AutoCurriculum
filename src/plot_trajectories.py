import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import List, Dict
import argparse
from datetime import datetime

def plot_trajectory_rewards(output_dir: Path, num_bins: int = 5):
    """Plot average rewards over time for trajectories, grouped into bins by training order."""
    trajectories = []
    
    # Read all trajectories
    with open(output_dir / 'trajectories.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['trajectory'].get('rewards'):  # Only include trajectories with rewards
                trajectories.append({
                    'rewards': data['trajectory']['rewards'],
                    'was_trained': data['was_trained']
                })
    
    if not trajectories:
        return
    
    # Split trajectories into bins
    trained_trajectories = [t for t in trajectories if t['was_trained']]
    bin_size = max(1, len(trained_trajectories) // num_bins)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot average rewards for each bin
    for bin_idx in range(num_bins):
        start_idx = bin_idx * bin_size
        end_idx = min(start_idx + bin_size, len(trained_trajectories))
        if start_idx >= len(trained_trajectories):
            break
            
        bin_trajectories = trained_trajectories[start_idx:end_idx]
        
        # Find max length and pad shorter trajectories
        max_len = max(len(t['rewards']) for t in bin_trajectories)
        padded_rewards = []
        for t in bin_trajectories:
            rewards = t['rewards']
            if len(rewards) < max_len:
                rewards = rewards + [rewards[-1]] * (max_len - len(rewards))
            padded_rewards.append(rewards)
        
        # Calculate mean rewards at each step
        mean_rewards = np.mean(padded_rewards, axis=0)
        steps = np.arange(len(mean_rewards))
        
        label = f'Trajectories {start_idx}-{end_idx-1}'
        plt.plot(steps, mean_rewards, label=label)
    
    plt.xlabel('Step in Trajectory')
    plt.ylabel('Average Reward')
    plt.title('Reward Progression During Training')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / 'trajectory_rewards.png')
    plt.close()

    """
    # Heatmap visualization code preserved for future use
    def plot_trajectory_rewards_heatmap(run_dir: Path):
        # Create figure with heatmap
        plt.figure(figsize=(12, 6))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((max_length, num_trajs))
        heatmap_data[:] = np.nan
        
        for i, traj in enumerate(trajectories):
            if traj['trajectory']['rewards']:
                rewards = traj['trajectory']['rewards']
                heatmap_data[:len(rewards), i] = rewards
        
        # Plot heatmap
        im = plt.imshow(heatmap_data, aspect='auto', cmap='viridis',
                       extent=[0, num_trajs, max_length+0.5, 0.5],
                       vmin=0, vmax=1)
        
        plt.xlabel('Trajectory Number')
        plt.ylabel('Step in Trajectory')
        plt.title('Reward Heatmap Over Training')
        plt.colorbar(im, label='Reward')
        
        plt.tight_layout()
        plt.savefig(run_dir / 'trajectory_rewards_heatmap.png')
        plt.close()
    """

def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory based on timestamp."""
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise ValueError(f"No run directories found in {runs_dir}")
    
    latest_dir = max(run_dirs, key=lambda d: datetime.strptime(d.name, '%Y%m%d_%H%M%S'))
    return latest_dir

def main():
    parser = argparse.ArgumentParser(description='Plot trajectory rewards progression')
    parser.add_argument('--run-dir', type=str, help='Path to specific run directory')
    parser.add_argument('--runs-dir', type=str, default='runs', 
                       help='Path to directory containing all runs (default: runs)')
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs_dir = Path(args.runs_dir)
        run_dir = find_latest_run(runs_dir)
        print(f"Plotting most recent run: {run_dir}")
    
    try:
        plot_trajectory_rewards(run_dir)
        print(f"Plot saved to {run_dir}/trajectory_rewards.png")
    except Exception as e:
        print(f"Error creating plot: {e}")
        raise

if __name__ == '__main__':
    main() 
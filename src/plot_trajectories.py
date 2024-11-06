import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import List, Dict
import argparse
from datetime import datetime

def plot_trajectory_rewards(run_dir: Path):
    """Plot trajectory rewards as quintile lines."""
    # Read trajectories
    trajectories = []
    with open(run_dir / 'trajectories.jsonl', 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Calculate quintile lines (5 groups)
    num_trajs = len(trajectories)
    quintile_size = num_trajs // 5
    max_length = max(len(traj['trajectory']['rewards']) 
                    for traj in trajectories if traj['trajectory']['rewards'])
    
    # Initialize arrays for quintiles
    quintile_rewards = []
    for i in range(5):
        start_idx = i * quintile_size
        end_idx = (i + 1) * quintile_size if i < 4 else num_trajs
        quintile_trajs = trajectories[start_idx:end_idx]
        
        rewards_sum = np.zeros(max_length)
        counts = np.zeros(max_length)
        
        for traj in quintile_trajs:
            if traj['trajectory']['rewards']:
                rewards = traj['trajectory']['rewards']
                rewards_sum[:len(rewards)] += rewards
                counts[:len(rewards)] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_rewards = np.where(counts > 0, rewards_sum / counts, np.nan)
        quintile_rewards.append(avg_rewards)
    
    # Plot quintile lines
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, rewards in enumerate(quintile_rewards):
        steps = np.arange(1, len(rewards) + 1)
        valid_mask = ~np.isnan(rewards)
        plt.plot(steps[valid_mask], rewards[valid_mask], 
                color=colors[i], 
                label=f'Trajectories {i*20}%-{(i+1)*20}%',
                alpha=0.7)
    
    plt.xlabel('Step in Trajectory')
    plt.ylabel('Average Reward')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Reward Progression Within Trajectories')
    
    plt.tight_layout()
    plt.savefig(run_dir / 'trajectory_rewards.png')
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
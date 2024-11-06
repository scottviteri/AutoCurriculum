import argparse
from datetime import datetime
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import asdict

from curriculum import (
    BooleanFormula,
    generate_random_formula,
    ExpertIterationTrainer,
    Trajectory
)

from plot_training import plot_training_progress
from plot_trajectories import plot_trajectory_rewards

class TrajectoryEncoder(json.JSONEncoder):
    """Custom JSON encoder for Trajectory objects."""
    def default(self, obj):
        if isinstance(obj, Trajectory):
            trajectory_dict = asdict(obj)
            # Convert boolean values to integers for JSON
            trajectory_dict['observations'] = [int(x) for x in trajectory_dict['observations']]
            for action in trajectory_dict['actions']:
                for k, v in action.items():
                    action[k] = int(v)
            return trajectory_dict
        return super().default(obj)

def log_trajectory(file_path: Path, trajectory: Trajectory, loss: float, formula: BooleanFormula):
    """Log a trajectory, its loss, and the formula to a JSONL file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'trajectory': trajectory,
        'loss': float(loss) if loss is not None else None,
        'formula': formula.to_dict()
    }
    with open(file_path, 'a') as f:
        json.dump(entry, f, cls=TrajectoryEncoder)
        f.write('\n')

def get_default_print_interval(num_vars: int) -> int:
    """Return appropriate print interval based on number of variables."""
    if num_vars == 2:
        return 20
    elif num_vars == 3:
        return 10
    elif num_vars == 4:
        return 5
    else:
        return 1

def main():
    parser = argparse.ArgumentParser(description='Train boolean formula solver')
    parser.add_argument('--num-vars', type=int, default=2, help='Number of variables in formulas')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth of formulas')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--model-name', type=str, default='gpt2', help='Model to use')
    
    # First parse to get num_vars
    temp_args, _ = parser.parse_known_args()
    
    # Set default print interval based on num_vars
    default_print_interval = get_default_print_interval(temp_args.num_vars)
    parser.add_argument('--print-interval', type=int, default=default_print_interval,
                       help=f'Number of episodes between progress updates (default: {default_print_interval})')
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Set max_steps based on num_vars
    args.max_steps = 2 ** args.num_vars

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'runs/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Initialize models and tokenizer
    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    critic_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Initialize trainer
    trainer = ExpertIterationTrainer(
        actor_model=actor_model,
        critic_model=critic_model,
        tokenizer=tokenizer,
        num_vars=args.num_vars
    )

    # Training loop
    log_file = output_dir / 'trajectories.jsonl'
    stats_file = output_dir / 'stats.jsonl'
    training_count = 0
    episode_count = 0
    local_rewards = []  # Track rewards since last print
    
    plot_interval = 100  # Plot every 100 episodes
    
    for episode in range(args.num_episodes):
        # Generate random formula
        formula = generate_random_formula(
            num_vars=args.num_vars,
            max_depth=args.max_depth
        )
        
        # Generate trajectory and potentially train
        trajectory = trainer.generate_and_train(
            formula=formula,
            max_steps=args.max_steps
        )
        
        # Get loss if we trained on this trajectory
        loss = None
        if trajectory.rewards is not None:
            avg_reward = sum(trajectory.rewards) / len(trajectory.rewards)
            local_rewards.append(avg_reward)  # Track for local stats
            if avg_reward >= trainer.stats.mean + trainer.stats.std:
                loss = trainer.calculate_loss(trajectory).item()
                training_count += 1
        episode_count += 1
        
        # Log trajectory, loss, and formula
        log_trajectory(log_file, trajectory, loss, formula)
        
        # Log global statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'mean': trainer.stats.mean,
            'std': trainer.stats.std,
            'n': trainer.stats.n,
            'threshold': trainer.stats.mean + trainer.stats.std
        }
        with open(stats_file, 'a') as f:
            json.dump(stats, f)
            f.write('\n')
        
        # Plot progress periodically
        if episode % plot_interval == 0:
            try:
                plot_training_progress(output_dir)
                plot_trajectory_rewards(output_dir)
            except Exception as e:
                print(f"Warning: Failed to create plots: {e}")
        
        # Print progress using configured interval
        if episode % args.print_interval == 0 and episode > 0:
            training_ratio = training_count / max(1, episode_count)
            local_mean = sum(local_rewards) / len(local_rewards) if local_rewards else 0
            local_std = (
                (sum((r - local_mean) ** 2 for r in local_rewards) / len(local_rewards)) ** 0.5
                if len(local_rewards) > 1 else 0
            )
            
            print(f"Episode {episode}/{args.num_episodes}")
            print(f"Recent mean reward: {local_mean:.3f}")
            print(f"Recent reward std: {local_std:.3f}")
            print(f"Global training threshold: {trainer.stats.mean + trainer.stats.std:.3f}")
            print(f"Training ratio: {training_ratio:.2%}")
            print("-" * 40)
            
            # Reset local counters and stats
            training_count = 0
            episode_count = 0
            local_rewards = []

if __name__ == '__main__':
    main() 
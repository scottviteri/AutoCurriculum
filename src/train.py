import argparse
from datetime import datetime
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from dataclasses import asdict
import random

from curriculum import (
    generate_boolean_function,
    ExpertIterationTrainer,
    Trajectory,
    LinearFormula
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
            # Actions: now a list of lists of bool -> list of lists of 0/1
            new_actions = []
            for act_list in trajectory_dict['actions']:
                new_actions.append([int(b) for b in act_list])
            trajectory_dict['actions'] = new_actions

            # Round rewards to 3 decimals
            if trajectory_dict['rewards'] is not None:
                trajectory_dict['rewards'] = [
                    round(r, 3) for r in trajectory_dict['rewards']
                ]
            return trajectory_dict
        return super().default(obj)

def log_trajectory(file_path: Path, trajectory: Trajectory, loss: float, formula):
    """Log a trajectory, its loss, and the formula to a JSONL file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'trajectory': trajectory,
        'was_trained': trajectory.was_trained,
        'avg_reward': trajectory.avg_reward,
        'training_threshold': trajectory.training_threshold if hasattr(trajectory, 'training_threshold') else None,
        'formula': formula.to_dict()
    }
    with open(file_path, 'a') as f:
        json.dump(entry, f, cls=TrajectoryEncoder)
        f.write('\n')

def get_default_episodes(num_vars: int, gen_mode: str) -> int:
    """Return the number of unique possible formulas for this mode."""
    if gen_mode == "linear":
        return 2 ** (num_vars + 1) - 2  # 2^n choices for 'a', times 2 choices for 'b', minus all-zeros and all-ones cases
    else:  # random
        return 2 ** (2 ** num_vars)  # number of possible truth tables

def main():
    parser = argparse.ArgumentParser(description='Train boolean formula solver')
    parser.add_argument('--num-vars', type=int, default=2, help='Number of variables in formulas')
    parser.add_argument('--gen-mode', type=str, default='random',
                        choices=['random','linear'],
                        help='Which kind of boolean function to generate (random or linear).')
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--model-name', type=str, default='gpt2',
                        help='Model to use')
    parser.add_argument('--print-interval', type=int, default=10,
                        help='Number of episodes between progress updates')
    parser.add_argument('--sd-factor', type=float, default=1.0,
                        help='Multiplier for the standard deviation when computing training threshold')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for token sampling')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training updates (default: 1)')

    args = parser.parse_args()
    
    if args.num_episodes is None:
        args.num_episodes = get_default_episodes(args.num_vars, args.gen_mode)
        print(f"[INFO] Using default of {args.num_episodes} episodes "
              f"(number of unique {args.gen_mode} formulas for {args.num_vars} variables).")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'runs/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    critic_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_model.to(device)
    critic_model.to(device)
    
    trainer = ExpertIterationTrainer(actor_model, critic_model, tokenizer, num_vars=args.num_vars, device=device, sd_factor=args.sd_factor, temperature=args.temperature)
    
    log_file = output_dir / 'trajectories.jsonl'
    stats_file = output_dir / 'stats.jsonl'
    training_count = 0
    episode_count = 0
    local_rewards = []
    local_repeats = []
    batch_trajectories = []   # List to accumulate trajectories for batched training.
    
    for episode in range(args.num_episodes-1):
        formula = generate_boolean_function(
            num_vars=args.num_vars,
            gen_mode=args.gen_mode
        )
        
        if args.batch_size > 1:
            # Generate a batch of trajectories using batched generation.
            trajectories = trainer.generate_trajectories_batch(formula, batch_size=args.batch_size, max_steps=2**args.num_vars)
            # Compute rewards for the entire batch.
            rewards_tensor = trainer.batched_model_reward(trajectories)  # Tensor of shape [batch_size, L]
            # Update each trajectory with its average reward and training threshold.
            for i, traj in enumerate(trajectories):
                r = rewards_tensor[i]
                avg = r.mean().item()
                traj.rewards = r.tolist()
                traj.avg_reward = avg
                traj.was_trained = False
                current_threshold = trainer.stats.mean + trainer.sd_factor * trainer.stats.std
                traj.training_threshold = current_threshold
                trainer.update_stats(avg)

            # Recompute current threshold.
            current_threshold = trainer.stats.mean + trainer.sd_factor * trainer.stats.std
            # Filter trajectories that meet the threshold.
            train_batch = [traj for traj in trajectories if traj.avg_reward >= current_threshold]
            if train_batch:
                try:
                    trainer.train_on_trajectories(train_batch)
                    for traj in train_batch:
                        traj.was_trained = True
                except Exception as e:
                    print(f"Error training on batch: {e}")

            # Log each trajectory.
            for traj in trajectories:
                log_trajectory(log_file, traj, traj.avg_reward if traj.was_trained else None, formula)

            # Update cumulative stats for all trajectories in the batch.
            for traj in trajectories:
                local_repeats.append(getattr(traj, "repeated_count", 0))
                local_rewards.append(traj.avg_reward)
                if traj.was_trained:
                    training_count += 1
                episode_count += 1
        else:
            trajectory = trainer.generate_and_train(
                formula=formula,
                max_steps=args.num_vars
            )
            log_trajectory(log_file, trajectory, trajectory.avg_reward if trajectory.was_trained else None, formula)
            local_repeats.append(getattr(trajectory, "repeated_count", 0))
            local_rewards.append(trajectory.avg_reward)
            if trajectory.was_trained:
                training_count += 1
                episode_count += 1
        
        if args.batch_size > 1:
            current_mean = sum([traj.avg_reward for traj in trajectories]) / len(trajectories)
        else:
            current_mean = trajectory.avg_reward
            
        stats = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'avg_reward': current_mean,
            'mean': trainer.stats.mean,
            'std': trainer.stats.std,
            'n': trainer.stats.n,
            'threshold': trainer.stats.mean + trainer.stats.std
        }
        with open(stats_file, 'a') as f:
            json.dump(stats, f)
            f.write('\n')
        
        if episode % args.print_interval == 0 and episode > 0:
            training_ratio = training_count / max(1, episode_count)
            local_mean = sum(local_rewards) / len(local_rewards) if local_rewards else 0
            local_std = (
                (sum((r - local_mean) ** 2 for r in local_rewards) / len(local_rewards)) ** 0.5
                if len(local_rewards) > 1 else 0
            )
            avg_repeats = sum(local_repeats) / len(local_repeats) if local_repeats else 0
            print(f"Episode {episode}/{args.num_episodes}")
            print(f"Recent mean reward: {local_mean}")
            print(f"Recent reward std: {local_std}")
            print(f"Global training threshold: {trainer.stats.mean + trainer.stats.std}")
            print(f"Training ratio: {training_ratio}")
            print(f"Average repeated actions: {avg_repeats}")
            print("-" * 40)
            training_count = 0
            episode_count = 0
            local_rewards = []
            local_repeats = []
            
            try:
                plot_training_progress(output_dir)
                plot_trajectory_rewards(output_dir)
            except Exception as e:
                print(f"Warning: Failed to create plots: {e}")
    
    # Process any remaining trajectories.
    if batch_trajectories:
        try:
            trainer.train_on_trajectories(batch_trajectories)
        except Exception as e:
            print(f"Error training on final batch: {e}")

if __name__ == '__main__':
    main() 

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

from curriculum import (
    generate_boolean_function,
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

    args = parser.parse_args()
    
    # If num_episodes not specified, use the number of unique formulas
    if args.num_episodes is None:
        args.num_episodes = get_default_episodes(args.num_vars, args.gen_mode)
        print(f"[INFO] Using default of {args.num_episodes} episodes "
              f"(number of unique {args.gen_mode} formulas for {args.num_vars} variables).")

    # create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'runs/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # NEW: set for storing formulas we've seen already
    seen_formulas = set()

    # set up models
    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    critic_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_model.to(device)
    critic_model.to(device)

    # trainer
    trainer = ExpertIterationTrainer(actor_model, critic_model, tokenizer, num_vars=args.num_vars, device=device, sd_factor=args.sd_factor)

    # training loop
    log_file = output_dir / 'trajectories.jsonl'
    stats_file = output_dir / 'stats.jsonl'
    training_count = 0
    episode_count = 0
    local_rewards = []
    local_repeats = []  # track repeated actions per trajectory since last print
    
    for episode in range(args.num_episodes):
        # Generate a new formula we haven't seen before
        attempt_count = 0
        while True:
            attempt_count += 1
            formula = generate_boolean_function(
                num_vars=args.num_vars,
                gen_mode=args.gen_mode
            )
            formula_str = json.dumps(formula.to_dict(), sort_keys=True)

            if formula_str not in seen_formulas:
                seen_formulas.add(formula_str)
                break

        # Generate trajectory and potentially train
        trajectory = trainer.generate_and_train(
            formula=formula,
            max_steps=2 ** args.num_vars
        )
        
        # Accumulate repeated actions for averaging
        repeated_count = getattr(trajectory, "repeated_count", 0)
        local_repeats.append(repeated_count)

        # Track if we trained on this trajectory
        if trajectory.was_trained:  # New field we'll add
            training_count += 1
        episode_count += 1
        # Track rewards for all trajectories, not just trained ones
        local_rewards.append(trajectory.avg_reward)
        
        # Log trajectory, loss, and formula
        log_trajectory(log_file, trajectory, trajectory.avg_reward if trajectory.was_trained else None, formula)
        
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
        
        # Print progress using configured interval
        if episode % args.print_interval == 0 and episode > 0:
            training_ratio = training_count / max(1, episode_count)
            local_mean = sum(local_rewards) / len(local_rewards) if local_rewards else 0
            local_std = (
                (sum((r - local_mean) ** 2 for r in local_rewards) / len(local_rewards)) ** 0.5
                if len(local_rewards) > 1 else 0
            )

            avg_repeats = sum(local_repeats) / len(local_repeats) if local_repeats else 0

            print(f"Episode {episode}/{args.num_episodes}")
            print(f"Recent mean reward: {local_mean:.3f}")
            print(f"Recent reward std: {local_std:.3f}")
            print(f"Global training threshold: {trainer.stats.mean + trainer.stats.std:.3f}")
            print(f"Training ratio: {training_ratio:.2%}")
            print(f"Average repeated actions: {avg_repeats:.3f}")
            print("-" * 40)
            
            # Reset local counters and stats
            training_count = 0
            episode_count = 0
            local_rewards = []
            local_repeats = []

            # NEW: Also plot now, each time we print
            try:
                plot_training_progress(output_dir)
                plot_trajectory_rewards(output_dir)
            except Exception as e:
                print(f"Warning: Failed to create plots: {e}")

if __name__ == '__main__':
    main() 

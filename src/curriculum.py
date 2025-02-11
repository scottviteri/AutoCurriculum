import random
from typing import Dict, List, Union, Literal, Tuple, Optional, Callable
from dataclasses import dataclass
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from math import sqrt
from pathlib import Path
from datetime import datetime
import json
import math

OperatorType = Literal["AND", "OR", "NOT", "VAR"]


class BooleanFormula:
    """
    Abstract base for a boolean function on n variables.
    Must implement .evaluate(assignment: Dict[str, bool]) -> bool
    """
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        raise NotImplementedError("Subclasses must implement .evaluate()")


class RandomTableFormula(BooleanFormula):
    """
    A formula that is completely random for all 2^n assignments.
    We store a list of True/False, of length 2^n.
    """
    def __init__(self, num_vars: int):
        super().__init__()
        self.num_vars = num_vars
        self.table_size = 2 ** num_vars
        # For each possible input index, pick True/False at random
        self.truth_table = [random.choice([False, True]) for _ in range(self.table_size)]

    def evaluate(self, assignment: List[bool]) -> bool:
        # Convert list of booleans [True,False,...] to an index
        idx = 0
        for i, bit_val in enumerate(assignment):
            if bit_val:
                idx |= (1 << i)
        return self.truth_table[idx]

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict so we can log this formula."""
        return {
            "type": "RandomTableFormula",
            "num_vars": self.num_vars,
            "truth_table": [int(b) for b in self.truth_table]  # store 0/1 instead of bool
        }


class LinearFormula(BooleanFormula):
    """
    A formula F(x1..xn) = (a1*x1) XOR (a2*x2) XOR ... (an*xn) XOR b
    where a1..an,b are random bits in {0,1}.
    At least one ai must be 1 to avoid constant functions.
    We'll store them, and evaluate by XORing the indicated bits.
    """
    def __init__(self, num_vars: int):
        super().__init__()
        self.num_vars = num_vars
        # pick random bits a1..an, ensuring at least one is 1
        while True:
            self.a = [random.randint(0,1) for _ in range(num_vars)]
            if any(self.a):  # at least one coefficient must be 1
                break
        self.b = random.randint(0,1)

    def evaluate(self, assignment: List[bool]) -> bool:
        val = self.b
        for i, bit_val in enumerate(assignment):
            if self.a[i] == 1 and bit_val:
                val ^= 1
        return (val == 1)

    def __str__(self) -> str:
        """Return a readable representation of the formula."""
        terms = []
        for i, coef in enumerate(self.a):
            if coef == 1:
                terms.append(f"x{i}")
        if not terms:  # should never happen now
            return str(self.b)
        formula = " XOR ".join(terms)
        if self.b == 1:
            formula += " XOR 1"
        return formula

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict so we can log this formula."""
        return {
            "type": "LinearFormula",
            "num_vars": self.num_vars,
            "a": self.a,    # list of 0/1
            "b": self.b     # 0 or 1
        }


def generate_boolean_function(num_vars: int, gen_mode: str) -> BooleanFormula:
    """
    Create a boolean function in one of two ways:
    1) "random": random truth table of size 2^n
    2) "linear": linear XOR-based function
    """
    if gen_mode == "random":
        return RandomTableFormula(num_vars)
    elif gen_mode == "linear":
        return LinearFormula(num_vars)
    else:
        raise ValueError(f"Unknown gen_mode: {gen_mode}")


def generate_all_assignments(num_vars: int) -> List[List[bool]]:
    """
    Generate all possible variable assignments as a list of booleans,
    e.g. [True,False,...].
    """
    assignments = []
    for i in range(2**num_vars):
        assignment = []
        for j in range(num_vars):
            assignment.append(bool((i >> j) & 1))
        assignments.append(assignment)
    return assignments


@dataclass
class Trajectory:
    observations: List[bool]  # Results of evaluating the formula
    actions: List[List[bool]]  # Variable assignments
    rewards: Optional[List[float]] = None  # Added after trajectory completion


PolicyType = Callable[[Trajectory], Dict[str, bool]]


def enumerative_policy(num_vars: int) -> Callable[[Optional[Trajectory]], Dict[str, bool]]:
    """
    Create policy that systematically tries all possible variable assignments.
    
    Args:
        num_vars: Number of variables in formula
        
    Returns:
        Policy function that takes current trajectory and returns next assignment
    """
    # Generate all possible assignments
    assignments = []
    for i in range(2 ** num_vars):
        assignment = []
        for j in range(num_vars):
            assignment.append(bool((i >> j) & 1))
        assignments.append(assignment)
    
    current_idx = 0
    
    def policy(trajectory: Optional[Trajectory] = None) -> Dict[str, bool]:
        nonlocal current_idx
        if current_idx >= len(assignments):
            raise ValueError("Policy has exhausted all possible assignments")
        assignment = assignments[current_idx]
        current_idx += 1
        return assignment
    
    return policy


def model_policy(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, num_vars: int
) -> PolicyType:
    """Creates a policy that uses language model to generate assignments."""

    def policy(trajectory: Trajectory) -> Dict[str, bool]:
        prompt = format_trajectory_prompt(trajectory)
        assignment, _ = get_model_action(model, tokenizer, num_vars, prompt)
        return assignment

    return policy


# Add new type alias for reward functions
RewardFunction = Callable[[Trajectory], List[float]]


def constant_reward(_: Trajectory) -> float:
    """Always returns 1.0 as the reward."""
    return 1.0


def model_reward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> RewardFunction:
    """
    Create reward function that calculates reward based on model's prediction accuracy.
    Returns probability of correct result at each step.
    """
    def reward_fn(trajectory: Trajectory) -> List[float]:
        # Format full trajectory
        input_text = format_trajectory_prompt(trajectory)
        tokens = tokenizer.encode(input_text)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        # Get model predictions
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        # NEW: Dynamically obtain arrow token id instead of hardcoded 4613.
        # note the leading space i n " ->"
        arrow_token_ids = tokenizer.encode(" ->", add_special_tokens=False)
        if len(arrow_token_ids) != 1:
            print("Warning: Tokenization for ' ->' did not result in a single token:", arrow_token_ids)
        arrow_token = arrow_token_ids[0]
        arrow_positions = [i for i, t in enumerate(tokens) if t == arrow_token]
        
        # Calculate reward for each step
        rewards = []
        true_token = tokenizer.encode(" True", add_special_tokens=False)[0]
        false_token = tokenizer.encode(" False", add_special_tokens=False)[0]
        
        for pos, result in zip(arrow_positions, trajectory.observations):
            next_token_logits = logits[0, pos]
            true_false_logits = next_token_logits[[true_token, false_token]]
            probs = torch.softmax(true_false_logits, dim=0)
            # Use the probability of the correct prediction
            prob_correct = probs[0] if result else probs[1]
            rewards.append(prob_correct.item())
        
        return rewards
    return reward_fn


def generate_trajectory(
    formula: BooleanFormula,
    policy: Callable[[Optional[Trajectory]], List[bool]],
    max_steps: Optional[int] = None,
    gen_mode: str = "random",
) -> Trajectory:
    """
    Generate a trajectory by repeatedly applying policy and evaluating formula.
    For linear formulas, replace duplicate actions with random unused ones.
    For random formulas, skip duplicates and continue.
    """
    observations = []
    actions = []
    seen_actions = set()
    repeated_count = 0

    # For linear formulas, maintain a set of unused assignments
    unused_assignments = None
    if gen_mode == "linear":
        # Generate all possible assignments if we're in linear mode
        all_assignments = []
        for i in range(2 ** len(formula.a)):  # formula.a length is num_vars
            assignment = []
            for j in range(len(formula.a)):
                assignment.append(bool((i >> j) & 1))
            all_assignments.append(tuple(assignment))
        unused_assignments = set(all_assignments)

    while True:
        current_trajectory = Trajectory(
            observations=observations,
            actions=actions,
            rewards=None
        )
        
        action = policy(current_trajectory)
        action_tuple = tuple(action)

        if action_tuple in seen_actions:
            repeated_count += 1
            if gen_mode == "linear" and unused_assignments:
                # Replace duplicate with random unused assignment
                action_tuple = random.choice(tuple(unused_assignments))
                action = list(action_tuple)
            else:
                # For random formulas or if no unused assignments left
                continue

        seen_actions.add(action_tuple)
        if gen_mode == "linear" and unused_assignments is not None:
            unused_assignments.discard(action_tuple)

        result = formula.evaluate(action)
        actions.append(action)
        observations.append(result)
        
        if max_steps is not None and len(actions) >= max_steps:
            break
            
    traj = Trajectory(observations=observations, actions=actions, rewards=None)
    setattr(traj, "repeated_count", repeated_count)
    return traj


def format_trajectory_prompt(trajectory: Trajectory) -> str:
    """
    Convert the Trajectory into a textual representation
    that the language model can parse.
    """
    lines = []
    for action, obs in zip(trajectory.actions, trajectory.observations):
        # Convert action to compact string like "[T,F,T,F]"
        action_str = "[" + ",".join("T" if b else "F" for b in action) + "]"
        line = f"{action_str} -> {obs}"
        lines.append(line)

    return "\n".join(lines)


@dataclass
class ModelOutput:
    action_tokens: Tensor
    action_logprobs: Tensor
    value: Optional[Tensor] = None


def encode_assignment(
    assignment: Dict[str, bool], tokenizer: PreTrainedTokenizer
) -> Tensor:
    """
    Encode a variable assignment as tokens.
    Example: "x0=True, x1=False" -> token_ids

    Args:
        assignment: Dictionary of variable assignments
        tokenizer: Tokenizer to use for encoding

    Returns:
        Tensor of token ids
    """
    assignment_str = ", ".join(f"{k}={v}" for k, v in sorted(assignment.items()))
    return torch.tensor(tokenizer.encode(assignment_str))


def decode_assignment(
    tokens: Tensor, tokenizer: PreTrainedTokenizer, num_vars: int
) -> Dict[str, bool]:
    """
    Decode tokens into a variable assignment.

    Args:
        tokens: Tensor of token ids
        tokenizer: Tokenizer to use for decoding
        num_vars: Number of variables expected

    Returns:
        Dictionary of variable assignments
    """
    text = tokenizer.decode(tokens)
    assignments = {}

    # Parse text like "x0=True, x1=False"
    parts = text.split(", ")
    for part in parts:
        if "=" not in part:
            continue
        var, val = part.split("=")
        var = var.strip()
        val = val.strip().lower() == "true"
        if var.startswith("x"):
            assignments[var] = val

    # Validate all variables are present
    for i in range(num_vars):
        var = f"x{i}"
        if var not in assignments:
            raise ValueError(f"Missing variable {var} in decoded assignment")

    return assignments


def get_model_action(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_vars: int,
    prompt: str,
) -> Tuple[List[bool], ModelOutput]:
    """
    Get the model's action by showing it previous assignments and asking for a new one.
    Returns a list of booleans representing the new assignment.
    """
    # Show the model previous assignments and ask for a complete new one
    full_prompt = (
        f"{prompt}\n"
        "Please choose a new assignment that isn't in the list above.\n"
        "New assignment: ["
    )

    assignment = []
    action_tokens = []
    action_logprobs = []

    for i in range(num_vars):
        # Add what we've built so far plus next position
        current_prompt = full_prompt + ",".join(
            "T" if b else "F" for b in assignment
        )
        if i > 0:
            current_prompt += ","

        input_ids = torch.tensor(tokenizer.encode(current_prompt)).unsqueeze(0)
        input_ids = input_ids.to(model.device)

        # Get logits from model
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            # For the next position, we want T or F
            mask = torch.full_like(logits, float("-inf"))
            t_token = tokenizer.encode("T")[0]
            f_token = tokenizer.encode("F")[0]
            mask[:, [t_token, f_token]] = 0
            masked_logits = logits + mask

            # Sample from masked distribution
            probs = torch.nn.functional.softmax(masked_logits, dim=-1)
            chosen_token = torch.multinomial(probs[0], num_samples=1)

            # Calculate log probability of chosen token
            log_prob = torch.nn.functional.log_softmax(masked_logits, dim=-1)[
                0, chosen_token
            ]

            # Update collections
            action_tokens.append(chosen_token)
            action_logprobs.append(log_prob)

            # Update assignment
            is_true = (chosen_token.item() == t_token)
            assignment.append(is_true)

    return assignment, ModelOutput(
        action_tokens=torch.cat(action_tokens),
        action_logprobs=torch.cat(action_logprobs),
    )


class RunningStats:
    """Keep track of mean and standard deviation in an online manner."""
    def __init__(self):
        self.n = 1  # Start with n=1 to avoid division by zero
        self.mean = 0.5  # Default mean for probabilities
        self.M2 = 0.0833  # Variance of uniform distribution on [0,1]
        
    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    @property
    def std(self) -> float:
        if self.n < 2:
            return sqrt(0.0833)  # std of uniform distribution on [0,1]
        return sqrt(self.M2 / (self.n - 1))


class ExpertIterationTrainer:
    def __init__(
        self,
        actor_model: PreTrainedModel,
        critic_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_vars: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.tokenizer = tokenizer
        self.num_vars = num_vars
        self.device = device

        # Ensure models are on the correct device
        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

        # Disable gradients for critic model
        for param in self.critic_model.parameters():
            param.requires_grad = False

        self.stats = RunningStats()
        self.optimizer = torch.optim.Adam(self.actor_model.parameters())
        self.reward_fn = model_reward(self.critic_model, self.tokenizer)

    def update_stats(self, reward: float):
        """Update running statistics with new reward."""
        self.stats.update(reward)

    def train_on_trajectory(self, trajectory: Trajectory) -> None:
        """Train the actor model on a trajectory using policy gradient."""
        # Format trajectory
        input_text = format_trajectory_prompt(trajectory)
        print("trajectory")
        if self.current_formula is not None:
            print(f"Formula: {self.current_formula}")
        print(input_text)

        # Get model output on full trajectory
        tokens = self.tokenizer.encode(input_text)
        input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
        
        self.optimizer.zero_grad()
        outputs = self.actor_model(
            input_ids=input_ids,
            labels=input_ids,
        )
        
        # Weight the loss by rewards
        loss = outputs.loss * torch.tensor(trajectory.rewards).mean()
        loss.backward()
        self.optimizer.step()

    def generate_and_train(
        self, formula: BooleanFormula, max_steps: Optional[int] = None
    ) -> Trajectory:
        """Generate trajectory and potentially train on it."""
        # Store current formula for printing in calculate_loss
        self.current_formula = formula
        
        # Create policy
        policy = model_policy(self.actor_model, self.tokenizer, self.num_vars)
        
        # Generate trajectory
        trajectory = generate_trajectory(
            formula=formula,
            policy=policy,
            max_steps=max_steps,
            gen_mode=formula.__class__.__name__.replace("Formula", "").lower()
        )
        
        # Calculate rewards
        rewards = self.reward_fn(trajectory)
        trajectory.rewards = rewards
        
        # Calculate average reward
        avg_reward = sum(rewards) / len(rewards)
        setattr(trajectory, "avg_reward", avg_reward)
        setattr(trajectory, "was_trained", False)  # Default to False
        # Store the threshold used for this trajectory
        setattr(trajectory, "training_threshold", self.stats.mean + self.stats.std)
        
        # Update statistics
        previous_mean, previous_std = self.stats.mean, self.stats.std
        self.update_stats(avg_reward)
        
        # Train if trajectory is good enough
        if avg_reward >= previous_mean + previous_std:
            self.train_on_trajectory(trajectory)
            setattr(trajectory, "was_trained", True)
            
        return trajectory


def log_trajectory(log_file: Path, trajectory: Trajectory, loss: Optional[float], formula: BooleanFormula):
    """Log trajectory data to jsonl file."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'formula': formula.to_dict(),
        'trajectory': {
            'actions': trajectory.actions,
            'observations': trajectory.observations,
            'rewards': trajectory.rewards
        },
        'loss': loss
    }
    with open(log_file, 'a') as f:
        json.dump(data, f)
        f.write('\n')

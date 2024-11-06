import random
from typing import Dict, List, Union, Literal, Tuple, Optional, Callable
from dataclasses import dataclass
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from math import sqrt

OperatorType = Literal["AND", "OR", "NOT", "VAR"]


class BooleanFormula:
    def __init__(
        self, operator: OperatorType, operands: Union[List["BooleanFormula"], str]
    ):
        """
        Create a boolean formula node.

        Args:
            operator: One of "AND", "OR", "NOT", or "VAR"
            operands: For AND/OR: list of subformulas
                     For NOT: single-element list with subformula
                     For VAR: string with variable name
        """
        self.operator = operator
        self.operands = operands

        # Validate operands
        if operator == "VAR":
            if not isinstance(operands, str):
                raise ValueError("VAR operator requires string operand")
        elif operator == "NOT":
            if not isinstance(operands, list) or len(operands) != 1:
                raise ValueError("NOT operator requires exactly one operand")
        elif operator in ["AND", "OR"]:
            if not isinstance(operands, list) or len(operands) < 2:
                raise ValueError(f"{operator} operator requires at least two operands")
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate the formula given an assignment of variables to boolean values.

        Args:
            assignment: Dictionary mapping variable names to boolean values

        Returns:
            Boolean result of evaluating the formula
        """
        if self.operator == "VAR":
            if self.operands not in assignment:
                raise ValueError(f"Variable {self.operands} not found in assignment")
            return assignment[self.operands]

        elif self.operator == "NOT":
            return not self.operands[0].evaluate(assignment)

        elif self.operator == "AND":
            return all(op.evaluate(assignment) for op in self.operands)

        elif self.operator == "OR":
            return any(op.evaluate(assignment) for op in self.operands)

        raise ValueError(f"Unknown operator: {self.operator}")

    def __str__(self) -> str:
        """Return a string representation of the formula."""
        if self.operator == "VAR":
            return self.operands

        elif self.operator == "NOT":
            return f"NOT({str(self.operands[0])})"

        elif self.operator in ["AND", "OR"]:
            return f"{self.operator}({', '.join(str(op) for op in self.operands)})"

        raise ValueError(f"Unknown operator: {self.operator}")


def generate_random_formula(num_vars: int, max_depth: int) -> BooleanFormula:
    """
    Generate a random boolean formula with the given number of variables.

    Args:
        num_vars: Number of variables to use (named x0, x1, ...)
        max_depth: Maximum depth of the formula tree

    Returns:
        A random BooleanFormula
    """

    def _generate(depth: int) -> BooleanFormula:
        if depth >= max_depth:
            # At max depth, just return a variable
            var_name = f"x{random.randint(0, num_vars-1)}"
            return BooleanFormula("VAR", var_name)

        # Otherwise, randomly choose an operator
        operator = random.choice(["AND", "OR", "NOT"])

        if operator == "NOT":
            return BooleanFormula("NOT", [_generate(depth + 1)])

        else:  # AND or OR
            num_operands = 2
            operands = [_generate(depth + 1) for _ in range(num_operands)]
            return BooleanFormula(operator, operands)

    return _generate(0)


def generate_all_assignments(num_vars: int) -> List[Dict[str, bool]]:
    """
    Generate all possible variable assignments for the given number of variables.

    Args:
        num_vars: Number of variables

    Returns:
        List of all possible assignments
    """
    assignments = []
    for i in range(2**num_vars):
        assignment = {}
        for j in range(num_vars):
            assignment[f"x{j}"] = bool((i >> j) & 1)
        assignments.append(assignment)
    return assignments


@dataclass
class Observation:
    result: bool  # Formula evaluation result
    reward: float  # Could be based on prediction accuracy


@dataclass
class Step:
    action: Dict[str, bool]  # Variable assignment
    observation: Observation


@dataclass
class Trajectory:
    formula: BooleanFormula
    steps: List[Step]


PolicyType = Callable[[Trajectory], Dict[str, bool]]


def enumerative_policy(num_vars: int) -> PolicyType:
    """Creates a policy that enumerates all possible assignments in binary order."""
    assignments = generate_all_assignments(num_vars)
    current_idx = 0

    def policy(trajectory: Trajectory) -> Dict[str, bool]:
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
RewardType = Callable[[Trajectory], float]


def constant_reward(_: Trajectory) -> float:
    """Always returns 1.0 as the reward."""
    return 1.0


def model_reward(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> RewardType:
    """Creates a reward function using the language model's prediction probability."""

    def reward(trajectory: Trajectory) -> float:
        return calculate_prediction_reward(model, tokenizer, trajectory).item()

    return reward


def generate_trajectory(
    formula: BooleanFormula,
    policy: PolicyType,
    reward_fn: RewardType,
    max_steps: Optional[int] = None,
) -> Trajectory:
    """
    Generate a trajectory using the given policy to generate actions.

    Args:
        formula: The boolean formula to evaluate
        policy: Function that generates next action given trajectory
        reward_fn: Function that calculates reward for a trajectory
        max_steps: Maximum number of steps (None for unlimited)

    Returns:
        Trajectory object containing all steps
    """
    steps = []
    step_count = 0

    while max_steps is None or step_count < max_steps:
        try:
            # Get action from policy
            action = policy(Trajectory(formula=formula, steps=steps))

            # Get observation from environment
            result = formula.evaluate(action)

            # Create partial trajectory to calculate reward
            partial_trajectory = Trajectory(
                formula=formula,
                steps=steps
                + [
                    Step(
                        action=action,
                        observation=Observation(result=result, reward=0.0),
                    )
                ],
            )
            reward = reward_fn(partial_trajectory)

            # Add step to trajectory
            steps.append(
                Step(
                    action=action, observation=Observation(result=result, reward=reward)
                )
            )
            step_count += 1

        except ValueError:  # Policy has no more actions
            break

    return Trajectory(formula=formula, steps=steps)


def format_trajectory_prompt(trajectory: Trajectory) -> str:
    """
    Format a trajectory as a prompt for the model.
    Shows all steps in order.

    Args:
        trajectory: The trajectory object

    Returns:
        Formatted prompt string
    """
    lines = []
    for step in trajectory.steps:
        vars_str = ", ".join(f"{k}={v}" for k, v in sorted(step.action.items()))
        lines.append(f"{vars_str} → {step.observation.result}")
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
) -> Tuple[Dict[str, bool], ModelOutput]:
    """
    Get the model's action (variable assignment) for the current step.
    Uses constrained sampling to only choose between True/False for each variable.

    Args:
        model: Language model to use
        tokenizer: Tokenizer for the model
        num_vars: Number of variables
        prompt: Current trajectory formatted as prompt

    Returns:
        Tuple of (assignment dict, model output)
    """
    TRUE_TOKEN = 17821  # Token ID for "True"
    FALSE_TOKEN = 25101  # Token ID for "False"

    action_tokens = []
    action_logprobs = []
    assignment = {}

    for i in range(num_vars):
        # Add the variable prefix to the prompt
        var_prompt = f"{prompt}\nx{i}="
        input_ids = torch.tensor(tokenizer.encode(var_prompt)).unsqueeze(0)

        # Get logits from model
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Mask all logits except True/False tokens
            mask = torch.full_like(logits, float("-inf"))
            mask[:, [TRUE_TOKEN, FALSE_TOKEN]] = 0
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
            is_true = chosen_token.item() == TRUE_TOKEN
            assignment[f"x{i}"] = is_true

            # Add comma for next variable (except last one)
            if i < num_vars - 1:
                prompt = f"{var_prompt}{'True' if is_true else 'False'}, "

    return assignment, ModelOutput(
        action_tokens=torch.cat(action_tokens),
        action_logprobs=torch.cat(action_logprobs),
    )


def calculate_prediction_reward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    trajectory: Trajectory,
) -> Tensor:
    """
    Calculate the reward (log probability) that the model assigns to the final True/False
    token in the complete trajectory.

    Args:
        model: Language model to use
        tokenizer: Tokenizer for the model
        trajectory: Complete trajectory including final observation

    Returns:
        Reward tensor (log probability of correct final token)
    """
    if len(trajectory.steps) == 0:
        return torch.tensor(0.0)  # No reward for empty trajectory

    # Format the complete trajectory as input
    input_text = format_trajectory_prompt(trajectory)
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -2, :]  # Get logits for final True/False token
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get the actual token that was used (True or False)
        final_result = trajectory.steps[-1].observation.result
        target_token = tokenizer.encode(f" {final_result}")[0]
        reward = log_probs[0, target_token]

    return reward


@dataclass
class TrajectoryStats:
    """Statistics about trajectory rewards"""

    mean: float
    std: float
    count: int


class ExpertIterationTrainer:
    def __init__(
        self,
        actor_model: PreTrainedModel,
        critic_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_vars: int,
        learning_rate: float = 1e-5,
    ):
        """
        Initialize trainer for expert iteration.

        Args:
            actor_model: Model to be trained for generating actions
            critic_model: Frozen model for calculating rewards
            tokenizer: Tokenizer for both models
            num_vars: Number of variables in formulas
            learning_rate: Learning rate for actor model updates
        """
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.tokenizer = tokenizer
        self.num_vars = num_vars
        self.optimizer = torch.optim.Adam(actor_model.parameters(), lr=learning_rate)

        # Initialize statistics
        self.stats = TrajectoryStats(mean=-2.0, std=0.1, count=0)

    def calculate_trajectory_reward(self, trajectory: Trajectory) -> float:
        """Calculate average reward across trajectory steps."""
        if not trajectory.steps:
            return 0.0
        return sum(step.observation.reward for step in trajectory.steps) / len(
            trajectory.steps
        )

    def update_stats(self, reward: float):
        """Update running statistics with new trajectory reward."""
        n = self.stats.count
        old_mean = self.stats.mean
        old_std = self.stats.std

        # Update mean
        new_mean = (n * old_mean + reward) / (n + 1)

        # Update std using Welford's online algorithm
        if n > 0:
            new_std = sqrt(
                (
                    n * (old_std**2 + (old_mean - new_mean) ** 2)
                    + (reward - new_mean) ** 2
                )
                / (n + 1)
            )
        else:
            new_std = 1.0

        self.stats = TrajectoryStats(mean=new_mean, std=new_std, count=n + 1)

    def train_on_trajectory(self, trajectory: Trajectory):
        """
        Train actor model if trajectory is significantly better than average.
        Makes the variable assignments (True/False after "=") more likely.

        Args:
            trajectory: Trajectory to potentially learn from
        """
        # Calculate trajectory reward
        reward = self.calculate_trajectory_reward(trajectory)
        previous_mean, previous_std = self.stats.mean, self.stats.std
        # Update statistics
        self.update_stats(reward)

        # Check if trajectory is good enough to learn from
        if reward < previous_mean + previous_std:
            return

        # Prepare input sequence
        input_text = format_trajectory_prompt(trajectory)
        tokens = self.tokenizer.encode(input_text)
        input_ids = torch.tensor(tokens).unsqueeze(0)

        # Create attention mask that only includes True/False tokens after "="
        attention_mask = torch.zeros_like(input_ids)
        equals_positions = [i for i, t in enumerate(tokens) if t == 28]  # "=" token ID
        for pos in equals_positions:
            attention_mask[0, pos + 1] = 1  # Set position after "=" to 1

        # Train step
        self.actor_model.train()
        self.optimizer.zero_grad()

        outputs = self.actor_model(
            input_ids=input_ids,
            labels=input_ids,  # Use same sequence as labels
            attention_mask=attention_mask,  # Only compute loss on True/False tokens
        )
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        self.actor_model.eval()

    def generate_and_train(
        self, formula: BooleanFormula, max_steps: Optional[int] = None
    ):
        """
        Generate a trajectory and potentially train on it.

        Args:
            formula: Boolean formula to solve
            max_steps: Maximum number of steps

        Returns:
            Generated trajectory
        """
        # Create policies and reward function
        policy = model_policy(self.actor_model, self.tokenizer, self.num_vars)
        reward_fn = model_reward(self.critic_model, self.tokenizer)

        # Generate trajectory
        trajectory = generate_trajectory(formula, policy, reward_fn, max_steps)

        # Train if trajectory is good
        self.train_on_trajectory(trajectory)

        return trajectory

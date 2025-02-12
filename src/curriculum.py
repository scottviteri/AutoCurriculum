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
from bitsandbytes.optim import Adam8bit

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
        while True:
            self.a = [random.randint(0,1) for _ in range(num_vars)]
            if any(self.a):
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

    @classmethod
    def from_index(cls, num_vars: int, index: int) -> 'LinearFormula':
        """
        Create a LinearFormula from a given index in the range [1, 2^(num_vars+1)-2].
        We order the formulas as follows:
          - For each valid a ∈ {1,...,2^num_vars - 1} (ensuring a is nonzero),
            and for each b ∈ {0,1} (with b ordered as 0 then 1),
            assign an index. For example, with num_vars = 3, the total count is 2^(3+1)-2 = 14.
        """
        total = 2 ** (num_vars + 1) - 2
        if not (1 <= index <= total):
            raise ValueError(f"Index {index} out of range for linear formulas with {num_vars} variables")
        # Convert to 0-indexed.
        idx = index - 1
        # For each valid a, there are 2 formulas (one for each b)
        a_index = idx // 2  # Ranges from 0 to (2^num_vars - 1) - 1.
        b = idx % 2
        a_val = a_index + 1  # Shift so that a is in [1, 2^num_vars-1].
        # Format a_val as binary with fixed width
        a_bits = list(map(int, bin(a_val)[2:].zfill(num_vars)))
        instance = cls.__new__(cls)
        instance.num_vars = num_vars
        instance.a = a_bits
        instance.b = b
        return instance


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
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, num_vars: int, temperature: float = 1.5
) -> PolicyType:
    """Creates a policy that uses a language model to generate assignments with a given temperature."""
    def policy(trajectory: Trajectory) -> Dict[str, bool]:
        prompt = format_trajectory_prompt(trajectory)
        assignment, _ = get_model_action(model, tokenizer, num_vars, prompt, temperature=temperature)
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
            #print("duplicate", action_tuple)
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
    temperature: float = 1.5
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
            t_token = tokenizer.encode("T", add_special_tokens=False)[0]
            f_token = tokenizer.encode("F", add_special_tokens=False)[0]
            mask[:, [t_token, f_token]] = 0
            masked_logits = logits + mask

            # Apply temperature scaling before softmax:
            scaled_logits = masked_logits / temperature

            # Sample from masked distribution
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            assert probs.nonzero().shape[-1] == 2
            chosen_token = torch.multinomial(probs[0], num_samples=1)

            # Calculate log probability of chosen token
            log_prob = torch.nn.functional.log_softmax(scaled_logits, dim=-1)[
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
        sd_factor: float = 1.0,
        temperature: float = 1.5,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_vars = num_vars
        self.device = device
        self.sd_factor = sd_factor
        self.temperature = temperature

        # Ensure models are on the correct device
        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

        # Disable gradients for critic model
        for param in self.critic_model.parameters():
            param.requires_grad = False

        self.stats = RunningStats()
        self.optimizer = Adam8bit(self.actor_model.parameters(), lr=1e-4)
        self.reward_fn = model_reward(self.critic_model, self.tokenizer)

    def update_stats(self, reward: float):
        """Update running statistics with new reward."""
        self.stats.update(reward)

    def train_on_trajectory(self, trajectory: Trajectory) -> None:
        # Format the full trajectory prompt (includes both proposals and critic answers)
        full_prompt = format_trajectory_prompt(trajectory)
        print("Training on full prompt (with masked answer tokens):")
        if self.current_formula is not None:
            print(f"Formula: {self.current_formula}")
        print(full_prompt)
        tokens = self.tokenizer.encode(full_prompt)
        input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
        labels = list(tokens)
        arrow_token_ids = self.tokenizer.encode(" ->", add_special_tokens=False)
        if len(arrow_token_ids) != 1:
            print("Warning: Unexpected tokenization for ' ->':", arrow_token_ids)
        arrow_token = arrow_token_ids[0]
        i = 0
        while i < len(labels):
            if labels[i] == arrow_token:
                labels[i] = -100
                if i + 1 < len(labels):
                    labels[i+1] = -100
                i += 2
            else:
                i += 1
        labels = torch.tensor(labels, device=self.device).unsqueeze(0)
        self.optimizer.zero_grad()
        outputs = self.actor_model(
            input_ids=input_ids,
            labels=labels,
        )
        loss = outputs.loss * torch.tensor(trajectory.rewards, device=self.device).mean()
        loss.backward()
        self.optimizer.step()

    def train_on_trajectories(self, trajectories: list[Trajectory]) -> None:
        """
        Batch trains the actor model on a list of trajectories.
        1. Converts each trajectory into a full prompt (via format_trajectory_prompt)
           and tokenizes them as a single batch.
        2. Constructs batched labels by copying the tokenized input and masking tokens corresponding
           to the " ->" marker and its following token.
        3. Computes the loss from the batched forward pass, scales it by the average over all trajectories'
           average rewards, and steps the optimizer.
        """
        import torch  # ensure torch is imported
        # Create a list of full prompts.
        prompts = [format_trajectory_prompt(traj) for traj in trajectories]
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Build labels by taking a copy of input_ids and masking out tokens corresponding to the critic answers.
        arrow_token_ids = self.tokenizer.encode(" ->", add_special_tokens=False)
        if len(arrow_token_ids) != 1:
            print("Warning: Unexpected tokenization for ' ->':", arrow_token_ids)
        arrow_token = arrow_token_ids[0]
        
        # Create labels for each batch element.
        batch_labels = []
        input_ids_list = input_ids.tolist()  # list of token lists
        for tokens in input_ids_list:
            labels = tokens.copy()
            i = 0
            while i < len(labels):
                if labels[i] == arrow_token:
                    labels[i] = -100  # Mask the arrow token.
                    if i + 1 < len(labels):
                        labels[i+1] = -100  # Also mask the next token.
                    i += 2
                else:
                    i += 1
            batch_labels.append(labels)
        batch_labels = torch.tensor(batch_labels, device=self.device)
        
        self.optimizer.zero_grad()
        outputs = self.actor_model(input_ids=input_ids, labels=batch_labels)
        
        # Scale loss by the mean of the average rewards from each trajectory.
        avg_rewards = torch.tensor([traj.avg_reward for traj in trajectories], device=self.device)
        scale = avg_rewards.mean()
        loss = outputs.loss * scale
        loss.backward()
        self.optimizer.step()

    def generate_trajectories_batch(self, formula: BooleanFormula, batch_size: int, max_steps: int) -> list[Trajectory]:
        """
        Batch generates trajectories for the given formula.
        For each trajectory, maintain a set of available assignments (in bracketed string format)
        and a set of seen assignments. At each generation step, use batched sampling (masked to allow only
        tokens corresponding to "T" or "F") to propose the next token. If the candidate assignment (when concatenated)
        has already been seen in that trajectory, replace it by sampling a random assignment from the remaining
        available set, incrementing the trajectory's repeated_count.
        
        Returns a list of Trajectory objects with a complete assignment in trajectory.actions
        and corresponding observations.
        """
        # Initialize batch of trajectories.
        trajectories = []
        full_assignments = ["[" + ",".join("T" if b else "F" for b in assign) + "]" 
                            for assign in generate_all_assignments(self.num_vars)]
        for _ in range(batch_size):
            traj = Trajectory(actions=[], observations=[], rewards=[])
            traj.repeated_count = 0
            # Each trajectory gets its own available assignments set (deepcopy not needed, as we rebuild from full_assignments)
            traj.available_assignments = set(full_assignments)
            traj.seen_assignments = set()
            traj.generated_tokens = ""  # will accumulate tokens for the assignment
            trajectories.append(traj)
        
        # For each generation step (we expect exactly num_vars tokens per assignment)
        for step in range(self.num_vars):
            # Build batched prompts: each prompt = full prompt so far + instruction for new token.
            prompts = []
            for traj in trajectories:
                base_prompt = format_trajectory_prompt(traj)
                # Append generation instruction similar to get_model_action.
                prompt = base_prompt + "\nPlease choose an unseen boolean variable assignment in the form [X,Y,...].\nNew assignment: ["
                prompts.append(prompt)
            
            batch_enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = batch_enc["input_ids"].to(self.device)
            attention_mask = batch_enc["attention_mask"].to(self.device)
            
            # Allowed tokens: only "T" and "F"
            t_token = self.tokenizer.encode("T", add_special_tokens=False)[0]
            f_token = self.tokenizer.encode("F", add_special_tokens=False)[0]
            allowed_ids = [t_token, f_token]
            
            outputs = self.actor_model(input_ids=input_ids, attention_mask=attention_mask)
            lengths = attention_mask.sum(dim=1) - 1  # index of last non-padded token for each sample
            last_logits = outputs.logits[torch.arange(input_ids.size(0)), lengths, :]
            
            # Mask logits: allow only allowed_ids tokens.
            mask = torch.full_like(last_logits, float('-inf'))
            mask[:, allowed_ids] = 0
            scaled_logits = (last_logits + mask) / self.temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape: (batch_size,)
            
            for i, traj in enumerate(trajectories):
                new_token = self.tokenizer.decode([sampled[i]]).strip()
                candidate = traj.generated_tokens + new_token  # candidate assignment so far
                full_candidate = "[" + candidate + "]"
                if full_candidate in traj.seen_assignments:
                    # Duplicate detected: choose a random remaining assignment.
                    if traj.available_assignments:
                        replacement = random.choice(list(traj.available_assignments))
                        traj.available_assignments.remove(replacement)
                        traj.repeated_count += 1
                        candidate = replacement[1:-1]  # remove brackets
                        full_candidate = replacement
                    # If no alternative remains, continue with candidate.
                else:
                    # Accept candidate and remove from available if present.
                    if full_candidate in traj.available_assignments:
                        traj.available_assignments.remove(full_candidate)
                    traj.seen_assignments.add(full_candidate)
                traj.generated_tokens = candidate
                # If this is the final token, record the complete assignment and observation.
                if step == self.num_vars - 1:
                    traj.actions.append("[" + candidate + "]")
                    # Convert candidate (e.g., "TFTF") into a list of booleans.
                    action_list = [True if ch == "T" else False for ch in candidate]
                    observation = formula.evaluate(action_list)
                    traj.observations.append(observation)
        return trajectories

    def batched_model_reward(self, trajectories: list[Trajectory]) -> Tensor:
        """
        Batched reward computation for a list of trajectories.
        For each trajectory, the full prompt is tokenized and all occurrences of the arrow token
        (" ->") are identified. For each occurrence, the token immediately following (i.e. index+1)
        is used to retrieve its log-probability from the critic model's output.
        Returns a tensor of shape [batch_size, 2^(num_vars)], where each row contains the log-probs
        for each proposal.
        """
        # Get full prompts for the batch.
        prompts = [format_trajectory_prompt(traj) for traj in trajectories]
        arrow_token = self.tokenizer.encode(" ->", add_special_tokens=False)[0]
        expected = 2 ** self.num_vars

        # For each prompt, tokenize (without padding) and search for all occurrences of arrow_token.
        token_lists = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        reward_indices_list = []
        for tokens in token_lists:
            indices = [i + 1 for i, t in enumerate(tokens) if t == arrow_token and (i + 1) < len(tokens)]
            assert len(indices) == expected
            reward_indices_list.append(indices)

        # Obtain batched encoding (with padding/truncation) of prompts.
        batch_enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch_enc["input_ids"].to(self.device)
        attention_mask = batch_enc["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.critic_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        batch_rewards = []
        for i, indices in enumerate(reward_indices_list):
            traj_rewards = []
            for idx in indices:
                if idx == -1 or idx >= input_ids.size(1):
                    traj_rewards.append(torch.tensor(float('-inf'), device=self.device))
                else:
                    token_id = input_ids[i, idx]
                    lp = log_probs[i, idx, token_id]
                    traj_rewards.append(lp)
            traj_rewards_tensor = torch.stack(traj_rewards)  # shape: [expected]
            batch_rewards.append(traj_rewards_tensor)

        return torch.stack(batch_rewards)  # Tensor of shape [batch_size, expected]

    def generate_and_train(self, formula: BooleanFormula, max_steps: Optional[int] = None) -> Trajectory:
        """
        Single trajectory generation and training (for backward compatibility).
        Uses the old non-batched method.
        """
        self.current_formula = formula
        policy = model_policy(self.actor_model, self.tokenizer, self.num_vars, temperature=self.temperature)
        trajectory = generate_trajectory(
            formula=formula,
            policy=policy,
            max_steps=max_steps,
            gen_mode=formula.__class__.__name__.replace("Formula", "").lower()
        )
        rewards = self.reward_fn(trajectory)
        trajectory.rewards = rewards
        avg_reward = sum(rewards) / len(rewards)
        setattr(trajectory, "avg_reward", avg_reward)
        setattr(trajectory, "was_trained", False)
        threshold = self.stats.mean + self.sd_factor * self.stats.std
        setattr(trajectory, "training_threshold", threshold)
        previous_mean, previous_std = self.stats.mean, self.stats.std
        self.update_stats(avg_reward)
        if avg_reward >= previous_mean + self.sd_factor * previous_std:
            self.train_on_trajectory(trajectory)
            setattr(trajectory, "was_trained", True)
        return trajectory

# ... end of ExpertIterationTrainer class, and rest of file

    def generate_and_train(
        self, formula: BooleanFormula, max_steps: Optional[int] = None
    ) -> Trajectory:
        """Generate trajectory and potentially train on it."""
        # Store current formula for printing in calculate_loss
        self.current_formula = formula
        
        # Create policy with the configured temperature
        policy = model_policy(self.actor_model, self.tokenizer, self.num_vars, temperature=self.temperature)
        
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
        # Store the threshold used for this trajectory, configurable now with sd_factor
        threshold = self.stats.mean + self.sd_factor * self.stats.std
        setattr(trajectory, "training_threshold", threshold)
        
        # Update statistics
        previous_mean, previous_std = self.stats.mean, self.stats.std
        self.update_stats(avg_reward)
        
        # Train if trajectory is good enough using the configurable threshold
        if avg_reward >= previous_mean + self.sd_factor * previous_std:
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

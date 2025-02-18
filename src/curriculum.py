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
        true_token = tokenizer.encode("True", add_special_tokens=False)[0]
        false_token = tokenizer.encode("False", add_special_tokens=False)[0]
        
        for pos, result in zip(arrow_positions, trajectory.observations):
            next_token_logits = logits[0, pos]
            true_false_logits = next_token_logits[[true_token, false_token]]
            logit_T = true_false_logits[0]
            logit_F = true_false_logits[1]
            max_logit = max(logit_T, logit_F)
            exp_T = torch.exp(logit_T - max_logit)
            exp_F = torch.exp(logit_F - max_logit)
            p_T = exp_T / (exp_T + exp_F)
            p_F = exp_F / (exp_T + exp_F)
            prob_correct = p_T if result else p_F
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
        # If no trajectories are provided, nothing to do.
        if not trajectories:
            return

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

    def _finalize_assignment(self, generated_tokens: torch.Tensor, seen_actions: set, all_assignments: set) -> (list[bool], bool):
        """
        Decodes generated tokens into a boolean assignment and checks against seen_actions.
        If the assignment is a duplicate, falls back to a randomly selected assignment
        from the set of available assignments (all_assignments – seen_actions).
        
        Returns a tuple:
          (assignment: List[bool], repeated: bool)
        """
        # The output from batched_policy is now a list of booleans.
        assignment = generated_tokens
        assert len(assignment) == self.num_vars
        repeated = False
        if tuple(assignment) in seen_actions:
            available = list(all_assignments - seen_actions)
            if available:
                assignment = list(random.choice(available))
                repeated = True
        seen_actions.add(tuple(assignment))
        return assignment, repeated

    def batched_policy(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """
        Computes the next action for each trajectory in the batch.
        Builds a prompt for each trajectory by concatenating its current prompt (via format_trajectory_prompt)
        with the text "\nNew action: ", tokenizes the prompts, and then samples a fixed number of tokens.
        The output is a tensor of shape [batch_size, action_length], where action_length = 2*num_vars + 2.
        """
        BASE_PROMPT = "Random New Assignment: ["
        prompts = []
        for traj in trajectories:
            # Use base prompt appended to the formatted trajectory prompt.
            prompt = format_trajectory_prompt(traj) + BASE_PROMPT
            prompts.append(prompt)
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        action_length = self.num_vars  # number of T/F tokens to sample
        # Allowed tokens: only "True" and "False"
        allowed_symbols = ["True", "False"]
        allowed_ids = [self.tokenizer.encode(sym, add_special_tokens=False)[0] for sym in allowed_symbols]
        comma_id = self.tokenizer.encode(",", add_special_tokens=False)[0]
        
        # For each sampling step, sample a T/F token using only logits for T and F, then append a comma.
        for _ in range(action_length):
            outputs = self.actor_model(input_ids=input_ids, attention_mask=attention_mask)
            lengths = attention_mask.sum(dim=1) - 1  # last token index for each sample
            last_logits = outputs.logits[torch.arange(input_ids.size(0)), lengths, :]  # shape: (B, vocab_size)
            # Restrict logits to allowed tokens and apply temperature scaling.
            allowed_logits = last_logits[:, allowed_ids] / self.temperature
            max_logits, _ = torch.max(allowed_logits, dim=-1, keepdim=True)
            exp_logits = torch.exp(allowed_logits - max_logits)
            sum_exp = exp_logits.sum(dim=-1, keepdim=True)
            probs = exp_logits / sum_exp  # shape: (B, 2)

            # Sample from probabilities over T and F.
            sampled_idx = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            allowed_ids_tensor = torch.tensor(allowed_ids, device=self.device)
            sampled_token = allowed_ids_tensor[sampled_idx.squeeze(-1)]
            sampled_token = sampled_token.unsqueeze(1)

            # Append the sampled T/F token.
            input_ids = torch.cat([input_ids, sampled_token], dim=1)
            # Append a comma token after each T/F token.
            comma = torch.full((input_ids.size(0), 1), comma_id, dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, comma], dim=1)

            # Update attention mask accordingly (2 tokens added).
            extra_mask = torch.ones((input_ids.size(0), 2), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

        # Post-process: for each sample, decode the last 2*action_length tokens,
        # filter out commas, and convert "True"/"False" tokens to booleans.
        final_assignments = []
        total_tokens = 2 * action_length
        for i in range(input_ids.size(0)):
            tokens = input_ids[i, -total_tokens:]
            token_strs = [self.tokenizer.decode(tok).strip() for tok in tokens]
            bool_assignment = []
            for token in token_strs:
                if token == "True":
                    bool_assignment.append(True)
                elif token == "False":
                    bool_assignment.append(False)
                # skip commas and any other tokens
            final_assignments.append(bool_assignment)
        return final_assignments

    def generate_trajectories_batch(self, formula: BooleanFormula, batch_size: int, max_steps: int) -> list[Trajectory]:
        """
        Batch generates trajectories for the given formula.
        For each trajectory, we start with a base prompt and then iteratively sample new assignment tokens using
        masked sampling (allowing only tokens corresponding to "T" or "F").
        After max_steps tokens have been generated, the tokens are decoded and combined to form an assignment
        string (e.g. "[T,F,T,T]"), which is then converted to a list of booleans. The formula is immediately
        evaluated on that assignment, and the trajectory stores that assignment and observation.
        Now, it also avoids duplicates by checking against previously generated assignments.
        """
        import torch, random
        
        # No need to build a prompt here—the batched_policy handles prompt formatting.
        # Simply initialize one empty trajectory per batch element.
        trajectories = []
        for _ in range(batch_size):
            traj = Trajectory(actions=[], observations=[], rewards=None)
            traj.seen_actions = set()
            traj.all_possible = set(tuple(a) for a in generate_all_assignments(self.num_vars))
            traj.repeated_count = 0
            trajectories.append(traj)
        
        # Loop for max_steps iterations; each iteration produces one action and observation per trajectory.
        for step in range(max_steps):
            # Call the batched_policy to generate an action for each trajectory.
            action_tokens = self.batched_policy(trajectories)  # Tensor of shape (batch_size, action_length)
            # For each trajectory, decode the action, apply duplicate check, and update trajectories.
            for i, traj in enumerate(trajectories):
                tokens = action_tokens[i]
                assignment, repeated = self._finalize_assignment(tokens, traj.seen_actions, traj.all_possible)
                traj.actions.append(assignment)
                obs = formula.evaluate(assignment)
                traj.observations.append(obs)
                if repeated:
                    traj.repeated_count += 1
        
        return trajectories

    def batched_model_reward(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """
        Batched reward computation for a list of trajectories.
        For each trajectory, the full prompt is tokenized and all occurrences of the arrow token
        (" ->") are identified. For each occurrence, the token immediately following
        (i.e. index+1) is used to retrieve its probability from the critic model's output.
        The correct probability (for " True" if the observation is True, or " False" otherwise)
        is used as the reward. Returns a tensor of shape [batch_size, 2^(num_vars)],
        where each row contains the rewards for each generation step.
        """
        # Construct prompts from trajectories.
        prompts = [format_trajectory_prompt(traj) for traj in trajectories]
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.critic_model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])
        logits = outputs.logits  # shape: (B, seq_len, vocab_size)
        
        # Determine arrow token id.
        arrow_token_ids = self.tokenizer.encode(" ->", add_special_tokens=False)
        if len(arrow_token_ids) != 1:
            print("Warning: Unexpected tokenization for ' ->':", arrow_token_ids)
        arrow_token = arrow_token_ids[0]
        
        # Compute arrow positions using first sample (assumed consistent across batch).
        sample_tokens = batch["input_ids"][0].tolist()
        arrow_positions = [i for i, token in enumerate(sample_tokens) if token == arrow_token]
        # Answer positions: immediately after each arrow.
        answer_positions = [pos + 1 for pos in arrow_positions]
        B = batch["input_ids"].size(0)
        L = len(answer_positions)
        answer_positions_tensor = torch.tensor(answer_positions, device=self.device)
        
        # Gather logits at answer positions: shape (B, L, vocab_size)
        selected_logits = logits[:, answer_positions_tensor, :]
        
        # Get token ids for " True" and " False" (with leading spaces).
        true_token = self.tokenizer.encode(" True", add_special_tokens=False)[0]
        false_token = self.tokenizer.encode(" False", add_special_tokens=False)[0]
        
        # Instead of computing softmax over the entire vocabulary, restrict to the two tokens.
        # Gather logits for only these two tokens.
        true_false_logits = selected_logits[:, :, [true_token, false_token]] / self.temperature  # shape: (B, L, 2)
        # Compute custom softmax over dimension -1.
        max_logits, _ = torch.max(true_false_logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(true_false_logits - max_logits)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        probs = exp_logits / sum_exp  # shape: (B, L, 2)
        
        # Initialize rewards tensor.
        rewards = torch.zeros((B, L), device=self.device)
        # For each trajectory (each batch sample) and for each step in the trajectory,
        # use the reduced probabilities: index 0 corresponds to " True" and index 1 to " False".
        for i, traj in enumerate(trajectories):
            # Assume traj.observations is a list of booleans of length L.
            for j, obs in enumerate(traj.observations):
                if obs:
                    rewards[i, j] = probs[i, j, 0]
                else:
                    rewards[i, j] = probs[i, j, 1]
        return rewards

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

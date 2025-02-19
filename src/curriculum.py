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
import numpy as np

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


def random_unique_policy(num_vars: int) -> Callable[[Optional[Trajectory]], List[bool]]:
    """Policy that generates all possible assignments in random order without repeats."""
    # Generate and shuffle all possible assignments
    assignments = []
    for i in range(2 ** num_vars):
        assignment = []
        for j in range(num_vars):
            assignment.append(bool((i >> j) & 1))
        assignments.append(assignment)
    random.shuffle(assignments)
    
    current_idx = 0
    
    def policy(trajectory: Optional[Trajectory] = None) -> List[bool]:
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


def format_trajectory_prompt(trajectory: Trajectory) -> str:
    """
    Convert the Trajectory into a textual representation
    that the language model can parse.
    """
    lines = []
    for action, obs in zip(trajectory.actions, trajectory.observations):
        # Convert action to compact string like "[T,F,T,F]"
        action_str = "[" + ",".join("True" if b else "False" for b in action) + "]"
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
        device: torch.device = torch.device("cuda:0"),
        sd_factor: float = 1.0,
        temperature: float = 1.0,
        lr: float = 1e-3,
        
    ):
        self.actor_model = actor_model
        self.critic_model = actor_model
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_vars = num_vars
        self.device = device
        self.sd_factor = sd_factor
        self.temperature = temperature
        self.lr = lr

        # Only move the model if it's not using offloading
        if not self._is_offloaded(self.actor_model):
            self.actor_model.to(self.device)
        if not self._is_offloaded(self.critic_model):
            self.critic_model.to(self.device)

        # Disable gradients for critic model
        for param in self.critic_model.parameters():
            param.requires_grad = False

        self.stats = RunningStats()
        self.optimizer = Adam8bit(self.actor_model.parameters(), lr=lr)
        self.reward_fn = self.batched_model_reward

    def _is_offloaded(self, model):
        # This is a simple heuristic: offloaded models typically have parameters on device "meta".
        # Adjust the logic as needed for your use-case.
        for param in model.parameters():
            if param.device.type == "meta":
                return True
        return False

    def update_stats(self, reward: float):
        """Update running statistics with new reward."""
        self.stats.update(reward)

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
        if not trajectories:
            return

        # Prepend instruction matching the reward computation
        instruction = "[INST] Predict evaluations of a hidden linear boolean formula in the form b xor a_1 x_1 xor a_2 x_2 xor ... xor x_n. [/INST]\n"
        instr_tokens = self.tokenizer(instruction, add_special_tokens=False, return_tensors="pt").input_ids[0]
        
        # Tokenize all trajectories
        prompts = [instruction + format_trajectory_prompt(traj) for traj in trajectories]
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Create labels, masking the instruction part
        labels = input_ids.clone()
        labels[:, :len(instr_tokens)] = -100  # Mask instruction tokens
        
        # Get arrow token ID
        arrow_token = self.tokenizer.encode(" ->", add_special_tokens=False)[0]
        
        # Get structural token IDs
        bracket_open = self.tokenizer.encode("[", add_special_tokens=False)[0]
        bracket_close = self.tokenizer.encode("]", add_special_tokens=False)[0]
        comma = self.tokenizer.encode(",", add_special_tokens=False)[0]
        newline = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        structural_tokens = {bracket_open, bracket_close, comma, arrow_token, newline}
        
        # Mask structural tokens and arrow answers
        for batch_idx in range(input_ids.size(0)):
            for pos in range(input_ids.size(1)):
                token = input_ids[batch_idx, pos].item()
                
                # Mask structural tokens (only after instruction)
                if token in structural_tokens:
                    labels[batch_idx, pos] = -100
                    
                # Special handling for arrow token's answer
                if token == arrow_token and (pos + 1) < input_ids.size(1):
                    labels[batch_idx, pos + 1] = -100

        # For autoregressive training, labels should be shifted right
        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        self.optimizer.zero_grad()
        outputs = self.actor_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Scale loss by normalized rewards
        normalized_rewards = [
            (traj.avg_reward - self.stats.mean) / self.stats.std 
            for traj in trajectories
        ]
        scale = torch.tensor(np.mean(normalized_rewards), device=self.device).detach()
        loss = outputs.loss * scale
        
        loss.backward()
        self.optimizer.step()

        # Calculate rewards using batched method
        rewards = self.batched_model_reward(trajectories)
        for traj, traj_rewards in zip(trajectories, rewards):
            traj.rewards = traj_rewards.tolist()
            traj.avg_reward = traj_rewards.mean().item()

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
            traj.avg_reward = 0.0                # Initialize average reward
            traj.successful = False              # Initialize successful flag
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
        # Validate answer positions are consistent across batches
        print("Starting batched_model_reward computation")
        # Prepend instruction to help model understand the task
        instruction = "[INST] Predict evaluations of a hidden linear boolean formula in the form b xor a_1 x_1 xor a_2 x_2 xor ... xor x_n. [/INST]\n"
        prompts = [instruction + format_trajectory_prompt(traj) for traj in trajectories]
        
        # Tokenize all trajectories
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.critic_model(**batch)
        print(f"Model outputs shape: {outputs.logits.shape}")
        logits = outputs.logits  # shape: (B, seq_len, vocab_size)
        
        # Extract logits for " True" and " False" tokens
        true_token = self.tokenizer.encode(" True", add_special_tokens=False)[0]
        false_token = self.tokenizer.encode(" False", add_special_tokens=False)[0]
        print(f"' True' token ID: {true_token}")  # Debug token IDs
        print(f"' False' token ID: {false_token}")
        
        # Determine arrow token id.
        arrow_token_ids = self.tokenizer.encode(" ->", add_special_tokens=False)
        if len(arrow_token_ids) != 1:
            print("Warning: Unexpected tokenization for ' ->':", arrow_token_ids)
        arrow_token = arrow_token_ids[0]
        
        # Compute arrow positions using first sample (assumed consistent across batch).
        sample_tokens = batch["input_ids"][0].tolist()
        arrow_positions = [i for i, token in enumerate(sample_tokens) if token == arrow_token]
        print(f"Found arrow positions: {arrow_positions}")
        
        # Verify arrow positions are consistent across all batches
        for batch_idx in range(1, batch["input_ids"].size(0)):
            current_tokens = batch["input_ids"][batch_idx].tolist()
            for pos in arrow_positions:
                if pos >= len(current_tokens):
                    continue  # Skip if position is beyond padding
                assert current_tokens[pos] == arrow_token, f"Batch {batch_idx} position {pos} has token {current_tokens[pos]} != arrow {arrow_token}"
        
        # Answer positions: immediately after each arrow.
        answer_positions = [pos + 1 for pos in arrow_positions]
        B = batch["input_ids"].size(0)
        L = len(answer_positions)
        answer_positions_tensor = torch.tensor(answer_positions, device=self.device)
        
        # Verify answer positions point to True/False tokens
        for batch_idx in range(batch["input_ids"].size(0)):
            current_tokens = batch["input_ids"][batch_idx].tolist()
            for ans_pos in answer_positions:
                if ans_pos < len(current_tokens):
                    assert current_tokens[ans_pos] in {true_token, false_token}, f"Position {ans_pos} has invalid token {current_tokens[ans_pos]}"
        
        # Gather logits at answer positions: shape (B, L, vocab_size)
        selected_logits = logits[:, answer_positions_tensor, :]
        
        # Instead of computing softmax over the entire vocabulary, restrict to the two tokens.
        # Gather logits for only these two tokens.
        true_false_logits = selected_logits[:, :, [true_token, false_token]] / self.temperature  # shape: (B, L, 2)
        # Compute custom softmax over dimension -1.
        max_logits, _ = torch.max(true_false_logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(true_false_logits - max_logits)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        # Debug: Print sample probabilities
        sample_probs = exp_logits[0, 0, :] / sum_exp[0, 0, :]
        print(f"Sample probabilities (True, False): {sample_probs.tolist()}")
        
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
                    print(f"True case: {probs[i,j,0].item():.3f}") if i == 0 else None
                else:
                    rewards[i, j] = probs[i, j, 1]
        return rewards

    def optimal_trajectory(self, formula: BooleanFormula, num_vars: int) -> Trajectory:
        """
        Generate the "optimal" trajectory based on the optimal assignment strategy described in the README,
        then, if desired, extend it with random unique actions.
        
        The optimal part consists of:
          1. First assignment: all False.
          2. Then, sequentially set the i-th variable to True.
         This yields num_vars+1 assignments.
         
        If max_steps is provided and is greater than num_vars+1, the trajectory is extended by filling
        with randomly selected unique assignments (i.e. assignments not already in the optimal part).
        
        Args:
            formula: A BooleanFormula used to evaluate assignments.
            num_vars: Number of variables in the formula.
            max_steps: Optional total number of steps in the trajectory. If provided and greater than num_vars+1,
                       the trajectory is extended.
        
        Returns:
            Trajectory: The trajectory containing the optimal sequence of actions (extended if applicable)
                        and the corresponding observations.
        """
        max_steps = 2**num_vars
        # Create the base trajectory with optimal assignments.
        traj = Trajectory(observations=[], actions=[], rewards=None)
        optimal_actions = []
        # Step 0: all False.
        all_false = [False] * num_vars
        optimal_actions.append(all_false)
        # Steps 1..n: for each variable, create an assignment with that variable set to True.
        for i in range(num_vars):
            assignment = [False] * num_vars
            assignment[i] = True
            optimal_actions.append(assignment)
        
        # Append optimal assignments to trajectory.
        for action in optimal_actions:
            traj.actions.append(action)
            traj.observations.append(formula.evaluate(action))
        
        # If max_steps is provided and there is room to add extra steps, extend with random unique actions.
        if max_steps is not None and max_steps > len(optimal_actions):
            remaining = max_steps - len(optimal_actions)
            # Generate all possible assignments.
            all_assignments = []
            for i in range(2 ** num_vars):
                assignment = []
                for j in range(num_vars):
                    assignment.append(bool((i >> j) & 1))
                all_assignments.append(assignment)
            # Exclude the assignments already in the optimal sequence.
            optimal_set = {tuple(a) for a in optimal_actions}
            available_assignments = [a for a in all_assignments if tuple(a) not in optimal_set]
            # Shuffle the available assignments.
            random.shuffle(available_assignments)
            fill_count = min(remaining, len(available_assignments))
            additional_actions = available_assignments[:fill_count]
            for action in additional_actions:
                traj.actions.append(action)
                traj.observations.append(formula.evaluate(action))
        return traj

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

def batched_random_unique_policy(num_vars: int, batch_size: int) -> Callable[[List[Trajectory]], List[List[bool]]]:
    """
    Create a batched policy that, for each trajectory in the batch, uses a fresh random unique policy.
    Returns a callable that, given a list of current trajectories, returns a list of assignments (one per trajectory).
    """
    # Create independent policy instances (one per trajectory).
    policies = [random_unique_policy(num_vars) for _ in range(batch_size)]
    
    def batched_policy(trajectories: List[Trajectory]) -> List[List[bool]]:
        actions = []
        for p, traj in zip(policies, trajectories):
            try:
                action = p(traj)
            except ValueError:
                raise ValueError("One of the policies is exhausted")
            actions.append(action)
        return actions
    
    return batched_policy

def generate_trajectories_batch(
    formula: BooleanFormula,
    batch_size: int,
    max_steps: Optional[int],
    batched_policy: Callable[[List[Trajectory]], List[List[bool]]]
) -> List[Trajectory]:
    """
    Generate a batch of trajectories using the provided batched_policy.
    The batched_policy is expected to generate a list of assignments (one for each trajectory)
    given the list of current trajectories.
    """
    # Initialize a list of empty trajectories.
    trajectories = [Trajectory(observations=[], actions=[], rewards=None) for _ in range(batch_size)]
    steps = 0
    while True:
        try:
            actions_batch = batched_policy(trajectories)
        except ValueError:
            break
        for traj, action in zip(trajectories, actions_batch):
            result = formula.evaluate(action)
            traj.actions.append(action)
            traj.observations.append(result)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return trajectories

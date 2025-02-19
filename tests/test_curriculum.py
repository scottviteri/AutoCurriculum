import pytest
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from src.curriculum import (
    RandomTableFormula,
    LinearFormula,
    generate_boolean_function,
    generate_all_assignments,
    format_trajectory_prompt,
    encode_assignment,
    decode_assignment,
    enumerative_policy,
    model_policy,
    ExpertIterationTrainer,
    Trajectory,
    batched_random_unique_policy,
    generate_trajectories_batch,
)

# ------------------------------------------------------------------
# Tests for Boolean Formula Implementations
# ------------------------------------------------------------------

def test_random_table_formula_evaluation():
    """Test that RandomTableFormula returns a boolean result."""
    num_vars = 2
    formula = RandomTableFormula(num_vars=num_vars)
    # Create a sample assignment for 2 variables
    assignment = [True, False]
    result = formula.evaluate(assignment)
    assert isinstance(result, bool)

def test_linear_formula_evaluation():
    """Test that LinearFormula returns a boolean result for all assignments."""
    num_vars = 2
    formula = LinearFormula(num_vars=num_vars)
    # Test all possible assignments (there are 2^num_vars assignments)
    for i in range(2 ** num_vars):
        assignment = []
        for j in range(num_vars):
            assignment.append(bool((i >> j) & 1))
        result = formula.evaluate(assignment)
        assert isinstance(result, bool)

def test_generate_boolean_function():
    """Test that generate_boolean_function returns the correct formula type."""
    # For random mode, should return a RandomTableFormula
    formula_random = generate_boolean_function(num_vars=2, gen_mode="random")
    from src.curriculum import RandomTableFormula
    assert isinstance(formula_random, RandomTableFormula)
    
    # For linear mode, should return a LinearFormula
    formula_linear = generate_boolean_function(num_vars=2, gen_mode="linear")
    from src.curriculum import LinearFormula
    assert isinstance(formula_linear, LinearFormula)

def test_dummy_formula_evaluation():
    class DummyFormula:
        def __init__(self, num_vars):
            self.num_vars = num_vars
        def evaluate(self, assignment):
            # For testing, simply return True if an even number of True, else False.
            return sum(assignment) % 2 == 0

    num_vars = 3
    dummy = DummyFormula(num_vars)
    assert dummy.evaluate([True, False, True]) == True  # 2 True's -> even -> True
    assert dummy.evaluate([True, False, False]) == False  # 1 True -> odd -> False

# ------------------------------------------------------------------
# Tests for Generating Assignments and Formatting
# ------------------------------------------------------------------

def test_generate_all_assignments():
    """Test that generate_all_assignments produces all unique assignments."""
    num_vars = 2
    assignments = generate_all_assignments(num_vars)
    # There should be 2^num_vars unique assignments.
    assert len(assignments) == 2 ** num_vars
    for a in assignments:
        assert isinstance(a, list)
        assert len(a) == num_vars
        # Each element in the assignment should be boolean.
        for b in a:
            assert isinstance(b, bool)

def test_enumerative_policy():
    """Test that the enumerative policy returns assignments in order and raises error when exhausted."""
    num_vars = 2
    policy = enumerative_policy(num_vars)
    # There are 2^num_vars assignments expected
    expected_assignments = 2 ** num_vars
    results = []
    for _ in range(expected_assignments):
        action = policy()
        assert isinstance(action, list)
        assert len(action) == num_vars
        results.append(action)
    # The next call must raise a ValueError.
    with pytest.raises(ValueError):
        policy()
    # Verify the first assignment represents 0 (i.e. [False, False])
    assert results[0] == [False, False]

def test_format_trajectory_prompt():
    """Test that formatting a trajectory returns a prompt string with expected structure."""
    num_vars = 2
    # Use a linear formula so that the 'gen_mode' string aligns with the trajectory generation.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    policy = enumerative_policy(num_vars)
    trajectory = generate_trajectory(formula, policy, max_steps=2, gen_mode="linear")
    prompt = format_trajectory_prompt(trajectory)
    lines = prompt.strip().split("\n")
    # Expect two lines for a two-step trajectory.
    assert len(lines) == 2
    for line in lines:
        # Each line should contain the arrow marker.
        assert "->" in line
        # The assignment portion should be enclosed in square brackets.
        assert line.strip().startswith("[")

# ------------------------------------------------------------------
# Tests for Rewards, Encoding/Decoding, and Model Policies
# ------------------------------------------------------------------

def test_model_reward():
    """Test that model_reward returns a list of floats in [0, 1]."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_vars = 2
    # Use random formula mode for reward test.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="random")
    policy = enumerative_policy(num_vars)
    trajectory = generate_trajectory(formula, policy, max_steps=2, gen_mode="random")
    
    reward_fn = model_reward(model, tokenizer)
    rewards = reward_fn(trajectory)
    # Ensure rewards is a list matching the number of observations.
    assert isinstance(rewards, list)
    assert len(rewards) == len(trajectory.observations)
    for r in rewards:
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0

def test_encode_decode_assignment():
    """Test that encoding and then decoding a variable assignment returns the original assignment."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assignment = {"x0": True, "x1": False}
    tokens = encode_assignment(assignment, tokenizer)
    assert isinstance(tokens, torch.Tensor)
    decoded = decode_assignment(tokens, tokenizer, num_vars=2)
    assert decoded == assignment

def test_model_policy():
    """Test that model_policy produces a valid assignment using GPT-2."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_vars = 2
    # Construct a dummy trajectory.
    dummy_trajectory = Trajectory(observations=[], actions=[], rewards=None)
    policy_fn = model_policy(model, tokenizer, num_vars=num_vars, temperature=1.0)
    assignment = policy_fn(dummy_trajectory)
    # Check that the result is a list of booleans with the expected length.
    assert isinstance(assignment, list)
    assert len(assignment) == num_vars
    for b in assignment:
        assert isinstance(b, bool)

# ------------------------------------------------------------------
# Tests for Expert Iteration Training Updates
# ------------------------------------------------------------------

def test_expert_iteration_updates_stats():
    """Test that ExpertIterationTrainer updates its running statistics after each training step."""
    actor_model = AutoModelForCausalLM.from_pretrained("gpt2")
    critic_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_vars = 2
    trainer = ExpertIterationTrainer(
        actor_model=actor_model,
        critic_model=critic_model,
        tokenizer=tokenizer,
        num_vars=num_vars,
    )
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="random")
    initial_n = trainer.stats.n
    num_episodes = 3
    for _ in range(num_episodes):
        trajectory = trainer.generate_and_train(formula, max_steps=2)
    # The running statistics counter should have increased by the number of episodes.
    assert trainer.stats.n == initial_n + num_episodes

# ------------------------------------------------------------------
# Tests for Tokenization Consistency in Assignment Formatting
# ------------------------------------------------------------------

def test_token_consistency_in_assignment_format():
    """
    Test that fixed-format assignment strings tokenized with GPT-2 have the same number of tokens,
    regardless of the specific boolean values.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_vars = 3
    # Create two different assignments.
    assignment1 = [True, False, True]
    assignment2 = [False, True, False]
    # Format these assignments using a consistent method.
    str1 = "[" + ",".join("T" if b else "F" for b in assignment1) + "]"
    str2 = "[" + ",".join("T" if b else "F" for b in assignment2) + "]"
    tokens1 = tokenizer.encode(str1)
    tokens2 = tokenizer.encode(str2)
    # They should have the same number of tokens.
    assert len(tokens1) == len(tokens2), f"Token lengths differ: {len(tokens1)} != {len(tokens2)}"

# ------------------------------------------------------------------
# Additional Tests for Special Token Consistency, Batched Tokenization,
# and Full Forward Pass Integration
# ------------------------------------------------------------------

def test_trajectory_arrow_count():
    """
    Test that tokenizer.encode(" ->")[0] appears exactly 2**num_vars times
    in a trajectory prompt string.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 2
    max_steps = 2 ** num_vars  # Assuming max_steps is set to 2^num_vars
    # Use a linear formula to ensure consistent formatting.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    policy = enumerative_policy(num_vars)
    trajectory = generate_trajectory(formula, policy, max_steps=max_steps, gen_mode="linear")
    prompt = format_trajectory_prompt(trajectory)
    arrow_token = tokenizer.encode(" ->", add_special_tokens=False)[0]
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    count = tokens.count(arrow_token)
    assert count == max_steps, (
        f"Expected arrow token to appear {max_steps} times, but found {count} times"
    )

def test_batched_tokenization_and_arrow_positions_consistency():
    """
    Generate multiple trajectories and verify that when using batched tokenization,
    the special token (" ->") appears at identical token positions for all trajectories.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 2
    max_steps = 2 ** num_vars  # e.g., 4 steps when num_vars == 2
    # We'll use random mode here to generate trajectories.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="random")
    
    # Generate three trajectories using independent enumerative policies.
    policy1 = enumerative_policy(num_vars)
    policy2 = enumerative_policy(num_vars)
    policy3 = enumerative_policy(num_vars)
    trajectory1 = generate_trajectory(formula, policy1, max_steps=max_steps, gen_mode="random")
    trajectory2 = generate_trajectory(formula, policy2, max_steps=max_steps, gen_mode="random")
    trajectory3 = generate_trajectory(formula, policy3, max_steps=max_steps, gen_mode="random")
    
    prompt1 = format_trajectory_prompt(trajectory1)
    prompt2 = format_trajectory_prompt(trajectory2)
    prompt3 = format_trajectory_prompt(trajectory3)
    
    batch_prompts = [prompt1, prompt2, prompt3]
    batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    
    # For each trajectory, find indices where the arrow token appears.
    arrow_token = tokenizer.encode(" ->", add_special_tokens=False)[0]
    def find_arrow_positions(tokens_list):
        return [i for i, token in enumerate(tokens_list) if token == arrow_token]
    
    positions_list = []
    for prompt in batch_prompts:
        tokens_ids = tokenizer.encode(prompt, add_special_tokens=False)
        positions_list.append(find_arrow_positions(tokens_ids))
    
    # All trajectories should have the same positions for the arrow token.
    for pos in positions_list[1:]:
        assert pos == positions_list[0], (
            f"Inconsistent arrow positions across trajectories: {positions_list}"
        )

def test_full_forward_pass_integration():
    """
    Test a full forward pass of a batched trajectory prompt.
    Confirm that the model processes a batch without error and that the logits output
    has the expected shape.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 2
    max_steps = 2 ** num_vars  # For example, 4 steps.
    
    # Generate a batch of three trajectories using a linear formula.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    policy1 = enumerative_policy(num_vars)
    policy2 = enumerative_policy(num_vars)
    policy3 = enumerative_policy(num_vars)
    
    trajectory1 = generate_trajectory(formula, policy1, max_steps=max_steps, gen_mode="linear")
    trajectory2 = generate_trajectory(formula, policy2, max_steps=max_steps, gen_mode="linear")
    trajectory3 = generate_trajectory(formula, policy3, max_steps=max_steps, gen_mode="linear")
    
    prompts = [
        format_trajectory_prompt(trajectory1),
        format_trajectory_prompt(trajectory2),
        format_trajectory_prompt(trajectory3)
    ]
    
    # Batch encode the prompts with padding.
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check that the outputs logits shape is [batch_size, seq_length, vocab_size].
    assert outputs.logits.shape[0] == len(prompts)
    assert outputs.logits.shape[1] == input_ids.shape[1]
    
    # Verify the logits for the last token are as expected.
    last_token_logits = outputs.logits[:, -1, :]
    assert last_token_logits.shape[0] == len(prompts)
    assert last_token_logits.shape[1] == model.config.vocab_size

def test_batched_generation_step():
    """
    Test a batched generation step in the style of get_model_action (masked sampling).
    Instead of manually performing the token-sampling loop, this test now uses
    generate_trajectories_batch (which produces complete assignments for a batch) and then verifies
    that each generated assignment is of the form "[T,F,T,T]" for num_vars tokens.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 4  # Example: using 4 variables.
    
    # Create a dummy linear formula.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    
    # Initialize the trainer with our actor model; we reuse the same model for critic here.
    trainer = ExpertIterationTrainer(
        actor_model=model,
        critic_model=model,
        tokenizer=tokenizer,
        num_vars=num_vars,
    )
    
    # Use the new batched generation method.
    trajectories = trainer.generate_trajectories_batch(formula, batch_size=3, max_steps=num_vars)
    
    assignments = []
    for traj in trajectories:
        # traj.actions is a list of assignments; here we expect exactly one complete assignment.
        assignment = traj.actions[0]
        token_strs = ["T" if b else "F" for b in assignment]
        assignment_str = "[" + ",".join(token_strs) + "]"
        assignments.append(assignment_str)
        # Verify that each token is in the allowed set.
        for token in token_strs:
            assert token in ["T", "F"], f"Token '{token}' not in allowed set ('T', 'F')"
    
    # Optionally print the generated assignments for manual inspection.
    for assignment_str in assignments:
        print("Generated assignment:", assignment_str)
    
    # Verify the assignment formatting using a regex.
    pattern = r"^\[(?:T|F)(?:,\s?(?:T|F)){" + f"{num_vars - 1}" + r"}\]$"
    for assignment_str in assignments:
        assert re.match(pattern, assignment_str), (
            f"Assignment '{assignment_str}' does not match expected format for {num_vars} variables."
        )

def test_assignment_token_count():
    """
    Test that the tokenization of assignment strings of the form [T,F,T,T] (or similar)
    always results in 2*num_vars+2 tokens. Print out the actual token IDs for verification.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # You may want to set the pad token to match other tests.
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 4  # Change this to test other numbers of variables as needed.
    
    # Test a few different assignment examples.
    assignments = [
         [True, False, True, True],
         [False, True, False, True],
         [True, True, True, True],
         [False, False, False, False],
    ]
    expected_tokens = 2 * num_vars + 1
    for assignment in assignments:
         # Format the assignment: e.g. "[T,F,T,T]"
         assignment_str = "[" + ",".join("T" if b else "F" for b in assignment) + "]"
         token_ids = tokenizer.encode(assignment_str, add_special_tokens=False)
         decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
         print(f"Assignment: {assignment_str} -> Tokens: {token_ids} -> Decoded: {decoded_tokens}")
         assert len(token_ids) == expected_tokens, (
             f"Expected {expected_tokens} tokens, got {len(token_ids)} for {assignment_str}"
         )

def test_batched_model_reward_arrow_positions():
    """
    Test that in batched tokenization for model reward computation,
    the arrow token " ->" appears at identical indices across all batch elements.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 3
    max_steps = 2 ** num_vars  # For example, for 3 variables, 8 steps.
    
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    
    trajectories = []
    for _ in range(3):
        policy = enumerative_policy(num_vars)
        traj = generate_trajectory(formula, policy, max_steps=max_steps, gen_mode="linear")
        trajectories.append(traj)
    
    prompts = [format_trajectory_prompt(traj) for traj in trajectories]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    arrow_token = tokenizer.encode(" ->", add_special_tokens=False)[0]
    positions_list = []
    for tokens in batch["input_ids"]:
        tokens_list = tokens.tolist()
        positions = [i for i, token in enumerate(tokens_list) if token == arrow_token]
        positions_list.append(positions)
    # Check that all positions lists are identical.
    for positions in positions_list[1:]:
        assert positions == positions_list[0], f"Arrow positions inconsistent: {positions_list}"

def test_batched_training_threshold_filtering():
    """
    Create dummy trajectories with preset average rewards and a dummy trainer (with fixed stats).
    Verify that only trajectories with avg_reward above the threshold are selected for training.
    """
    # Create three dummy trajectories with manually assigned avg_reward and was_trained false.
    traj1 = Trajectory(actions=[[True, False]], observations=[True], rewards=None)
    traj2 = Trajectory(actions=[[False, True]], observations=[False], rewards=None)
    traj3 = Trajectory(actions=[[True, True]], observations=[True], rewards=None)
    traj1.avg_reward = 0.8
    traj2.avg_reward = 0.4
    traj3.avg_reward = 0.9
    traj1.was_trained = False
    traj2.was_trained = False
    traj3.was_trained = False
    dummy_trajs = [traj1, traj2, traj3]

    # Create a dummy trainer.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    actor = AutoModelForCausalLM.from_pretrained("gpt2")
    critic = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    trainer = ExpertIterationTrainer(actor_model=actor, critic_model=critic, tokenizer=tokenizer, num_vars=2)

    # Manually set running stats: mean = 0.5, std = 0.1, sd_factor=1.0 so threshold = 0.6.
    trainer.stats.mean = 0.5
    trainer.stats.n = 2  # so that std = sqrt(M2/(n-1))
    trainer.stats.M2 = 0.01  # std = 0.1
    trainer.sd_factor = 1.0
    threshold = trainer.stats.mean + trainer.sd_factor * trainer.stats.std  # 0.5 + 0.1 = 0.6

    # Filter trajectories based on threshold.
    train_batch = [traj for traj in dummy_trajs if traj.avg_reward >= threshold]
    assert len(train_batch) == 2, "Expected 2 trajectories to pass the threshold"

    # Simulate training on the filtered batch.
    trainer.train_on_trajectories(train_batch)
    # For test purposes, mark those that were used for training.
    for traj in train_batch:
        traj.was_trained = True

    # Check that only trajectories with avg_reward above threshold are marked as trained.
    for traj in dummy_trajs:
        if traj.avg_reward >= threshold:
            assert traj.was_trained is True, "Trajectory above threshold was not marked as trained."
        else:
            assert traj.was_trained is False, "Trajectory below threshold should not be marked as trained."

def test_batch_training_integration():
    """
    Simulate the batch processing in main():
    - Generate a batch of trajectories using generate_trajectories_batch.
    - Obtain rewards by calling batched_model_reward (which returns a tensor of rewards
      for all trajectories).
    - Compute each trajectory's average reward from the rewards.
    - Filter and simulate training on those trajectories.
    Verify that only trajectories with avg_reward above the computed threshold get marked as trained.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    actor = AutoModelForCausalLM.from_pretrained("gpt2")
    critic = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 2
    trainer = ExpertIterationTrainer(actor_model=actor, critic_model=critic, tokenizer=tokenizer, num_vars=num_vars)

    # Generate a batch of trajectories.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    trajectories = trainer.generate_trajectories_batch(formula, batch_size=3, max_steps=num_vars)

    # Obtain rewards from the critic over the full batch.
    rewards_tensor = trainer.batched_model_reward(trajectories)
    # rewards_tensor should have shape [3, L] where L is the number of answer positions.

    # For each trajectory, compute the average reward and update trajectory properties.
    for i, traj in enumerate(trajectories):
        r = rewards_tensor[i]
        avg = r.mean().item()
        traj.rewards = r.tolist()
        traj.avg_reward = avg
        traj.was_trained = False
        threshold = trainer.stats.mean + trainer.sd_factor * trainer.stats.std
        traj.training_threshold = threshold
        trainer.update_stats(avg)

    # Compute the current threshold from trainer stats.
    current_threshold = trainer.stats.mean + trainer.sd_factor * trainer.stats.std
    # Filter trajectories whose avg_reward is above threshold.
    train_batch = [traj for traj in trajectories if traj.avg_reward >= current_threshold]

    # Train on the filtered batch.
    trainer.train_on_trajectories(train_batch)
    for traj in train_batch:
        traj.was_trained = True

    # Check that only trajectories with avg_reward above current threshold are marked as trained.
    for traj in trajectories:
        if traj.avg_reward >= current_threshold:
            assert traj.was_trained is True, "Trajectory above threshold was not marked as trained."
        else:
            assert traj.was_trained is False, "Trajectory below threshold should not be marked as trained."

def test_batched_assignment_uniqueness():
    """
    Test that generate_trajectories_batch avoids duplicate assignments within a trajectory.
    We simulate a scenario by generating a batch and then invoking the helper twice.
    The second call should fallback if it produces a duplicate.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import random

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 4
    # Use a linear formula for consistency.
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    
    trainer = ExpertIterationTrainer(actor_model=model, critic_model=model, tokenizer=tokenizer, num_vars=num_vars)
    
    # Generate a batch of trajectories (each obtains one action).
    trajectories = trainer.generate_trajectories_batch(formula, batch_size=3, max_steps=num_vars)
    
    # Save the first assignment from each trajectory.
    first_actions = [traj.actions[0] for traj in trajectories]
    
    # Now simulate generating a second action for each trajectory.
    # We'll call _finalize_assignment directly on a fixed set of tokens that decode to the same assignment.
    # First, encode a fixed token sequence corresponding to the first action.
    fixed_tokens = torch.tensor([tokenizer.encode("T", add_special_tokens=False)[0] 
                                 for _ in range(num_vars)], device=trainer.device)
    # For each trajectory, invoke _finalize_assignment with its seen_actions.
    new_actions = []
    for traj in trajectories:
        assignment, repeated = trainer._finalize_assignment(fixed_tokens, traj.seen_actions)
        new_actions.append(assignment)
        # Expect repeated==True because it should be a duplicate of "first_actions".
    
    # Check that for each trajectory the new action is different from the first.
    for first, new in zip(first_actions, new_actions):
        assert first != new, "Fallback did not produce a unique assignment when duplicate was detected."

def test_repeated_count_increases_on_duplicate():
    """
    Test that if a duplicate assignment is generated, the trajectory's repeated_count attribute is incremented.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 3
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    trainer = ExpertIterationTrainer(actor_model=model, critic_model=model, tokenizer=tokenizer, num_vars=num_vars)
    
    # Generate one trajectory.
    trajectories = trainer.generate_trajectories_batch(formula, batch_size=1, max_steps=num_vars)
    traj = trajectories[0]
    initial_count = getattr(traj, "repeated_count", 0)
    
    # Simulate generation of a duplicate assignment.
    fixed_tokens = torch.tensor([tokenizer.encode("T", add_special_tokens=False)[0] for _ in range(num_vars)], device=trainer.device)
    _, repeated = trainer._finalize_assignment(fixed_tokens, traj.seen_actions)
    if repeated:
        traj.repeated_count = initial_count + 1
    else:
        traj.repeated_count = initial_count
    assert traj.repeated_count == initial_count + 1, "Repeated count should increase on duplicate detection."

def test_inference_batch_size():
    """
    Test that generate_trajectories_batch returns the expected number of trajectories
    and initializes trajectory attributes properly.
    """
    num_vars = 4
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    trainer = ExpertIterationTrainer(
        actor_model=model,
        critic_model=model,
        tokenizer=tokenizer,
        num_vars=num_vars,
        device=torch.device("cuda")
    )
    formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
    inference_batch_size = 5
    max_steps = 2 ** num_vars  # For 2 variables, max_steps = 4
    trajectories = trainer.generate_trajectories_batch(formula, batch_size=inference_batch_size, max_steps=max_steps)
    assert len(trajectories) == inference_batch_size, (
        f"Expected {inference_batch_size} trajectories, got {len(trajectories)}"
    )

def test_training_batch_size():
    """Test batched training memory usage with Meta-Llama-3.1-8B"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping memory test")

    # Initialize model and trainer with conservative settings
    num_vars = 4
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    trainer = ExpertIterationTrainer(
        actor_model=model,
        critic_model=model,
        tokenizer=tokenizer,
        num_vars=num_vars,
        device=torch.device("cuda"),
        sd_factor=1.0
    )

    # Create template trajectory with UNMASKED tokens for training
    template_traj = Trajectory(
        # Valid assignment sequence that includes unmasked tokens
        actions=[[True]*num_vars, [False]*num_vars],  # Simple pattern
        observations=[True, False],
        rewards=[0.9, 0.8]
    )
    # Add context that will contain unmasked tokens
    template_traj.observations = [True, False] * 8  # 16 observations to match 2^4
    template_traj.rewards = [0.8 + (i*0.01) for i in range(16)]
    template_traj.avg_reward = sum(template_traj.rewards)/len(template_traj.rewards)
    template_traj.successful = True
    
    # Create batch
    batch_size = 2
    trajectories = [deepcopy(template_traj) for _ in range(batch_size)]

    # Test training
    #try:
    trainer.train_on_trajectories(trajectories)

def test_repeated_training_same_formula():
    """Test that training on the same formula multiple times improves performance."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 4
    trainer = ExpertIterationTrainer(
        actor_model=model,
        critic_model=model,
        tokenizer=tokenizer,
        num_vars=num_vars,
        sd_factor=1.0,
        lr=1e-3
    )
    
    # Configure batch sizes and print interval
    training_batch_size = 64
    inference_batch_size = 128
    print_interval = 10
    
    # Create a simple linear formula
    formula = LinearFormula(num_vars=num_vars)
    initial_rewards = []
    final_rewards = []
    
    # Train for 20 episodes on the same formula
    num_episodes = 100 
    for episode in range(num_episodes):
        # Generate trajectories with inference batch size
        trajectories = trainer.generate_trajectories_batch(formula, batch_size=inference_batch_size, max_steps=2**num_vars)
        # Compute rewards and update stats
        rewards_tensor = trainer.batched_model_reward(trajectories)
        for i, traj in enumerate(trajectories):
            traj.avg_reward = rewards_tensor[i].mean().item()
            traj.successful = traj.avg_reward >= (trainer.stats.mean + trainer.sd_factor * trainer.stats.std)
            trainer.update_stats(traj.avg_reward)
        
        # Accumulate successful trajectories
        successful_trajs = [traj for traj in trajectories if traj.successful]
        while len(successful_trajs) >= training_batch_size:
            trainer.train_on_trajectories(successful_trajs[:training_batch_size])
            successful_trajs = successful_trajs[training_batch_size:]
        
        # Track initial and final rewards
        if episode == 0:
            initial_rewards.extend([t.avg_reward for t in trajectories])
        if episode == num_episodes - 1:
            final_rewards.extend([t.avg_reward for t in trajectories])
        
        # Simulate print interval
        if episode % print_interval == 0:
            print(f"Episode {episode} - Avg Reward: {np.mean([t.avg_reward for t in trajectories]):.3f}")
    
    # Verify performance improves
    assert np.mean(final_rewards) > np.mean(initial_rewards), (
        "Final rewards should be higher than initial rewards"
    )
    
    # Verify training ratio increases as model learns
    training_ratios = [r for r in trainer.stats.get_history()['training_ratio'] if r is not None]
    assert np.mean(training_ratios[-3:]) > np.mean(training_ratios[:3]), (
        "Later training ratios should be higher than initial ones"
    )

def test_random_policy_reward_distribution():
    """Verify random policy produces diverse rewards and visualize trajectories."""
    # Set model type: "gpt2" or "llama"
    model_name = "llama"  # Change to "llama" for Meta-Llama model.
    if model_name.lower() == "llama":
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        num_trajectories = 100 
        reward_compute_batch_size = 4
    else:
        model_id = model_name
        num_trajectories = 10000
        reward_compute_batch_size = 500

    num_vars = 6
    max_steps = 2 ** num_vars
    
    # Initialize model components
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", offload_folder="offload")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    trainer = ExpertIterationTrainer(
        actor_model=model,
        critic_model=model,
        tokenizer=tokenizer,
        num_vars=num_vars
    )
    
    # Create simple linear formula
    formula = LinearFormula(num_vars=num_vars)
    
    # Generate trajectories using the batched version of our random unique policy.
    batched_policy = batched_random_unique_policy(num_vars, num_trajectories)
    trajectories = generate_trajectories_batch(
        formula,
        batch_size=num_trajectories,
        max_steps=max_steps,
        batched_policy=batched_policy
    )
    
    # Calculate rewards using batched model reward in smaller mini-batches to reduce memory usage.
    for i in range(0, len(trajectories), reward_compute_batch_size):
        batch = trajectories[i:i+reward_compute_batch_size]
        rewards_tensor = trainer.batched_model_reward(batch)
        for traj, rewards in zip(batch, rewards_tensor):
            traj.rewards = rewards.tolist()
            traj.avg_reward = rewards.mean().item()
    
    # Create plot of quintile averages (pointwise)
    plt.figure(figsize=(12, 6))

    # Sort trajectories by avg_reward
    sorted_trajs = sorted(trajectories, key=lambda t: t.avg_reward)
    n = len(sorted_trajs)
    quintile_size = math.ceil(n / 5)

    # For each quintile, compute the pointwise mean of rewards:
    for i in range(5):
        group = sorted_trajs[i*quintile_size : (i+1)*quintile_size]
        if len(group) == 0:
            continue
        # Assumes all trajectories have the same number of steps.
        rewards_matrix = np.array([traj.rewards for traj in group])
        avg_rewards = rewards_matrix.mean(axis=0)
        steps = np.arange(len(avg_rewards))
        plt.plot(steps, avg_rewards, label=f"Quintile {i+1} (n={len(group)})", linewidth=2)

    plt.xlabel("Step in Trajectory")
    plt.ylabel("Instant Reward")
    plt.title(f"Random Policy Reward Distribution (Pointwise Quintile Averages, {num_trajectories} trajectories)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Compute and plot the optimal policy by averaging over multiple formulas.
    # Instead of generating several optimal trajectories for the same formula,
    # we generate a new formula for each trial.
    num_optimal_trials = 32 
    optimal_trajs = []
    for _ in range(num_optimal_trials):
         # Generate a new formula instance (this could also be random if desired)
         new_formula = LinearFormula(num_vars=num_vars)
         optimal_trajs.append(trainer.optimal_trajectory(new_formula, num_vars))
    # Calculate rewards for the optimal trajectories in mini-batches using reward_compute_batch_size.
    optimal_rewards_list = []
    for i in range(0, len(optimal_trajs), reward_compute_batch_size):
         batch = optimal_trajs[i:i+reward_compute_batch_size]
         rewards_tensor = trainer.batched_model_reward(batch)
         optimal_rewards_list.append(rewards_tensor)
    optimal_rewards_tensor = torch.cat(optimal_rewards_list, dim=0)
    optimal_rewards_avg = optimal_rewards_tensor.mean(dim=0).tolist()
    steps_opt = np.arange(len(optimal_rewards_avg))
    plt.plot(steps_opt, optimal_rewards_avg, label="Optimal Policy (avg over formulas)", 
             linewidth=3, linestyle='--', marker='o', color='black')

    plot_path = "random_policy_rewards_prompt.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def test_batched_random_unique_policy_output():
    from src.curriculum import batched_random_unique_policy, Trajectory
    num_vars = 3
    batch_size = 5
    batched_policy = batched_random_unique_policy(num_vars, batch_size)
    # Create a batch of empty trajectories:
    trajectories = [Trajectory(observations=[], actions=[], rewards=None) for _ in range(batch_size)]
    actions_batch = batched_policy(trajectories)
    assert len(actions_batch) == batch_size
    for action in actions_batch:
        assert isinstance(action, list)
        assert len(action) == num_vars

def test_generate_trajectories_batch():
    from src.curriculum import generate_trajectories_batch, batched_random_unique_policy
    # Use a very simple dummy formula for testing.
    class DummyFormula:
        def __init__(self, num_vars):
            self.num_vars = num_vars
            self.a = [None] * num_vars  # Dummy attribute for compatibility
        def evaluate(self, assignment):
            # Return True if at least one True present
            return any(assignment)
    num_vars = 3
    batch_size = 4
    max_steps = 7
    formula = DummyFormula(num_vars)
    batched_policy = batched_random_unique_policy(num_vars, batch_size)
    trajectories = generate_trajectories_batch(formula, batch_size, max_steps, batched_policy)
    assert len(trajectories) == batch_size
    for traj in trajectories:
        assert len(traj.actions) == max_steps
        assert len(traj.observations) == max_steps
        # Verify that the observation equals what DummyFormula.evaluate would produce.
        for action, obs in zip(traj.actions, traj.observations):
            assert obs == any(action)
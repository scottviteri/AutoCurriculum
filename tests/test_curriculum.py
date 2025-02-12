import pytest
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from src.curriculum import (
    BooleanFormula,  # legacy interface (not used for training)
    RandomTableFormula,
    LinearFormula,
    generate_boolean_function,
    generate_all_assignments,
    generate_trajectory,
    format_trajectory_prompt,
    encode_assignment,
    decode_assignment,
    enumerative_policy,
    model_policy,
    constant_reward,
    model_reward,
    ExpertIterationTrainer,
    Trajectory,
    RunningStats,
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
    Generate a batch of trajectories, then iteratively sample new assignment tokens using
    a mask to allow only "T" or "F" tokens. Verify that the final generated assignment (of length num_vars)
    is in the expected bracketed format, e.g. "[T,F,T,T]" for num_vars == 4.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    num_vars = 4  # Example: using 4 variables.
    
    # Create a batch of 3 trajectories using enumerative_policy and a linear formula.
    trajectories = []
    for _ in range(3):
        formula = generate_boolean_function(num_vars=num_vars, gen_mode="linear")
        policy = enumerative_policy(num_vars)
        # Generate one-step trajectory.
        trajectory = generate_trajectory(formula, policy, max_steps=1, gen_mode="linear")
        trajectories.append(trajectory)
    
    # Build a batch of prompts. Each prompt is the formatted trajectory plus a generation instruction.
    prompts = []
    for traj in trajectories:
        prompt = format_trajectory_prompt(traj)
        full_prompt = (
            prompt +
            "\nPlease choose a new assignment that isn't in the list above.\nNew assignment: ["
        )
        prompts.append(full_prompt)
    
    # Batch encode the prompts with padding.
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    # Allowed tokens: only "T" or "F".
    t_token = tokenizer.encode("T", add_special_tokens=False)[0]
    f_token = tokenizer.encode("F", add_special_tokens=False)[0]
    allowed_ids = [t_token, f_token]
    
    temperature = 1.0
    generated_tokens = []  # Will collect one token per generation step for each sample.
    
    # Iteratively sample num_vars tokens, one per variable.
    for i in range(num_vars):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Determine the index of the last non-padded token for each sample.
        lengths = attention_mask.sum(dim=1) - 1  # shape: (batch_size,)
        last_logits = outputs.logits[torch.arange(input_ids.size(0)), lengths, :]
        
        # Create mask that sets all logits to -inf except for allowed_ids.
        mask = torch.full_like(last_logits, float('-inf'))
        mask[:, allowed_ids] = 0
        masked_logits = last_logits + mask
        scaled_logits = masked_logits / temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
        generated_tokens.append(sampled)
        
        # Append the sampled token to input_ids.
        input_ids = torch.cat([input_ids, sampled], dim=1)
        extra_mask = torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, extra_mask], dim=1)
    
    # Now, for each sample in the batch, extract the newly generated tokens (last num_vars tokens)
    # and form the assignment string.
    new_assignments = []
    for i in range(input_ids.size(0)):
        new_tokens = input_ids[i, -num_vars:]
        token_strs = [tokenizer.decode(tok).strip() for tok in new_tokens]
        assignment_str = "[" + ",".join(token_strs) + "]"
        new_assignments.append(assignment_str)
        # Verify that each token is either "T" or "F".
        for token in token_strs:
            assert token in ["T", "F"], f"Token '{token}' not in allowed set ('T', 'F')"
    
    # Optionally print the generated assignments for manual inspection.
    for assignment_str in new_assignments:
        print("Generated assignment:", assignment_str)
    
    # Verify that each assignment string matches the format using a regex.
    pattern = r"^\[(?:T|F)(?:,\s?(?:T|F)){" + f"{num_vars - 1}" + r"}\]$"
    for assignment_str in new_assignments:
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

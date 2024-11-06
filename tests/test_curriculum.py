import pytest
from src.curriculum import (
    BooleanFormula,
    generate_random_formula,
    generate_trajectory,
    format_trajectory_prompt,
    encode_assignment,
    decode_assignment,
    get_model_action,
    calculate_assignment_loss,
    enumerative_policy,
    model_policy,
    constant_reward,
    model_reward,
    ExpertIterationTrainer,
    Trajectory,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy


def test_boolean_formula_evaluation():
    # Test simple formulas
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "y")]),
        ],
    )

    assert formula.evaluate({"x": True, "y": False}) == True
    assert formula.evaluate({"x": True, "y": True}) == False
    assert formula.evaluate({"x": False, "y": False}) == False


def test_random_formula_generation():
    formula = generate_random_formula(num_vars=2, max_depth=3)

    # Test all possible assignments
    assignments = [{"x0": a, "x1": b} for a in [True, False] for b in [True, False]]

    # Verify formula can be evaluated for all assignments
    for assignment in assignments:
        result = formula.evaluate(assignment)
        assert isinstance(result, bool)


def test_formula_evaluation_with_extra_variables():
    # Create a simple formula using only variables x and y
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "y")]),
        ],
    )

    # Test with additional variables z and w in the assignment
    assignment = {
        "x": True,
        "y": False,
        "z": True,  # Extra variable
        "w": False,  # Extra variable
    }

    # Should work fine and ignore the extra variables
    assert formula.evaluate(assignment) == True

    # Test with a random formula
    random_formula = generate_random_formula(num_vars=2, max_depth=3)  # Uses x0, x1

    # Create assignment with extra variables
    extended_assignment = {
        "x0": True,
        "x1": False,
        "x2": True,  # Extra variable
        "x3": False,  # Extra variable
        "foo": True,  # Extra variable with different naming
    }

    # Should evaluate without errors
    result = random_formula.evaluate(extended_assignment)
    assert isinstance(result, bool)


def test_trajectory_generation():
    """Test basic trajectory generation without rewards."""
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, max_steps=4)

    # Should have exactly 4 steps
    assert len(trajectory.actions) == 4
    assert len(trajectory.observations) == 4
    assert trajectory.rewards is None

    # Check first step
    first_action = trajectory.actions[0]
    first_result = trajectory.observations[0]
    assert first_action == {"x0": False, "x1": False}
    assert first_result == False


def test_trajectory_prompt_formatting():
    """Test formatting of trajectory into prompt string."""
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, max_steps=2)

    # Test prompt formatting
    prompt = format_trajectory_prompt(trajectory)
    lines = prompt.split("\n")
    assert len(lines) == 2  # Should show both steps

    # Verify format of steps
    for line in lines:
        assert "->" in line
        assert line.startswith("x0=")
        assert "x1=" in line
        assert line.endswith("True") or line.endswith("False")


def test_reward_functions():
    """Test that reward functions return correct length lists of floats."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )
    
    trajectory = generate_trajectory(
        formula, 
        enumerative_policy(num_vars=2), 
        max_steps=2
    )
    
    # Test constant reward
    const_reward_fn = constant_reward()
    const_rewards = const_reward_fn(trajectory)
    assert len(const_rewards) == len(trajectory.observations)
    assert all(r == 0.0 for r in const_rewards)
    
    # Test model reward
    model_reward_fn = model_reward(model, tokenizer)
    model_rewards = model_reward_fn(trajectory)
    assert len(model_rewards) == len(trajectory.observations)
    assert all(isinstance(r, float) for r in model_rewards)
    assert all(0.0 <= r <= 1.0 for r in model_rewards)  # Probabilities should be between 0 and 1


def test_loss_invariant_to_result_formatting():
    """Test that loss calculation ignores tokens after variable assignments."""
    actor_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )
    
    base_trajectory = generate_trajectory(
        formula, 
        enumerative_policy(num_vars=2), 
        max_steps=1
    )
    base_loss = calculate_assignment_loss(actor_model, tokenizer, base_trajectory)
    
    # Modify result formatting
    modified_trajectory = deepcopy(base_trajectory)
    tokens = tokenizer.encode(format_trajectory_prompt(modified_trajectory))
    last_equals_pos = max(i for i, t in enumerate(tokens) if t == 28)
    modified_tokens = tokens[:last_equals_pos + 2]  # Keep up to True/False after last =
    modified_tokens.extend([2] * (len(tokens) - len(modified_tokens)))  # Pad with constant token
    modified_loss = calculate_assignment_loss(actor_model, tokenizer, modified_trajectory)
    
    assert abs(base_loss.item() - modified_loss.item()) < 1e-5, \
        "Loss shouldn't change when modifying result formatting"


def test_loss_sensitive_to_assignments():
    """Test that loss changes when variable assignments change."""
    actor_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )
    
    base_trajectory = generate_trajectory(
        formula, 
        enumerative_policy(num_vars=2), 
        max_steps=1
    )
    base_loss = calculate_assignment_loss(actor_model, tokenizer, base_trajectory)
    
    # Modify last variable assignment
    modified_trajectory = deepcopy(base_trajectory)
    modified_trajectory.actions[-1]["x1"] = not modified_trajectory.actions[-1]["x1"]
    modified_loss = calculate_assignment_loss(actor_model, tokenizer, modified_trajectory)
    
    assert abs(base_loss.item() - modified_loss.item()) > 1e-5, \
        "Loss should change when modifying variable assignments"


def test_expert_iteration_token_patterns():
    """Test that ExpertIterationTrainer maintains correct token patterns."""
    actor_model = AutoModelForCausalLM.from_pretrained("gpt2")
    critic_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    trainer = ExpertIterationTrainer(
        actor_model=actor_model,
        critic_model=critic_model,
        tokenizer=tokenizer,
        num_vars=2,
    )
    
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )
    
    # Generate trajectory and verify token patterns
    trajectory = trainer.generate_and_train(formula, max_steps=2)
    
    prompt = format_trajectory_prompt(trajectory)
    tokens = tokenizer.encode(prompt)
    
    equals_positions = [i for i, t in enumerate(tokens) if t == 28]
    arrow_positions = [i for i, t in enumerate(tokens) if t == 4613]
    
    assert len(equals_positions) == len(trajectory.actions) * 2  # 2 variables per action
    assert len(arrow_positions) == len(trajectory.observations)
    
    # Verify alternating pattern is maintained throughout training
    for step_idx in range(len(trajectory.observations)):
        step_equals = equals_positions[step_idx * 2:(step_idx + 1) * 2]
        step_arrow = arrow_positions[step_idx]
        assert all(eq < step_arrow for eq in step_equals)

def test_assignment_encoding_decoding():
    """Test encoding and decoding of variable assignments."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assignment = {"x0": True, "x1": False}

    # Test encoding
    tokens = encode_assignment(assignment, tokenizer)
    assert isinstance(tokens, torch.Tensor)

    # Test decoding
    decoded = decode_assignment(tokens, tokenizer, num_vars=2)
    assert decoded == assignment

def test_model_policy():
    """Test that model policy generates valid assignments."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = model_policy(model, tokenizer, num_vars=2)
    trajectory = generate_trajectory(formula, policy, max_steps=2)

    # Verify structure of actions
    for action in trajectory.actions:
        assert isinstance(action, dict)
        assert set(action.keys()) == {"x0", "x1"}
        assert all(isinstance(v, bool) for v in action.values())

def test_expert_iteration():
    """Test that ExpertIterationTrainer properly updates statistics."""
    actor_model = AutoModelForCausalLM.from_pretrained("gpt2")
    critic_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    trainer = ExpertIterationTrainer(
        actor_model=actor_model,
        critic_model=critic_model,
        tokenizer=tokenizer,
        num_vars=2,
    )

    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    # Generate several trajectories
    trajectories = []
    for _ in range(5):  # Reduced from 10 to speed up tests
        trajectory = trainer.generate_and_train(formula, max_steps=2)
        trajectories.append(trajectory)

    # Verify statistics are being updated
    assert trainer.stats.n == 5
    assert trainer.stats.mean != 0.0  # Should have been updated
    assert trainer.stats.std != float('inf')  # Should have been updated after first trajectory

def test_tokenization_equals():
    """Test that "=" is assigned its own token for proper alignment."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test single variable assignment
    text = "x0=True"
    tokens = tokenizer.encode(text)
    
    # Find position of "="
    equals_positions = [i for i, t in enumerate(tokens) if t == 28]
    assert len(equals_positions) == 1, "Expected exactly one '=' token"
    assert tokenizer.decode([tokens[equals_positions[0]]]) == "=", "Token 28 should decode to '='"

    # Test multiple assignments
    text = "x0=True, x1=False"
    tokens = tokenizer.encode(text)
    equals_positions = [i for i, t in enumerate(tokens) if t == 28]
    assert len(equals_positions) == 2, "Expected exactly two '=' tokens"

    # Verify True/False tokens follow equals
    for pos in equals_positions:
        next_token = tokens[pos + 1]
        decoded = tokenizer.decode([next_token])
        assert decoded.strip() in ["True", "False"], f"Expected True/False after =, got {decoded}"

def test_tokenization_patterns_complex():
    """Test tokenization patterns with more complex formulas."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Test with 3 variables
    formula = generate_random_formula(num_vars=3, max_depth=3)
    policy = enumerative_policy(num_vars=3)
    trajectory = generate_trajectory(formula, policy, max_steps=2)
    
    prompt = format_trajectory_prompt(trajectory)
    tokens = tokenizer.encode(prompt)
    
    equals_positions = [i for i, t in enumerate(tokens) if t == 28]
    arrow_positions = [i for i, t in enumerate(tokens) if t == 4613]
    
    # Should have 3 '=' tokens per step (one per variable)
    assert len(equals_positions) == len(trajectory.actions) * 3
    assert len(arrow_positions) == len(trajectory.observations)
    
    # Verify token sequence for each step
    for step_idx in range(len(trajectory.observations)):
        step_equals = equals_positions[step_idx * 3:(step_idx + 1) * 3]
        step_arrow = arrow_positions[step_idx]
        
        # All equals should come before arrow
        assert all(eq < step_arrow for eq in step_equals)
        
        # Equals should be properly spaced (for variable assignments)
        assert step_equals[1] > step_equals[0] + 1
        assert step_equals[2] > step_equals[1] + 1

def test_rewards_sensitive_to_results():
    """Test that rewards change when results change but only for the modified step."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_fn = model_reward(model, tokenizer)
    
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )
    
    base_trajectory = generate_trajectory(
        formula, 
        enumerative_policy(num_vars=2), 
        max_steps=2
    )
    base_rewards = reward_fn(base_trajectory)
    
    # Modify last result
    modified_trajectory = deepcopy(base_trajectory)
    modified_trajectory.observations[-1] = not modified_trajectory.observations[-1]
    modified_rewards = reward_fn(modified_trajectory)
    
    # Check that only the last reward changed
    assert len(modified_rewards) == len(base_rewards)
    assert all(mr == br for mr, br in zip(modified_rewards[:-1], base_rewards[:-1])), \
        "Earlier rewards shouldn't change when modifying last result"
    assert abs(modified_rewards[-1] - base_rewards[-1]) > 1e-5, \
        "Last reward should change when modifying last result"

def test_rewards_invariant_to_formatting():
    """Test that rewards are invariant to formatting changes after the result."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_fn = model_reward(model, tokenizer)
    
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )
    
    base_trajectory = generate_trajectory(
        formula, 
        enumerative_policy(num_vars=2), 
        max_steps=2
    )
    base_rewards = reward_fn(base_trajectory)
    
    # Add an extra observation without changing results
    modified_trajectory = deepcopy(base_trajectory)
    modified_trajectory.actions.append({"x0": True, "x1": True})  # Add action without observation
    modified_rewards = reward_fn(modified_trajectory)
    
    # Check that rewards are unchanged
    assert len(modified_rewards) == len(base_rewards)
    assert all(abs(mr - br) < 1e-5 for mr, br in zip(modified_rewards, base_rewards)), \
        "Rewards shouldn't change when adding incomplete steps"

import pytest
from src.curriculum import (
    BooleanFormula,
    generate_random_formula,
    generate_trajectory,
    format_trajectory_prompt,
    encode_assignment,
    decode_assignment,
    get_model_action,
    calculate_prediction_reward,
    enumerative_policy,
    model_policy,
    RewardType,
    constant_reward,
    model_reward,
    ExpertIterationTrainer,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, reward_fn=constant_reward)

    # Should have 2^2 = 4 steps
    assert len(trajectory.steps) == 4

    # Check first step
    first_step = trajectory.steps[0]
    assert first_step.action == {"x0": False, "x1": False}
    assert first_step.observation.result == False
    assert first_step.observation.reward == 1.0


def test_trajectory_prompt_formatting():
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, reward_fn=constant_reward)

    # Test prompt formatting
    prompt = format_trajectory_prompt(trajectory)
    lines = prompt.split("\n")
    assert len(lines) == 4  # Should show all 4 steps

    # Verify format of steps
    for line in lines:
        assert "→" in line
        assert line.startswith("x0=")
        assert "x1=" in line
        assert line.endswith("True") or line.endswith("False")


def test_assignment_encoding_decoding():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assignment = {"x0": True, "x1": False}

    # Test encoding
    tokens = encode_assignment(assignment, tokenizer)
    assert isinstance(tokens, torch.Tensor)

    # Test decoding
    decoded = decode_assignment(tokens, tokenizer, num_vars=2)
    assert decoded == assignment


def test_model_action_generation():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, reward_fn=constant_reward)

    # Test action generation
    assignment, output = get_model_action(
        model, tokenizer, num_vars=2, prompt=format_trajectory_prompt(trajectory)
    )

    # Verify structure
    assert isinstance(assignment, dict)
    assert len(assignment) == 2
    assert set(assignment.keys()) == {"x0", "x1"}
    assert all(isinstance(v, bool) for v in assignment.values())

    # Verify token properties
    assert isinstance(output.action_tokens, torch.Tensor)
    assert len(output.action_tokens) == 2  # One token per variable
    assert all(
        t.item() in [17821, 25101] for t in output.action_tokens
    )  # Only True/False tokens

    # Verify logprobs
    assert isinstance(output.action_logprobs, torch.Tensor)
    assert len(output.action_logprobs) == 2  # One logprob per variable


def test_prediction_reward():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, reward_fn=constant_reward)

    # Test reward calculation
    reward = calculate_prediction_reward(model, tokenizer, trajectory)
    assert isinstance(reward, torch.Tensor)
    assert reward.ndim == 0  # Scalar tensor
    assert reward.item() <= 0.0  # Log probability should be non-positive


def test_enumerative_policy():
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, reward_fn=constant_reward)

    # Should have 2^2 = 4 steps
    assert len(trajectory.steps) == 4

    # Verify we got all possible assignments
    assignments = [step.action for step in trajectory.steps]
    assert {"x0": False, "x1": False} in assignments
    assert {"x0": False, "x1": True} in assignments
    assert {"x0": True, "x1": False} in assignments
    assert {"x0": True, "x1": True} in assignments


def test_model_policy():
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
    trajectory = generate_trajectory(
        formula, policy, max_steps=4, reward_fn=constant_reward
    )

    # Verify trajectory structure
    assert len(trajectory.steps) <= 4
    for step in trajectory.steps:
        assert set(step.action.keys()) == {"x0", "x1"}
        assert isinstance(step.observation.result, bool)


def test_enumerative_trajectory():
    formula = BooleanFormula(
        "AND",
        [
            BooleanFormula("VAR", "x0"),
            BooleanFormula("NOT", [BooleanFormula("VAR", "x1")]),
        ],
    )

    policy = enumerative_policy(num_vars=2)
    trajectory = generate_trajectory(formula, policy, reward_fn=constant_reward)

    # Should have 2^2 = 4 steps
    assert len(trajectory.steps) == 4
    # All rewards should be 1.0
    assert all(step.observation.reward == 1.0 for step in trajectory.steps)


def test_model_trajectory():
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
    reward_fn = model_reward(model, tokenizer)
    trajectory = generate_trajectory(formula, policy, reward_fn=reward_fn, max_steps=4)

    # Verify trajectory structure
    assert len(trajectory.steps) <= 4
    for step in trajectory.steps:
        assert set(step.action.keys()) == {"x0", "x1"}
        assert isinstance(step.observation.result, bool)
        assert step.observation.reward <= 0.0  # Log probability should be non-positive


def test_expert_iteration():
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
    for _ in range(10):
        trajectory = trainer.generate_and_train(formula, max_steps=4)
        trajectories.append(trajectory)

    # Verify statistics are being updated
    assert trainer.stats.count == 10
    assert trainer.stats.mean != 0.0  # Should have been updated
    assert trainer.stats.std != 1.0  # Should have been updated


def test_tokenization_equals():
    """Test that "=" is assigned its own token for proper alignment."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test single variable assignment
    text = "x0=True"
    tokens = tokenizer.encode(text)
    decoded = [tokenizer.decode([t]) for t in tokens]

    # Find position of "="
    equals_positions = [i for i, t in enumerate(tokens) if t == 28]
    assert len(equals_positions) == 1, "Expected exactly one '=' token"
    assert (
        tokenizer.decode([tokens[equals_positions[0]]]) == "="
    ), "Token 28 should decode to '='"

    ## Verify next token is " True"
    # true_token = tokens[equals_positions[0] + 1]
    # assert tokenizer.decode([true_token]) == " True"

    # Test multiple assignments in trajectory format
    text = "x0=True, x1=False"
    tokens = tokenizer.encode(text)

    # Find all "=" positions
    # equals_positions = [i for i, t in enumerate(tokens) if t == 28]
    # assert len(equals_positions) == 2, "Expected exactly two '=' tokens"

    ## Verify each "=" is followed by True/False
    # for pos in equals_positions:
    #    assert tokens[pos] == 28, "Equals token should be 28"
    #    next_token = tokens[pos + 1]
    #    decoded_token = tokenizer.decode([next_token])
    #    assert decoded_token in [" True", " False"]

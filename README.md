# AutoCurriculum

A framework for using Reinforcement Learning (RL) to enable a language model to select its own training data order for solving boolean formula problems.

## Overview

AutoCurriculum implements an expert-iteration training strategy to teach language models (LMs) how to choose the order in which they see their training examples. Instead of passively receiving data, the LM actively selects assignments to query and then receives feedback based on its selected strategy. This approach demonstrates that Reinforcement Learning can be applied to allow an LM to determine its own curriculum.

### Why Linear Formulas?

We use linear boolean formulas as a proof of concept because they reveal a significant gap between random and optimal data orders:

- **Linear Boolean Formulas:**  
  A linear boolean formula has the form:
  ```
  F(x₁, ..., xₙ) = (a₁ * x₁) XOR (a₂ * x₂) XOR ... XOR (aₙ * xₙ) XOR b
  ```
- **Optimal Querying Strategy:**  
  For optimal learning:
  1. The LM should first choose an assignment with all variables set to `False` to learn the bias element (`b`).
  2. Next, it should sequentially set the i-th variable to `True` (with the others remaining `False`) to learn each coefficient (`aᵢ`).

  This optimal order requires only `(n+1)` queries, as opposed to the many more queries that might be required with a random order. By learning this strategy through RL, the LM demonstrates that it can self-direct the order of its training data to achieve faster and more efficient learning.

### The Training Process

1. **Boolean Formula Generation:**  
   The system generates boolean formulas (using either random or linear modes) on which the LM is trained.
2. **Assignment Proposal:**  
   The LM proposes variable assignments based on a policy (which uses RL) to decide the next query.
3. **Evaluation and Reward Calculation:**  
   Each assignment is evaluated against the formula. A reward is computed based on the effectiveness of the proposal.
4. **Curriculum Learning:**  
   Using a dynamic threshold based on running statistics (mean and std), the system selects promising trajectories for training, updating the LM only when the average reward surpasses this threshold.
5. **Training Update:**  
   The LM is updated using only the selected strategies, reinforcing the optimal sequence of queries.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AutoCurriculum.git
cd AutoCurriculum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train the language model to autonomously decide its training data order:
```bash
python src/train.py
```

Common options:
```bash
python src/train.py --num-vars 3 --max-depth 4 --num-episodes 1000
```

Key parameters:
- `num-vars`: Number of variables in formulas (default: 2)
- `max-depth`: Maximum depth of generated formulas (default: 3)
- `num-episodes`: Number of training episodes (default: 1000)
- `print-interval`: Episodes between progress updates (automatically set based on num-vars)

The maximum steps per training trajectory is automatically set to 2^(num-vars).

### Visualization

Monitor training progress:
```bash
python src/plot_training.py
```

Analyze trajectory rewards:
```bash
python src/plot_trajectories.py
```

Both plotting scripts can be directed to analyze specific runs:
```bash
python src/plot_training.py --run-dir runs/YYYYMMDD_HHMMSS
```

## Project Structure

```
AutoCurriculum/
├── src/
│   ├── curriculum.py      # Core training logic and production of trajectories
│   ├── train.py           # Training script that applies RL for selecting training data order
│   ├── plot_training.py   # Visualize training progress
│   └── plot_trajectories.py  # Visualize trajectory reward analysis
├── tests/
│   └── test_curriculum.py # Unit tests
├── runs/                  # Training run outputs
│   └── YYYYMMDD_HHMMSS/   # Timestamped run directories
├── requirements.txt
└── README.md
```

## Output

Each training run generates a timestamped directory containing:
- `config.json`: Run configuration
- `trajectories.jsonl`: Detailed trajectory data
- `stats.jsonl`: Training statistics
- `training_progress.png`: Visualization of training metrics
- `trajectory_rewards.png`: Analysis of trajectory rewards

## Monitoring

The system provides several monitoring features:
1. **Console Updates:**  
   Displays the recent mean reward, reward standard deviation, global training threshold, and training ratio.
2. **Automatically Generated Plots:**  
   Visualizations of reward progression, training ratio, and trajectory analysis help to monitor the curriculum learning process.

## License

© 2024 All Rights Reserved.

This code is released for academic and non-commercial use only. For commercial use or redistribution, please contact the authors.

*Note:* This is preliminary research code intended to demonstrate that Reinforcement Learning can be used for an LM to autonomously determine the order in which it sees training data, using linear boolean formulas as an illustrative example.
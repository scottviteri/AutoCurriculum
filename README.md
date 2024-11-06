# AutoCurriculum

A framework for training language models to solve boolean formula problems using automatic curriculum learning.

## Overview

This project implements Expert Iteration (ExIt) to train language models on boolean formula evaluation. The system:
1. Generates random boolean formulas
2. Uses the model to generate variable assignments
3. Evaluates the correctness of these assignments
4. Trains on successful trajectories that exceed a dynamic threshold

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

Train a model with default settings:
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

The maximum steps per trajectory is automatically set to 2^num_vars.

### Visualization

Monitor training progress:
```bash
python src/plot_training.py
```

View trajectory rewards:
```bash
python src/plot_trajectories.py
```

Both plotting scripts can analyze specific runs:
```bash
python src/plot_training.py --run-dir runs/YYYYMMDD_HHMMSS
```

## Project Structure

```
AutoCurriculum/
├── src/
│   ├── curriculum.py      # Core training logic and data structures
│   ├── train.py          # Training script
│   ├── plot_training.py  # Training progress visualization
│   └── plot_trajectories.py  # Trajectory analysis visualization
├── tests/
│   └── test_curriculum.py # Unit tests
├── runs/                  # Training run outputs
│   └── YYYYMMDD_HHMMSS/  # Timestamped run directories
├── requirements.txt
└── README.md
```

## Output

Each training run creates a timestamped directory containing:
- `config.json`: Run configuration
- `trajectories.jsonl`: Detailed trajectory data
- `stats.jsonl`: Training statistics
- `training_progress.png`: Training metrics visualization
- `trajectory_rewards.png`: Trajectory reward analysis

## Monitoring

The system provides several ways to monitor training:
1. Regular console updates showing:
   - Recent mean reward
   - Recent reward standard deviation
   - Global training threshold
   - Training ratio
2. Automatically generated plots showing:
   - Reward progression
   - Training ratio
   - Trajectory analysis

## License

Copyright (c) 2024. All Rights Reserved.

This code is released for academic and non-commercial use only. For commercial use or redistribution, please contact the authors.

Note: This is preliminary research code that will be released under a more permissive license upon publication.
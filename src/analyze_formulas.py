import argparse
import math
import random
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm

def binary_entropy(p: float) -> float:
    """
    Binary entropy in bits for a probability p of True, (1-p) of False.
    Returns 0 if p=0 or p=1, and maximum 1 at p=0.5.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def make_random_boolean_function(n: int):
    """
    For n variables, produce a random boolean function by specifying
    its truth values for all 2^n inputs.
    
    Returns:
        A list of length 2^n, where each entry is True/False for
        the corresponding integer input 0..(2^n - 1).
    """
    num_inputs = 2 ** n
    # For each possible input, pick True/False with 50% probability
    table = [random.choice([False, True]) for _ in range(num_inputs)]
    return table

def make_linear_boolean_function(n: int):
    """
    For n variables, produce a linear boolean function of the form:
        f(x1,..,xn) = (a1*x1) ⊕ ... ⊕ (an*xn) ⊕ b
    where ai, b ∈ {0,1} drawn at random.
    
    We'll store the truth table of size 2^n in a list of booleans.
    """
    # Random bits a1..an, b
    a = [random.randint(0,1) for _ in range(n)]
    b = random.randint(0,1)
    num_inputs = 2 ** n
    
    table = []
    for x in range(num_inputs):
        # Evaluate the bits of x
        # x_j is the j-th variable bit (0 or 1)
        # We'll accumulate a_j * x_j in a variable, then XOR them
        val = b
        for j in range(n):
            x_j = (x >> j) & 1
            if a[j] == 1:
                val = val ^ x_j
        # if val=1 => True, else False
        table.append(bool(val))
    return table

def compute_fraction_and_entropy(booleans: list[bool]) -> tuple[float, float]:
    """
    Given a truth table (list of booleans), compute:
    1) fraction of True
    2) binary entropy of that fraction
    """
    num_inputs = len(booleans)
    true_count = sum(booleans)
    fraction_true = true_count / num_inputs
    ent = binary_entropy(fraction_true)
    return fraction_true, ent

def plot_xy(args):
    """
    For each n in [args.min_vars..args.max_vars], generate 'args.num_formulas'
    boolean functions of the chosen type (random or linear), compute their
    fraction of True and entropy, average them, then plot as an XY line.
    """
    xs = list(range(args.min_vars, args.max_vars + 1))
    frac_means = []
    ent_means = []
    
    for n in xs:
        frac_values = []
        ent_values = []
        
        for _ in tqdm(range(args.num_formulas), desc=f"Generating n={n}"):
            if args.gen_mode == "random":
                table = make_random_boolean_function(n)
            elif args.gen_mode == "linear":
                table = make_linear_boolean_function(n)
            else:
                raise ValueError(f"Unknown gen-mode: {args.gen_mode}")
            
            frac_true, ent = compute_fraction_and_entropy(table)
            frac_values.append(frac_true)
            ent_values.append(ent)
        
        frac_means.append(statistics.mean(frac_values))
        ent_means.append(statistics.mean(ent_values))
    
    # Now plot fraction of True and entropy on two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    
    ax1.plot(xs, frac_means, marker='o', label='Mean Fraction True')
    ax1.set_ylabel('Fraction True')
    ax1.legend()
    
    ax2.plot(xs, ent_means, marker='s', color='orange', label='Mean Entropy')
    ax2.set_ylabel('Binary Entropy (bits)')
    ax2.set_xlabel('Number of variables (n)')
    ax2.legend()
    
    title = f"{args.gen_mode.capitalize()} boolean functions\n(num_formulas={args.num_formulas})"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze random or linear boolean functions.")
    
    parser.add_argument('--gen-mode', type=str, required=True,
                        choices=['random','linear'],
                        help="Choose how to generate boolean functions: 'random' or 'linear'.")
    parser.add_argument('--plot-xy', action='store_true',
                        help="Plot fraction of True and binary entropy vs number of variables.")
    parser.add_argument('--min-vars', type=int, default=1,
                        help="Minimum number of variables to test in the plot.")
    parser.add_argument('--max-vars', type=int, default=5,
                        help="Maximum number of variables to test in the plot.")
    parser.add_argument('--num-formulas', type=int, default=10,
                        help="How many functions per data point to average.")
    
    args = parser.parse_args()
    
    if args.plot_xy:
        plot_xy(args)
    else:
        print("Nothing to do. Use --plot-xy to produce the XY plot.")

if __name__ == '__main__':
    main()
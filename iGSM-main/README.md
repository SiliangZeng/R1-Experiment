# iGSM
Welcome to the iGSM synthetic data generator/evaluator, featured in our research paper
* ["Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process"](https://arxiv.org/abs/2407.20311) and
* ["Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems"](https://arxiv.org/abs/2408.16293).

This code is designed to *generate* grade-school math problems in our designed problem class (see **Part 2.1**), as well as to *evaluate* if any given full solution (not only the decimal answer), when formatted properly, is fully correct.
Although we did not explore in the paper, this code also offers various difficulty level choices and generates ``networkx.DiGraph`` objects for any future use.
The example code to generate/evaluate such iGSM dataset is given in [example_iGSM.ipynb](example_iGSM.ipynb), with detailed documents below. 

We also provide a crisper example code to generate the box-over-box example (see Figure 1 of **Part 2.2**) in [example_box.ipynb](example_box.ipynb), which deserves no further documentation and provides hard reasoning tasks even for GPT-4o.

## Quick Example
Generate iGSM data with varying difficulty using the ``get_prob_sol_ans_triple`` function. Here’s how to use it:
```python
from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from typing import Literal

def get_prob_sol_ans_triple(tpy: Literal["med", "hard"]):
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    # Set parameters based on difficulty
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28

    id_gen = IdGen(
        max_op=max_op,        # Maximum # of operations
        max_edge=max_edge,    # Maximum # of edges (instance parameters) in the structure graph
        perm_level=5,         # Random shuffle level for problem description
        detail_level=0        # Most detailed solution format
    )

    id_gen.gen_prob([i for i in range(23)], p_format="pq")

    return id_gen

# generate 10 med problems
fix_seed(42)
for i in range(10):
    id_gen = get_prob_sol_ans_triple("med")
    print(f"\n\nProblem {i}:")
    print(tokenizer.decode(id_gen.prob_token))
    print(f"Solution:")
    print(tokenizer.decode(id_gen.sol_token))
    print(f"Answer:")
    print(tokenizer.decode(id_gen.ans_token))
```
 - **Input** ``tpy``: Specifies the difficulty level ('med' for medium, 'hard' for hard).
 - **Outputs**: Returns an ``IdGen`` instance.
This function configures and generates token sequences for problem-solving tasks, adjusting complexity based on the specified difficulty level.

## Requirements
iGSM requires or works with Python version 3.8.11 or newer.

# How iGSM works
The ``IdGen`` class from ``data_gen.pretrain.id_gen`` can be used to generate a problem along with its solution. Each instance of ``IdGen`` represents a unique problem-solution pair, containing all relevant decoded information.
```python
from data_gen.pretrain.id_gen import IdGen
from tools.tools import fix_seed

fix_seed(42)  # Ensures reproducibility

# Initialize IdGen with specific parameters
id_gen = IdGen(
    max_op=15,            # Maximum operations
    max_edge=20,          # Maximum edges in the Structure Graph
    perm_level=5,         # Level of randomization in problem description. 5 represents random shuffle
    detail_level=0        # Level of detail in the solution. 0 represents the most detailed solution format
)

# Generate a problem and format it
id_gen.gen_prob([i for i in range(23)], p_format="pq")
```

Upon execution, id_gen creates (but not limited to) the following:
```python
prob_id = id_gen.prob_token  # Problem's token IDs
sol_id = id_gen.sol_token    # Solution's token IDs
ans_id = id_gen.ans_token    # One-number answer's token IDs
```

These tokens can be decoded using GPT2's standard tokenizer:
```python
from tools.tools import tokenizer
print("Problem:")
print(tokenizer.decode(prob_id))
print("Solution:")
print(tokenizer.decode(sol_id))
print("Answer:")
print(tokenizer.decode(ans_id))
```
The expected output is:
```
Problem:
 The number of each Penguin Beach's Giraffe equals 6. The number of each Octopus Den's Leopard equals each Octopus Den's Giraffe. The number of each Rockpool Exhibit's Leopard equals 20 more than the sum of each Octopus Den's Giraffe and each Octopus Den's Leopard. The number of each Rockpool Exhibit's Giraffe equals 8 times as much as the sum of each Octopus Den's Giraffe and each Octopus Den's Leopard. The number of each Octopus Den's Giraffe equals 21. How many Animal does Penguin Beach have?
Solution:
 Define Penguin Beach's Giraffe as e; so e = 6. Define Penguin Beach's Animal as J; so J = e = 6.
Answer:
 6
```

Token ID arrangement for training datasets:
```python
# Assemble token IDs for model training
assert id_gen.token_id == [222] + prob_id + [223] + sol_id + [224] + ans_id + [50256]
```
Where ``222``, ``223``, ``224`` are markers for the beginning of the problem, the transition from problem to solution, and the start of the answer, respectively. ``50256`` serves as a stop token for the GPT2 model.

## Validating Solutions

Use the ``true_correct`` function to verify the correctness of a solution. This function analyzes the solution string, comparing it against the problem parameters to determine its accuracy.
```python
from tools.tools_test import true_correct

# Example of a solution string
wrong_sol = " Define Penguin Beach's Giraffe as t; so t = 6. Define Octopus Den's Leopard as r; so r = t = 6. Define Penguin Beach's Animal as J; so J = t = 6."

# Validate the solution
correct, my_print, parser = true_correct(wrong_sol, id_gen.problem)
print(f"Correct or not: {correct}")
my_print.display()
```
The expected output is:
```
correct or not: False
Mistake happended on param Octopus Den's Leopard
missing_but_required_params: Octopus Den's Giraffe
existing_but_not_required_params: Penguin Beach's Giraffe
```

**Explanation**
 - ``correct``: A boolean indicating the solution's correctness.
 - ``my_print``: An object storing debug information. Use ``my_print.display()`` to visualize this data.
 - ``parser``: An object that parses and holds structural information from the solution. If ``correct`` is ``False``, you will probably get a ``None`` placeholder.
This function assesses not just whether the answer is numerically correct but whether the solution accurately follows the problem's requirements.

**Note**: The ``true_correct`` function needs the ``Problem`` instance (``id_gen.problem``) to function correctly during testing.

## Solutions With Retry
For generating retry data and retry_weak data as described in our paper, use the functions below. The input parameters and outputs for these functions follow the same format as the ``get_prob_sol_ans_triple`` function described earlier.

```python
from data_gen.pretrain.id_retry_gen import IdGen as IdRetryGen
from data_gen.pretrain.id_retry_weak_gen import IdGen as IdRetryWeakGen
from typing import Literal

def get_retry_prob_sol_ans_triple(tpy: Literal["med", "hard"], retry_rate: float):
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    # Set parameters based on difficulty
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28

    id_gen = IdRetryGen(
        max_op=max_op,        # Maximum operations
        max_edge=max_edge,    # Maximum edges in the structure graph
        perm_level=5,         # Random shuffle level for problem description
        detail_level=0,       # Most detailed solution format
        retry_rate=retry_rate,
    )

    # Generate problem, solution, and answer tokens
    id_gen.gen_prob([i for i in range(23)], p_format="pq")
    id_gen.insert_retry()

    return id_gen

def get_retry_weak_prob_sol_ans_triple(tpy: Literal["med", "hard"], retry_rate: float):
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    # Set parameters based on difficulty
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28

    id_gen = IdRetryWeakGen(
        max_op=max_op,        # Maximum # of operations
        max_edge=max_edge,    # Maximum # of edges (instance parameters) in the structure graph
        perm_level=5,         # Random shuffle level for problem description
        detail_level=0,       # Most detailed solution format
        retry_rate=retry_rate,
    )

    # Generate problem, solution, and answer tokens
    id_gen.gen_prob([i for i in range(23)], p_format="pq")
    id_gen.insert_retry()

    return id_gen
```

## Full Documentation

### Problem and Graph Inheritance
When `id_gen.gen_prob()` is invoked, it initializes a `Problem` instance named `id_gen.problem`. The `Problem` class extends the `Graph` class, which is designed to handle the generation and management of specific details relevant to the problem:
- **Graph Class**: Stores structural and dependency graphs that outline the relationships and dependencies among different elements of the problem.
- **Problem Class**: Responsible for generating the names and exact values of parameters, along with crafting descriptive narratives for both the problem and its solution.

### Structure Graph
The structure graph is encoded in `id_gen.problem.G`, stored as a list of NumPy matrices with boolean values. Each entry `id_gen.problem.G[i][j, k]` signifies a connection between the node `(i, j)` and `(i+1, k)`, where `(i, j)` represents the `j`-th node at the `i`-th layer. This matrix helps visualize how nodes are interconnected layer by layer.

### Dependency Graph Nodes
The nodes within the dependency graph are represented by four-integer tuples, `(i, j, k, l)`, with specific meanings based on the value of `i`:
- **RNG Representation (`i = -1`)**: When `i` is -1, `j`, `k`, and `l` must all be 0, making the tuple `(-1, 0, 0, 0)` denote the **Random Number Generator (RNG)** used within the problem context.
- **Instance Parameter (`i = 0`)**: When `i` is 0, the tuple `(i, j, k, l)` identifies an **instance parameter**. It specifically counts the number of Item `(j, k)` in relation to Item `(j+1, k)`, such as counting the number of Music Rooms in Riverview High. The existence of such a parameter depends strictly on the truth of `id_gen.problem.G[j][k, l]`.
- **Abstract Parameter (`i = 1`)**: When `i` is 1, the tuple represents an **abstract parameter**, counting items of Category `k` within Item `(j, k)`, like the number of classrooms in Riverview High. Such parameters are only defined if feasible and if `j < l`.

The dependency graph is instantiated as ``id_gen.problem.template``, a directed graph using the ``networkx.DiGraph`` class.

### Additional Components
- **Value Lookup (`id_gen.problem.lookup`)**: This component is a dictionary mapping from the four-integer tuples to the respective parameter values.
- **Name Lookup (`id_gen.problem.N`)**: The array `id_gen.problem.N[i][j]` holds the name of the Item `(i, j)`.
- **Draw Graphs (`id_gen.problem.draw()`)**: This function will plot the structure graph and the dependency graph.

# Citation

Please cite this code and our iGSM dataset using
```bibtex
  @article{YXLA2024-gsm1,
    author = {Ye, Tian and Xu, Zicheng and Li, Yuanzhi and {Allen-Zhu}, Zeyuan},
    title = {{Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process}},
    journal = {ArXiv e-prints},
    year = 2024,
    month = jul,
    volume = {abs/2407.20311},
    note = {Full version available at \url{http://arxiv.org/abs/2407.20311}}
  }
```

If you plan to use our retry data or the box-over-box data, please also cite our Part 2.2 paper as follows:
```bibtex
  @article{YXLA2024-gsm2,
    author = {Ye, Tian and Xu, Zicheng and Li, Yuanzhi and {Allen-Zhu}, Zeyuan},
    title = {{Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems}},
    journal = {ArXiv e-prints},
    year = 2024,
    month = aug,
    volume = {abs/2408.16293},
    note = {Full version available at \url{http://arxiv.org/abs/2408.16293}}
  }
```

MIT License; please contact Tian Ye or Zeyuan Allen-Zhu if you have any questions.

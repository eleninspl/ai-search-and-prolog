# Maze Search and a Prolog Movie Recommender

Two Jupyter notebooks and two sets of written exercises on classical AI. The first notebook generates random mazes and compares Dijkstra, A\* and greedy best-first search on them, then adds a ghost that chases an alpha-beta agent through the maze. The second builds a Prolog knowledge base from a dataset of 4,803 movies, uses rules over it to find similar movies, and turns those rules into a recommender that is trained and evaluated on one user's ratings.

These are solutions to the assignments of **Artificial Intelligence** (Τεχνητή Νοημοσύνη), a 6th-semester course at the School of Electrical and Computer Engineering, National Technical University of Athens (ECE NTUA), academic year 2023–24 (spring semester 2024).

The two programming assignments were team work with **Georgios Isopoulos** (Γεώργιος Ισόπουλος). The written exercise sets are my own.

| Part | Files | Topic |
|------|-------|-------|
| [Lab 1](#lab-1-maze-search-and-adversarial-search) | `programming/lab1/AI_24_EX1.ipynb` | Maze generation, Dijkstra, A\*, greedy best-first search, alpha-beta |
| [Lab 2](#lab-2-prolog-movie-recommender) | `programming/lab2/AI_24_EX2.ipynb`, `programming/lab2/db.pl` | Prolog knowledge base, rule-based similarity, recommender evaluation |
| [Exercise set 1](#exercise-set-1-search-games-and-genetic-algorithms) | `theoretical/set1/` | Heuristic search, minimax and alpha-beta, genetic algorithms |
| [Exercise set 2](#exercise-set-2-logic-and-resolution) | `theoretical/set2/` | CNF, first-order logic, models, resolution |

The code is Python and Prolog. The notebooks' text, some code comments and the written answers are in Greek.

## Getting started

The notebooks were written for Google Colab, which is the easiest way to run them. They also run locally with a few cells skipped, as described in each lab.

### Prerequisites

- Python 3 with Jupyter
- The packages in [`requirements.txt`](requirements.txt): NumPy, Matplotlib, Pillow, IPython, pandas, scikit-learn, tqdm, ipywidgets and pyswip
- [SWI-Prolog](https://www.swi-prolog.org/) for lab 2 (pyswip needs it installed)

```bash
git clone https://github.com/eleninspl/ai-search-and-prolog.git
cd ai-search-and-prolog
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
```

## Lab 1: Maze search and adversarial search

**Task.** The course provides a skeleton notebook with a maze class, an A\*-style search class and an animation helper. The assignment has five parts:

1. Complete a maze generator so that the density of walls is controlled by a `threshold` parameter and there is always at least one path from start (S) to finish (F).
2. Complete the search so that one function covers several algorithms through the cost function g(n) and heuristic h(n). Identify which algorithm each combination gives and whether it always finds the shortest path.
3. Compare the algorithms on mazes of size N = 10…100, about 100 random mazes per size. Plot path length and number of expanded nodes against N.
4. A ghost chases the player through the maze, running its own A\* on every step. Complete the alpha-beta agent that tries to reach the finish without being caught, and design and compare heuristics for it.
5. (Bonus) Draw five mazes by hand that show specific behaviours, for example a maze where hill climbing fails, or one where A\* and Dijkstra take exactly the same steps.

**How it works.**

- **Maze generation.** Cells sit at odd coordinates of an N×N grid (N must be odd). The generator is a randomized depth-first search with a stack. At each step it picks a random unvisited neighbour two cells away and removes the wall between them. With probability `threshold` it also removes a wall to an already visited cell, which creates loops and so more than one path to the finish.
- **Search.** A single `pathfinder` class runs a best-first search with a priority queue ordered by g(n) + h(n). The cost function and the heuristic are arguments:

  | g step cost | h(n) | Algorithm | Shortest path guaranteed? |
  |---|---|---|---|
  | 1 | 0 | Dijkstra (uniform-cost search) | Yes |
  | 1 | Manhattan or Euclidean distance | A\* | Yes, both are admissible on a 4-connected grid |
  | 0 | Manhattan or Euclidean distance | Greedy best-first search | No |

- **Comparison.** For each odd N from 11 to 101, the notebook builds 100 random mazes with S = (1, 1) and F = (N−2, N−2) and runs all five variants on each one. Averages from the saved run (path length counts cells, including start and finish):

  | Algorithm | Path length, N=51 | Path length, N=101 | Expanded nodes, N=51 | Expanded nodes, N=101 |
  |---|---|---|---|---|
  | Greedy best-first, Euclidean | 230.9 | 483.0 | 345 | 797 |
  | Greedy best-first, Manhattan | 229.0 | 479.2 | 343 | 785 |
  | A\*, Euclidean | 196.7 | 369.6 | 838 | 3,923 |
  | A\*, Manhattan | 196.7 | 369.6 | 780 | 3,579 |
  | Dijkstra | 196.7 | 369.6 | 1,032 | 4,781 |

  A\* and Dijkstra always return paths of the same length. Manhattan distance is the tighter heuristic on a grid, so A\* with Manhattan expands the fewest nodes of the optimal methods. Greedy best-first search expands far fewer nodes but returns paths about 30% longer on the largest mazes.

  Plots from one run of the experiment (A = greedy Euclidean, B = greedy Manhattan, C = A\* Euclidean, D = A\* Manhattan, E = Dijkstra):

  | Path length | Expanded nodes |
  |---|---|
  | ![Average path length against maze size for the five algorithms](programming/lab1/Gifs%20Diagrams/path_length.png) | ![Average number of expanded nodes against maze size for the five algorithms](programming/lab1/Gifs%20Diagrams/expanded_nodes.png) |

  Animations of each algorithm on the same 31×31 maze are in [`programming/lab1/Gifs Diagrams/`](programming/lab1/Gifs%20Diagrams). On that maze, Dijkstra expanded 429 nodes, A\* 376 (Euclidean) and 369 (Manhattan), and greedy best-first about 255, with a path of 149–153 cells instead of 117.

  ![A* with the Manhattan heuristic searching a 31×31 maze and drawing the shortest path](programming/lab1/Gifs%20Diagrams/A__manhattan.gif)

- **Ghost chase.** The maze is generated sparse (`threshold=0.1`) so that there are several routes to the finish. The ghost recomputes an A\* path to the player and takes one step each turn. The player runs alpha-beta search to depth 5: the player is the maximizing side and the ghost's moves are the minimizing side. Three evaluation functions are defined:
  1. 0.5 × Manhattan distance to the ghost − Manhattan distance to the finish
  2. The same with Euclidean distances
  3. 0.7 × Euclidean distance to the ghost − length of a greedy best-first path to the finish

  The notebook concludes that the third function wins most often and is the slowest, because it runs a search at every leaf. Animations are in the same folder ([`heuristic1.gif`](programming/lab1/Gifs%20Diagrams/heuristic1.gif), [`heuristic2.gif`](programming/lab1/Gifs%20Diagrams/heuristic2.gif), [`heuristic3.gif`](programming/lab1/Gifs%20Diagrams/heuristic3.gif)). See [Known limitations](#known-limitations) before relying on this comparison.

- **Hand-drawn mazes (bonus).** A canvas in the notebook lets you draw a 15×15 maze, which is converted into a `Maze`. The notebook contains the five drawings, embedded as images, with a short explanation of each.

**Run.** In Colab, open the notebook and run all cells. Locally, start `jupyter notebook programming/lab1/AI_24_EX1.ipynb` and run parts 1–4. Skip the cell in part 3 that mounts Google Drive to save the GIFs, or change its `path` to a local folder. Part 5 needs Colab, because the drawing canvas uses `google.colab.output`. Part 3 runs 5,000 searches per maze size and takes over ten minutes on a laptop.

## Lab 2: Prolog movie recommender

**Task.** Using `movies_metadata.csv` (4,803 movies with genres, plot keywords, director, top three actors, language, production companies and countries, and year):

1. Build a Prolog knowledge base with facts such as `director(Movie, Director)` and `genre(Movie, Genre)`. Write rules that find movies with a common genre, director, plot keywords, actors, language, colour, studio, country or decade, at several levels of strength (for example "all three lead actors in common" against "at least one").
2. Combine the rules into graded similarity queries, `find_sim_5` (most similar) down to `find_sim_1`, and a function that returns similar movies in decreasing order of similarity.
3. Use one user's ratings to train a recommender on top of the similarity queries, then evaluate it on a separate set of that user's ratings with precision, recall and F1.

**How it works.**

- **Knowledge base.** The notebook reads the CSV with pandas, cleans the text, and asserts one fact per attribute value through pyswip: `director`, `language`, `actor_name`, `genre`, `plot_keyword`, `production_company`, `production_country`, `title_year` (rounded down to the decade), and numeric attributes such as `budget` and `vote_average`.
- **Rules.** [`db.pl`](programming/lab2/db.pl) defines rules such as `atLeastTwoCommonGenres`, `atLeastTwoCommonKeywords`, `atLeastOneCommonActor`, `sameDirector` and `sameLanguage`. The similarity levels combine them:

  | Query | Requires |
  |---|---|
  | `find_sim_5` | ≥ 2 genres, ≥ 2 keywords, ≥ 2 actors, same director, same language |
  | `find_sim_4` | ≥ 2 genres, ≥ 2 keywords, ≥ 1 actor, same director, same language |
  | `find_sim_3` | ≥ 2 genres, ≥ 1 keyword, ≥ 1 actor, same language |
  | `find_sim_2` | ≥ 2 genres, ≥ 1 keyword, same language |
  | `find_sim_1` | ≥ 1 genre, ≥ 1 keyword |

  `simple_recommender(movie)` runs the queries from level 5 down to level 1 and returns each matching movie once, most similar first. For example, `simple_recommender("Alien")` starts with *Avatar*, *9*, *After Earth*, *Alien: Resurrection*, *Aliens vs Predator: Requiem* and *Aliens*.
- **Training.** Every movie starts with a score of 0. For each movie the user rated, every similar movie gets a weight added to its score. Ratings are truncated to whole stars: 4 and 5 add a positive weight, 2 and 3 add nothing, and 1 subtracts a little. A test movie is predicted as "recommend" if its score is above 0, and the ground truth is "recommend" if the user rated it above 3.
- **Results.** The data is user 547's ratings: 100 for training and 141 for testing. The notebook trains on random subsets of the training ratings and averages 10 runs per size:

  | Training ratings | Precision | Recall | F1 |
  |---|---|---|---|
  | 10 | 0.49 | 0.10 | 0.16 |
  | 30 | 0.50 | 0.36 | 0.42 |
  | 100 (all) | 0.48 | 0.64 | 0.55 |

  More ratings mainly improve recall. A test movie that is not similar to any training movie gets no score and is never recommended, and with fewer training ratings more test movies are in that position.

**Run.** In Colab, copy `movies_metadata.csv`, `train_ratings.csv`, `test_ratings.csv` and `db.pl` to `MyDrive/NTUA/TN/AI_24/` in Google Drive, open the notebook and run all cells. The second code cell installs SWI-Prolog and pyswip. Locally, install SWI-Prolog first (for example `brew install swi-prolog` or `sudo apt install swi-prolog`). Then start Jupyter from `programming/lab2`, skip the two Colab-only cells (Drive mount and installation), and set `path = ''` in the cell that reads `movies_metadata.csv`.

## Exercise set 1: Search, games and genetic algorithms

Written answers in [`theoretical/set1/AI24_hwk1_03121087_NasopoulouEleni.pdf`](theoretical/set1/AI24_hwk1_03121087_NasopoulouEleni.pdf) (Greek). The exercises:

1. Run hill climbing, best-first search and A\* with a closed set on a given graph and trace the frontier step by step. List all solutions and say which algorithms find the optimal one and whether that could be known in advance. Check the heuristic for consistency, fix it with minimal changes, and give a more accurate consistent heuristic.
2. Give a graph with an admissible heuristic where A\* returns a non-optimal path if it stops as soon as it first finds a path to the goal, or prove that none exists.
3. Compute the minimax values of a given game tree and the order in which alpha-beta visits its nodes, and mark the nodes it prunes.
4. For a genetic algorithm with 8-digit chromosomes and fitness f(x) = (a+b+c+d) − (e+f+g+h), compute the fitness of a population, apply three kinds of crossover, compare the total fitness of the next generation, and find the optimal chromosome.

## Exercise set 2: Logic and resolution

Written answers in [`theoretical/set2/AI24_hkw2_03121087_NasopoulouEleni.pdf`](theoretical/set2/AI24_hkw2_03121087_NasopoulouEleni.pdf) (Greek). The exercises:

1. Convert a propositional formula and a first-order formula to conjunctive normal form, including Skolemization.
2. For a relation P on a 6-element set S with disjoint subsets A and B, write first-order axioms for reflexive, symmetric, antisymmetric, transitive and related properties, give models for each, and use models and resolution to check whether P being a partial order on A and B implies that it is a partial order on S.
3. Axiomatize natural-number addition and "less than" with `zero` and `succ`, then use resolution to prove 4 + 2 = 6, compute 6 + 3, and study whether resolution terminates on 0 < 0.

The bonus exercises of this set (description logic models and PAC learnability of axis-aligned rectangles) are not answered.

## Repository layout

```
programming/
├── lab1/
│   ├── AI_24_EX1.ipynb      Maze generation, search comparison, ghost chase, hand-drawn mazes
│   └── Gifs Diagrams/       Saved plots and search animations
└── lab2/
    ├── AI_24_EX2.ipynb      Knowledge base, recommender, evaluation
    ├── db.pl                Prolog similarity rules
    ├── movies_metadata.csv  Movie dataset provided with the assignment
    ├── train_ratings.csv    User 547: 100 ratings for training
    └── test_ratings.csv     User 547: 141 ratings for testing
theoretical/
├── set1/                    Written answers to exercise set 1 (PDF, Greek)
└── set2/                    Written answers to exercise set 2 (PDF, Greek)
```

## Known limitations

The notebooks are kept as they were submitted. While writing this README, I reviewed them again and found the issues below.

- **Lab 1, ghost chase.** `alpha_beta_agent` does not pass `heuristic_parameter` on to its recursive calls, so the leaves are always scored with the default, heuristic 3. Choosing heuristic 1 or 2 has no effect, and the comparison of the three heuristics in the notebook does not compare three different agents. `is_lose` compares the agent's real position with the list of ghost positions, so it is never true, and the search never sees a capture. It only avoids the ghost through the distance term of the evaluation. The evaluation function also receives the global `map` instead of the `maze` argument.
- **Lab 1, part 3 commentary.** The notebook describes the growth of expanded nodes as O(N·log N) for A\* and Dijkstra and O(N) for greedy search. N is the side of the maze. A\* and Dijkstra expand a large share of the N² cells, so their counts grow roughly with N², as the table above shows.
- **Lab 1, search.** `pathfinder` copies the whole frontier and the list of visited nodes after every neighbour, for the animation, even when `visualize=False`. This makes the part 3 experiment much slower than it needs to be.
- **Lab 2, knowledge base.** The cell that builds the facts stores each movie's running time as a second `director` fact (for example `director('Avatar', '162.0')`). `sameDirector`, which `find_sim_4` and `find_sim_5` use, therefore also matches movies with the same running time. Movies with no director all share the value `UNK`.
- **Lab 2, `db.pl`.** `ThreeCommonActors` starts with a capital letter, which Prolog reads as a variable, so that clause does not load. `blacknWhite` calls `plot_keywords` instead of `plot_keyword`, and `sameDecade` uses `decade/2`, but the notebook asserts `title_year/2`. None of these rules is used by the `find_sim_*` queries, so the results are not affected.
- **Lab 2, training.** The weight added for each similar movie is `rating_weights[rating] * score_weights[rating]`. `score_weights` was meant to be indexed by similarity level, but `simple_recommender` returns a flat list, so how similar a movie is does not affect its score.
- **Paths.** Both notebooks use a hard-coded Google Drive folder (`/content/drive/MyDrive/NTUA/TN/AI_24/`), as described under each lab.
- **Written answers.** Some answers in exercise set 2 contain mistakes. For example, the model given for a transitive relation, {(s1,s2), (s2,s3), (s3,s1)}, is not transitive, and some of the equivalence-relation models leave out the reflexive pairs.

## For students taking the course

This repository is here to help you understand the material: how the search algorithms behave on real mazes, and how a Prolog knowledge base can drive a simple recommender. Write your own solutions, and check anything you reuse against the issues listed above.

The course's own notes, and the chapters on search, adversarial search and logic in Russell and Norvig's *Artificial Intelligence: A Modern Approach*, are good references for everything here.

import random
from models import Problem
from types import List
from sqlmodel import Session, create_engine, select
from collections import defaultdict
from datetime import datetime, timedelta
import math

engine = create_engine("sqlite:///./database.db")

CATEGORY_PRIORITY = [
    "Arrays and Hashing",
    "Two Pointers",
    "Stack",
    "Sliding Window",
    "Binary Search",
    "Linked List",
    "Tree",
    "Tries",
    "Heap / Priority Queue",
    "Backtracking",
    "Intervals",
    "Greedy",
    "Graph",
    "Graph2",
    "1DDP",
    "2DDP",
    "Bit Manipulation",
    "Math",
]

DIFFICULTY_PRIORITY = ["Easy", "Medium", "Hard"]

def recommend_ordered_problem() -> Problem:
    categories = epsilon_greed()

    category = interleaving(categories)

    problems = sm_2(category)

    return random.choice(problems)

# Return 3 categories (Alex)
def epsilon_greed() -> List[str]:
    pass

# Return the category chosen (Alex)
def interleaving(categories) -> str:
    pass

# Return list of problems (Nathan)
def sm_2(category) -> List[Problem]:
    pass
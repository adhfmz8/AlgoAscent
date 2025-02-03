# backend/recommender.py
import random
from typing import List, Optional
from datetime import datetime
# from collections import defaultdict
from sqlmodel import Session, create_engine, select
from models import Problem, UserAttempt

# Database engine
engine = create_engine("sqlite:///./database.db")

# Constants
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
    "Math"
]

DIFFICULTY_PRIORITY = ["Easy", "Medium", "Hard"]
MASTERY_THRESHOLD = 3


def filter_problems_by_category_and_difficulty(problems: List[Problem], category: str, difficulty: str) -> List[Problem]:
    """Filter problems by category and difficulty."""
    return [problem for problem in problems if problem.neetcode_category == category and problem.difficulty == difficulty]


def get_solved_problems_in_category(attempts: List[UserAttempt], problems: List[Problem], category: str, difficulty: str) -> int:
    """Count the number of solved problems in a specific category and difficulty."""
    return len([
        attempt for attempt in attempts
        if attempt.problem_id in [p.id for p in problems if p.neetcode_category == category and p.difficulty == difficulty]
        and attempt.solved
    ])


def prioritize_problems(problems: List[Problem], attempts: List[UserAttempt], category: str, is_review: bool = False) -> List[Problem]:
    """Prioritize problems in a category based on difficulty and user attempts."""
    prioritized_problems = []
    solved_easy_problems = get_solved_problems_in_category(attempts, problems, category, "Easy")
    easy_problems_in_category = len([p for p in problems if p.difficulty == "Easy"])

    for difficulty in DIFFICULTY_PRIORITY:
        problems_with_difficulty = filter_problems_by_category_and_difficulty(problems, category, difficulty)
        if problems_with_difficulty:
            if difficulty == "Easy":
                prioritized_problems.extend(problems_with_difficulty)
            elif is_review or (solved_easy_problems >= MASTERY_THRESHOLD or solved_easy_problems == easy_problems_in_category):
                prioritized_problems.extend(problems_with_difficulty)

    return prioritized_problems


def recommend_ordered_problem() -> Optional[Problem]:
    """Recommend a problem based on user attempts, category priority, and difficulty."""
    with Session(engine) as session:
        # Fetch all problems and attempts
        all_problems = session.exec(select(Problem)).all()
        all_attempts = session.exec(select(UserAttempt)).all()

        # Identify completed problems
        completed_problem_ids = set(attempt.problem_id for attempt in all_attempts if attempt.solved)

        # Separate reviewable and unattempted problems
        now = datetime.now()
        reviewable_problems = []
        unattempted_problems = []

        for problem in all_problems:
            if problem.id in completed_problem_ids:
                continue

            user_attempt = next((attempt for attempt in all_attempts if attempt.problem_id == problem.id), None)
            if user_attempt and user_attempt.next_review_date <= now:
                reviewable_problems.append(problem)
            else:
                unattempted_problems.append(problem)

        # Prioritize reviewable problems
        prioritized_problems = []
        for category in CATEGORY_PRIORITY:
            review_problems_in_category = [p for p in reviewable_problems if p.neetcode_category == category]
            if review_problems_in_category:
                prioritized_problems.extend(prioritize_problems(review_problems_in_category, all_attempts, category, is_review=True))

        if prioritized_problems:
            return random.choice(prioritized_problems)

        # If no reviewable problems, prioritize unattempted problems
        for category in CATEGORY_PRIORITY:
            unattempted_problems_in_category = [p for p in unattempted_problems if p.neetcode_category == category]
            if unattempted_problems_in_category:
                # Check if any attempt has been made in this category yet
                has_attempt = any(attempt for attempt in all_attempts if attempt.problem_id in [p.id for p in unattempted_problems_in_category])
                if not has_attempt:
                    easy_problems = filter_problems_by_category_and_difficulty(unattempted_problems_in_category, category, "Easy")
                    if easy_problems:
                        prioritized_problems.extend(easy_problems)
                        continue  # Skip to the next category

                prioritized_problems.extend(prioritize_problems(unattempted_problems_in_category, all_attempts, category))

        if prioritized_problems:
            return random.choice(prioritized_problems)

        # Fallback: Return a random problem if no prioritized problems are found
        return random.choice(all_problems) if all_problems else None
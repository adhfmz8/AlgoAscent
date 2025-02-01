# backend/recommender.py
import random
from models import Problem, UserAttempt
from typing import List
from sqlmodel import Session, create_engine, select
from collections import defaultdict
from datetime import datetime, timedelta

engine = create_engine("sqlite:///./database.db")

CATEGORY_PRIORITY = [
    "Arrays and Hashing",
    "Two Pointers",
    "Stack",
    "Sliding Window",
    "Binary Search",
    "Linked List",
    "Tree",
    "Heap / Priority Queue",
    "Backtracking",
    "Graph",
]


def recommend_ordered_problem() -> Problem:
    with Session(engine) as session:
        statement = select(Problem)
        results = session.exec(statement)
        all_problems = results.all()

        statement = select(UserAttempt)
        results = session.exec(statement)
        all_attempts = results.all()

        completed_problem_ids = set(
            attempt.problem_id for attempt in all_attempts if attempt.solved
        )

        category_counts = defaultdict(int)

        for problem in all_problems:
            if problem.id not in completed_problem_ids:
                category_counts[problem.neetcode_category] += 1

        # Get the current date
        now = datetime.now()

        reviewable_problems = []
        unattempted_problems = []
        # Iterate through all problems and sort them into their proper place.
        for problem in all_problems:
            if (
                problem.id in completed_problem_ids
            ):  # Skip if problem is already completed
                continue

            # Find the users attempt
            user_attempt = next(
                (
                    attempt
                    for attempt in all_attempts
                    if attempt.problem_id == problem.id
                ),
                None,
            )

            # If the problem has already been attempted, add it to the reviewable list if it is needed
            if user_attempt:
                if user_attempt.next_review_date <= now:
                    reviewable_problems.append(problem)
            else:
                unattempted_problems.append(problem)

        # Sort the reviewable problems by category priority
        prioritized_problems = []

        if (
            reviewable_problems
        ):  # Check if reviewable problems has entries, if it does, prioritize those.
            for category in CATEGORY_PRIORITY:
                review_problems_in_category = [
                    problem
                    for problem in reviewable_problems
                    if problem.neetcode_category == category
                ]
                if review_problems_in_category:
                    prioritized_problems.extend(review_problems_in_category)
            return random.choice(prioritized_problems)

        else:  # Otherwise, use the prioritized unattempted problems list, or a random choice
            for category in CATEGORY_PRIORITY:
                unattempted_problems_in_category = [
                    problem
                    for problem in unattempted_problems
                    if problem.neetcode_category == category
                ]
                if unattempted_problems_in_category:
                    prioritized_problems.extend(unattempted_problems_in_category)
            if prioritized_problems:
                return random.choice(prioritized_problems)
            else:
                # Return random problem if nothing is found in those categories
                return random.choice(all_problems)

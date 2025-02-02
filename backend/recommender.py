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

        # Sort the reviewable problems by category priority and then difficulty.
        prioritized_problems = []
        if reviewable_problems:
             for category in CATEGORY_PRIORITY:
                  review_problems_in_category = [
                      problem
                      for problem in reviewable_problems
                      if problem.neetcode_category == category
                    ]
                  if review_problems_in_category:
                     # Count solved easy problems in the category
                    solved_easy_problems = len([
                        attempt
                        for attempt in all_attempts
                        if attempt.problem_id in [p.id for p in review_problems_in_category if p.difficulty == "Easy"]
                        and attempt.solved
                         and attempt.problem_id
                        not in completed_problem_ids
                        ])
                    for difficulty in DIFFICULTY_PRIORITY:
                        if difficulty == "Easy":
                           unattempted_problems_with_difficulty = [problem for problem in review_problems_in_category if problem.difficulty == difficulty]
                           if unattempted_problems_with_difficulty:
                             prioritized_problems.extend(unattempted_problems_with_difficulty)
                        elif difficulty == "Medium" and (solved_easy_problems >= MASTERY_THRESHOLD or solved_easy_problems == len([p for p in review_problems_in_category if p.difficulty == "Easy"])) : # Only add medium problems if the user has passed the required threshold, OR there are no more easy problems left
                            unattempted_problems_with_difficulty = [problem for problem in review_problems_in_category if problem.difficulty == difficulty]
                            if unattempted_problems_with_difficulty:
                             prioritized_problems.extend(unattempted_problems_with_difficulty)
                        elif difficulty == "Hard" and (solved_easy_problems >= MASTERY_THRESHOLD or solved_easy_problems == len([p for p in review_problems_in_category if p.difficulty == "Easy"])): # Only add hard problems if the user has passed the required threshold, OR there are no more easy problems left
                             unattempted_problems_with_difficulty = [problem for problem in review_problems_in_category if problem.difficulty == difficulty]
                             if unattempted_problems_with_difficulty:
                                prioritized_problems.extend(unattempted_problems_with_difficulty)

        if prioritized_problems:
            return random.choice(prioritized_problems)

        else:  # Otherwise, use the prioritized unattempted problems list, or a random choice
            for category in CATEGORY_PRIORITY:
                unattempted_problems_in_category = [
                    problem
                    for problem in unattempted_problems
                    if problem.neetcode_category == category
                ]
                if unattempted_problems_in_category:
                    # Check if any attempt has been made in this category yet
                     # Check if any attempt has been made in this category yet
                    has_attempt = any(attempt for attempt in all_attempts if attempt.problem_id in [p.id for p in unattempted_problems_in_category])
                    if not has_attempt:
                          unattempted_problems_with_difficulty = [problem for problem in unattempted_problems_in_category if problem.difficulty == "Easy"] # Always pick an easy problem if no attempts in this category
                          if unattempted_problems_with_difficulty:
                            prioritized_problems.extend(unattempted_problems_with_difficulty)
                            break #Skip the rest of the logic for this category, and go to next category
                    
                    # Count solved easy problems in the category
                    solved_easy_problems = len([
                        attempt
                        for attempt in all_attempts
                        if attempt.problem_id in [p.id for p in unattempted_problems_in_category if p.difficulty == "Easy"]
                        and attempt.solved
                        ])
                    easy_problems_in_category = len([p for p in unattempted_problems_in_category if p.difficulty == "Easy"])
                    for difficulty in DIFFICULTY_PRIORITY:
                        if difficulty == "Easy":
                            unattempted_problems_with_difficulty = [
                                problem
                                for problem in unattempted_problems_in_category
                                if problem.difficulty == difficulty
                            ]
                            if unattempted_problems_with_difficulty:
                                prioritized_problems.extend(unattempted_problems_with_difficulty)
                        elif difficulty == "Medium" and (solved_easy_problems >= MASTERY_THRESHOLD or solved_easy_problems == easy_problems_in_category) : # Only add medium problems if the user has passed the required threshold or if the problems are exhausted.
                            unattempted_problems_with_difficulty = [
                                problem
                                for problem in unattempted_problems_in_category
                                if problem.difficulty == difficulty
                            ]
                            if unattempted_problems_with_difficulty:
                              prioritized_problems.extend(unattempted_problems_with_difficulty)
                        elif difficulty == "Hard" and (solved_easy_problems >= MASTERY_THRESHOLD or solved_easy_problems == easy_problems_in_category): # Only add hard problems if the user has passed the required threshold or if the problems are exhausted.
                            unattempted_problems_with_difficulty = [
                            problem
                                for problem in unattempted_problems_in_category
                                if problem.difficulty == difficulty
                            ]
                            if unattempted_problems_with_difficulty:
                                prioritized_problems.extend(unattempted_problems_with_difficulty)

            if prioritized_problems:
                return random.choice(prioritized_problems)
            else:
                # Return random problem if nothing is found in those categories
                return random.choice(all_problems)
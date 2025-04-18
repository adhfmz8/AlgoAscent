# backend/recommender.py
import random
from models import Problem, UserAttempt, ProblemMemory
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
    "Math",
]

DIFFICULTY_PRIORITY = ["Easy", "Medium", "Hard"]
MASTERY_THRESHOLD = 3


def recommend_ordered_problem() -> Problem:
    with Session(engine) as session:
        now = datetime.now()

        # --- 1. Load Data ---
        problems = session.exec(select(Problem)).all()
        attempts = session.exec(select(UserAttempt)).all()
        memories = session.exec(select(ProblemMemory)).all()

        problem_map = {p.id: p for p in problems} # Map ID to problem for quick lookup
        completed_ids = {a.problem_id for a in attempts if a.solved}
        memory_by_id = {m.problem_id: m for m in memories}

        # --- 2. Priority 1: Problems Needing Review ---
        review_problem_ids = {
            p.id
            for p in problems
            if p.id in memory_by_id and memory_by_id[p.id].next_review_date <= now
        }
        if review_problem_ids:
            review_problems = [p for p in problems if p.id in review_problem_ids]
            # Optional: Prioritize reviews by earliest date? For now, random.
            print(f"Recommending review problem from {len(review_problems)} due.")
            return random.choice(review_problems)

        # --- 3. Calculate Mastery Per Category ---
        solved_easy_counts_per_category = defaultdict(int)
        for attempt in attempts:
            if attempt.solved:
                problem = problem_map.get(attempt.problem_id)
                if problem and problem.difficulty == "Easy":
                    solved_easy_counts_per_category[problem.neetcode_category] += 1

        # --- 4. Find Available Problems (Not Completed, Not Review) ---
        available_problems = [
            p for p in problems
            if p.id not in completed_ids and p.id not in review_problem_ids
        ]

        # --- 5. Linear Progression Logic ---
        print("Starting linear progression check...")
        for idx, category in enumerate(CATEGORY_PRIORITY):
            print(f"Checking Category: {category} (Index: {idx})")

            # Check eligibility based on the *previous* category's mastery
            if idx > 0:
                previous_category = CATEGORY_PRIORITY[idx - 1]
                mastery_in_previous = solved_easy_counts_per_category[previous_category]
                if mastery_in_previous < MASTERY_THRESHOLD:
                    print(f"  Mastery threshold ({MASTERY_THRESHOLD}) not met in previous category '{previous_category}' ({mastery_in_previous} solved). Stopping progression.")
                    # User is stuck on the *previous* category until mastery.
                    # Find an *easy* problem in the *previous* category if available.
                    stuck_category_problems = [
                        p for p in available_problems
                        if p.neetcode_category == previous_category and p.difficulty == "Easy"
                    ]
                    if stuck_category_problems:
                         print(f"  Recommending another Easy problem from '{previous_category}' to meet threshold.")
                         return random.choice(stuck_category_problems)
                    else:
                         # No more easy problems in the category they are stuck on.
                         # What to do? Fallback for now. Maybe offer medium? Or just go to general fallback.
                         print(f"  No more Easy problems found in '{previous_category}'. Breaking to fallback.")
                         break # Go to fallback logic

            # --- If eligible for this category ---
            print(f"  Eligible for category: {category}")
            category_problems = [
                p for p in available_problems if p.neetcode_category == category
            ]

            if not category_problems:
                print(f"  No available problems found for {category}. Continuing to next.")
                continue # Move to the next category

            # Prioritize Easy problems first
            easy_problems = [p for p in category_problems if p.difficulty == "Easy"]
            if easy_problems:
                print(f"  Recommending Easy problem from {category}.")
                return random.choice(easy_problems)

            # If no Easy problems left, check if ready for Medium/Hard in *this* category
            mastery_in_current = solved_easy_counts_per_category[category]
            total_easy_in_category = len([p for p in problems if p.neetcode_category == category and p.difficulty == "Easy"]) # Need total count here

            # Allow Medium/Hard if threshold met OR all easy problems in this category are solved
            if mastery_in_current >= MASTERY_THRESHOLD or mastery_in_current == total_easy_in_category :
                print(f"  Easy mastery met/completed for {category}. Checking Medium/Hard.")
                medium_problems = [p for p in category_problems if p.difficulty == "Medium"]
                if medium_problems:
                    print(f"  Recommending Medium problem from {category}.")
                    return random.choice(medium_problems)

                hard_problems = [p for p in category_problems if p.difficulty == "Hard"]
                if hard_problems:
                    print(f"  Recommending Hard problem from {category}.")
                    return random.choice(hard_problems)
            else:
                 print(f"  Easy mastery ({mastery_in_current}/{MASTERY_THRESHOLD}) not yet met for {category}. Cannot recommend Medium/Hard. Must solve Easy first.")
                 # Since easy_problems was empty, and mastery isn't met for Med/Hard,
                 # this implies user should continue with easy problems but there are none available *in this category*.
                 # The loop will continue to the next category check (which might fail on eligibility).


        # --- 6. Fallback Logic ---
        print("Linear progression finished or blocked. Using fallback.")
        if available_problems:
            # Fallback 1: Recommend any available (uncompleted, non-review) problem
            print(f"Fallback: Recommending a random available problem from {len(available_problems)} options.")
            return random.choice(available_problems)
        elif problems:
             # Fallback 2: All problems are either completed or need review.
             # If review queue wasn't empty, we'd have returned earlier.
             # This means all problems are completed, or something is wrong.
             # Let's just return a random problem overall as a last resort.
             print("Fallback: No available problems. Recommending a random problem overall.")
             return random.choice(problems)
        else:
            # Fallback 3: No problems in DB?
            raise ValueError("No problems found in the database.")




def calculate_new_easiness_factor(
    last_attempt: UserAttempt, solved: bool, time_taken: float
):
    easiness_factor = last_attempt.easiness_factor
    if solved:
        easiness_factor += 0.1
        if time_taken < 30:
            easiness_factor += 0.1
    else:
        easiness_factor -= 0.2
        if time_taken > 30:
            easiness_factor -= 0.2
    return max(1.3, easiness_factor)  # don't let easiness factor go too low


def calculate_next_review_date(easiness_factor: float):
    days_until_review = easiness_factor
    return datetime.now() + timedelta(days=days_until_review)


def update_sm2(memory: ProblemMemory, quality: int) -> ProblemMemory:
    ef = memory.easiness_factor
    r = memory.repetitions
    last_interval = memory.last_interval

    # SM-2 easiness factor update
    ef += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    ef = max(1.3, ef)

    if quality < 3:
        r = 0
        interval_days = 1
    else:
        r += 1
        if r == 1:
            interval_days = 1
        elif r == 2:
            interval_days = 6
        else:
            interval_days = last_interval * ef

    memory.easiness_factor = ef
    memory.repetitions = r
    memory.last_interval = interval_days
    memory.last_attempt_date = datetime.now()
    memory.next_review_date = memory.last_attempt_date + timedelta(
        days=round(interval_days)
    )

    return memory


def estimate_quality(solved: bool, time_taken: float) -> int:
    if not solved:
        return 2 if time_taken < 30 else 1
    else:
        return 5 if time_taken < 20 else 4

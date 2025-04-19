# Modified recommender.py to add dynamic interleaving
import random
from models import Problem, UserAttempt, ProblemMemory
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
MASTERY_THRESHOLD = 3

# How much we interleave (0 to 1)
INTERLEAVING_FACTOR = 0.3 
# Max number of categories to interleave
MAX_INTERLEAVING_CATEGORIES = 3 

def recommend_ordered_problem() -> Problem:
    with Session(engine) as session:
        now = datetime.now()

        # Fetch problems and attemps
        problems = session.exec(select(Problem)).all()
        attempts = session.exec(select(UserAttempt)).all()
        memory = session.exec(select(ProblemMemory)).all()

        problem_map = {p.id: p for p in problems}
        completed_ids = {a.problem_id for a in attempts if a.solved}
        memory_by_id = {m.problem_id: m for m in memory}

        # Prioritize problems that need review
        review_problem_ids = {
            p.id
            for p in problems
            if p.id in memory_by_id and memory_by_id[p.id].next_review_date <= now
        }
        if review_problem_ids:
            review_problems = [p for p in problems if p.id in review_problem_ids]
            # Interleaving for review problems across the different categories
            review_by_category = defaultdict(list)
            for p in review_problems:
                review_by_category[p.neetcode_category].append(p)

            # If we have review from multiple categories, apply inerleaving
            if len(review_by_category) > 1:
                categories = list(review_by_category.keys())
                # Randomly select a category with higher weight
                category_weights = [math.exp(-0.5*i) for i in range(len(categories))]
                selected_category = random.choices(categories, weights=category_weights)[0]
                print(f'Interleaving review: Selected category {selected_category}')
                return random.choice(review_by_category[selected_category])
            else:
                print(f'Recommending review problem from {len(review_problems)} due.')
                return random.choice(review_problems)
            
        # Calculate Mastery for each category
        solved_easy_counter_per_category = defaultdict(int)
        for attempt in attempts:
            if attempt.solved:
                problem = problem_map.get(attempt.problem_id)
                if problem and problem.difficulty == 'Easy':
                    solved_easy_counter_per_category[problem.neetcode_category] += 1

        # Find New problems
        new_problems = [
            p for p in problems
            if p.id not in completed_ids and p.id not in review_problem_ids
        ]

        # Find eligible categories for Interleaving
        eligible_categories = []

        for indx, category in enumerate(CATEGORY_PRIORITY):
            print(f'Checking Category: {category} (Index: {indx})')

            # Check eligibility based on previous category mastery
            category_eligible = True
            if indx > 0:
                previous_category = CATEGORY_PRIORITY[indx - 1]
                mastery_in_previous = solved_easy_counter_per_category[previous_category]
                if mastery_in_previous < MASTERY_THRESHOLD:
                    print(f'Master threshold not met in previous category: {previous_category}')
                    if indx == 1: # Always make the first category eligible
                        category_eligible = True
                    else:
                        category_eligible = False

            # Look for problems in the elibible categories
            if category_eligible:
                category_problems = [p for p in new_problems if p.neetcode_category == category]
                
                if category_problems:
                    # Check if we have easy problems, if not we've master Easy
                    easy_problems = [p for p in category_problems if p.difficulty == 'Easy']

                    if easy_problems:
                        eligible_categories.append((category, 'Easy'))
                    else:
                        # Check if ready for harder problems
                        mastery_in_current = solved_easy_counter_per_category[category]
                        total_easy_in_category = len([p for p in problems
                                                      if p.neetcode_category == category
                                                      and p.difficulty == 'Easy'])
                        
                        if mastery_in_current >= MASTERY_THRESHOLD or mastery_in_current == total_easy_in_category:
                            medium_problems = [p for p in category_problems if p.difficulty == 'Medium']
                            if medium_problems:
                                eligible_categories.append((category, 'Medium'))
                            else:
                                hard_problems = [p for p in category_problems if p.difficulty == 'Hard']
                                if hard_problems:
                                    eligible_categories.append(category, 'Hard')

            # Apply Dynamic Interleaving to select a category and difficulty
            if eligible_categories:
                print(f'Eligible categories for interleaving: {eligible_categories}')

                # See if we should interleave or go with primary category
                if random.random() < INTERLEAVING_FACTOR and len(eligible_categories) > 1:
                    num_categories = min(MAX_INTERLEAVING_CATEGORIES, len(eligible_categories))

                    # Apply and exponential weight to favor earlier categories
                    weight = [math.exp(-0.5*i) for i in range(len(eligible_categories))]
                    selected_categories = random.choices(
                        eligible_categories,
                        weights=weight,
                        k=num_categories
                    )

                    # Randomly select from selected category
                    selected_category, selected_difficulty = random.choice(selected_categories)
                    print(f'Interleaving applied: Selected {selected_category} ({selected_difficulty})')
                else:
                    # Default to the first eligible category
                    selected_category, selected_difficulty = eligible_categories[0]
                    print(f'Using first category: {selected_category}, ({selected_difficulty})')

                # Find a problem based on the selection
                matching_problems = [
                    p for p in new_problems
                    if p.neetcode_category == selected_category and p.difficulty == selected_difficulty
                ]

                if matching_problems:
                    return random.choice(matching_problems)
            
            # Fallback if no eligible problems are found
            print('No Eligible Problems Found! Using fallback')
            if new_problems:
                return random.choice(new_problems)
            elif problems:
                return random.choice(problems)
            else:
                raise ValueError('No problems found in the database')
            
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

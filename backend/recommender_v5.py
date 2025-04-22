import random
from models import Problem, UserAttempt, ProblemMemory
from typing import List, Optional
from sqlmodel import Session, create_engine, select
from collections import defaultdict
from datetime import datetime, timedelta
import math

engine = create_engine("sqlite:///./database.db")

"""
Legend for Master:
    3 = Relavtively straightforward
    4 = Moderate complexity
    5 = More complex
    6 = Advanced
    7 = Very advanced
"""
CATEGORY_PRIORITY = {
    "Arrays and Hashing": 3,
    "Two Pointers": 3,
    "Stack": 4,
    "Sliding Window": 4,
    "Binary Search": 4,
    "Linked List": 4,
    "Tree": 5,
    "Tries": 5,
    "Heap / Priority Queue": 5,
    "Backtracking": 6,
    "Intervals": 4,
    "Greedy": 5,
    "Graph": 6,
    "Graph2": 6,
    "1DDP": 6,
    "2DDP": 7,
    "Bit Manipulation": 5,
    "Math": 4,
}

DIFFICULTY_PRIORITY = ["Easy", "Medium", "Hard"]
MASTERY_THRESHOLD = 3 # Used as a backup if the category is not in the priority list

INTERLEAVING_FACTOR = 0.3 # Start with low interleaving
INTERLEAVING_INCREASE_RATE = 0.01 # How quickly to increase interleaving factor
EPSILON = 0.2 # Exploration rate

# Global cache for category rewards
category_rewards = {category: 1.0 for category in list(CATEGORY_PRIORITY.keys())}
# Global cache for category attempts
category_attempts = {category: 0 for category in list(CATEGORY_PRIORITY.keys())}
# Total attempts
total_attempts = 0

def recommend_ordered_problem() -> Problem:
    with Session(engine) as session:
        # Get the unlocked categories and the categories with due problems
        unlocked_categories = get_unlocked_categories(session)
        sm2_due_categories = get_categories_with_pending_reviews(session)

        # Select 3 categories using epsilon-greedy strategy
        categories = epsilon_greedy(unlocked_categories)

        # Apply interleaving to choose which category to recommend
        category = interleaving(categories, sm2_due_categories)

        # Get prolems from the selected category, preferring SM-2 due problems
        problem = sm_2(category, session)

        # Fallback if any fail
        if not problem:
            return get_fallback_problem(session)

        return problem

# Return 3 categories (Alex)
def epsilon_greedy(unlocked_categories: List[str]) -> List[str]:
    # Fallback
    if not unlocked_categories:
        print('No unlocked categories, using first category from priority list')
        return [list(CATEGORY_PRIORITY.keys())[0]]
    
    # Make sure we have a valid number to select
    num_to_select = min(3, len(unlocked_categories))

    # Exploration
    if random.random() < EPSILON:
        selected = random.sample(unlocked_categories, num_to_select)
        print(f'Epsilon-greedy selected {selected} from {unlocked_categories}')
        return selected
    # Exploitation
    else:
        # Sort categories by their rewards
        sorted_categories = sorted(
            unlocked_categories,
            key=lambda cat: category_rewards.get(cat, 0.0),
            reverse=True
        )

        # Take the top 3 categories
        selected = sorted_categories[:num_to_select]
        print(f'Epsilon-greedy selected {selected} from {unlocked_categories}')
        return selected

# Return the category chosen (Alex)
def interleaving(categories: List[str], sm2_due_categories: List[str]) -> str:
    global INTERLEAVING_FACTOR

    # combine categories and remove and duplicates
    combined_categories = list(set(categories + sm2_due_categories))

    # Fallback, if no categories are available
    if not combined_categories:
        print('No categories to select from')
        return list(CATEGORY_PRIORITY.keys())[0]
    
    weights = []

    for category in combined_categories:
        base_weight = category_rewards.get(category, 1.0)

        # Boost weight for SM-2 due categories
        if category in sm2_due_categories:
            sm2_boost = 1.5
            weight = base_weight * sm2_boost
        else:
            weight = base_weight
        
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(combined_categories)] * len(combined_categories)
    
    # Create a blended weights using interleaving factor
    uniform_weights = [1.0 / len(combined_categories)] * len(combined_categories)
    blended_weights = []

    for i in range(len(combined_categories)):
        blended_weight = (1 - INTERLEAVING_FACTOR) * weights[i] + INTERLEAVING_FACTOR * uniform_weights[i]
        blended_weights.append(blended_weight)

    # Select the category based on the blended weights
    selected_category = random.choices(
        combined_categories,
        weights=blended_weights,
        k=1
    )[0]

    print(f'Interleaving selected {selected_category} from {combined_categories}')
    print(f'Categories: {combined_categories}')
    print(f"Weights: {dict(zip(combined_categories, blended_weights))}")

    return selected_category

# Return problem from chosen category, prioritizing SM-2 due problems (Nathan)
def sm_2(category: str, session: Session) -> Optional[Problem]: # Added Optional type hint
    due_problems = get_due_problems(category, session)

    if due_problems:
        problem = random.choice(due_problems)
        print(f'Selected SM-2 due problem: {problem.title} (ID: {problem.id})')
        return problem

    # If no problem is due, get unsolved problems
    unsolved_problems = get_unsolved_problems(category, session)

    # Initialize problems_by_difficulty *before* checking unsolved_problems
    problems_by_difficulty = defaultdict(list)

    if unsolved_problems:
        # Populate the dictionary if there are problems
        for problem in unsolved_problems:
            problems_by_difficulty[problem.difficulty].append(problem)

    # Now this loop is safe, problems_by_difficulty always exists
    # If it's empty, the inner 'if' condition will just be false
    for difficulty in DIFFICULTY_PRIORITY:
        if problems_by_difficulty[difficulty]:
            problem = random.choice(problems_by_difficulty[difficulty])
            print(f'Selected unsolved problem: {problem.title} (ID: {problem.id})')
            return problem

    # Fallback if no problems are available (either due or unsolved)
    print(f'No problems available in category {category}')
    return None # Return None explicitly if no problem is found

# Return a list of categories that are unlocked based on user progress
def get_unlocked_categories(session: Session) -> List[str]:
    problems = session.exec(select(Problem)).all()
    attempts = session.exec(select(UserAttempt)).all()

    solved_counts = defaultdict(lambda: defaultdict(int))
    problem_map = {p.id: p for p in problems}

    for attempt in attempts:
        if attempt.solved and attempt.problem_id in problem_map:
            problem = problem_map[attempt.problem_id]
            solved_counts[problem.neetcode_category][problem.difficulty] += 1

    # count total problems by category and difficulty
    total_counts = defaultdict(lambda: defaultdict(int))
    for problem in problems:
        total_counts[problem.neetcode_category][problem.difficulty] += 1

    unlocked_categories = []

    # First and second categories are always unlocked
    category_keys = list(CATEGORY_PRIORITY.keys())
    if category_keys:
        unlocked_categories.append(category_keys[0])
        unlocked_categories.append(category_keys[1])

    # Check the remaining categories
    for i in range(2, len(category_keys)):
        prev_category = category_keys[i-1]
        current_category = category_keys[i]

        prev_easy_solved = solved_counts[prev_category]["Easy"]
        prev_easy_total = total_counts[prev_category]["Easy"]

        category_threshold = CATEGORY_PRIORITY.get(prev_category, MASTERY_THRESHOLD)

        # Unlock if we'eve solved enough easy problems or all the easy problems
        if (prev_easy_solved >= category_threshold or
            (prev_easy_total > 0 and prev_easy_solved == prev_easy_total)):
            unlocked_categories.append(current_category)
        else:
            # Once we hit one locked category, we stop
            break

    print(f'Unlocked categories: {unlocked_categories}')
    return unlocked_categories

#
def get_categories_with_pending_reviews(session: Session) -> List[str]:
    now = datetime.now()

    # Get all the problem memories that are due for review
    memories = session.exec(
        select(ProblemMemory).where(ProblemMemory.next_review_date <= now)
    ).all()

    if not memories:
        return []
    
    # Get problems associated with the due memories
    due_problem_ids = [m.problem_id for m in memories]
    due_problems = session.exec(
        select(Problem).where(Problem.id.in_(due_problem_ids))
    ).all()

    due_categories = list(set(p.neetcode_category for p in due_problems))

    print(f'Categories with pending reviews: {due_categories}')
    return due_categories

# Return a list of problems that are due for review based on SM-2 algorithm
def get_due_problems(category: str, session: Session) -> List[Problem]:
    now = datetime.now()

    # Get all the problems in the category
    category_problems = session.exec(
        select(Problem).where(Problem.neetcode_category == category)
    ).all()

    category_problem_ids = [p.id for p in category_problems]

    # Get all the problem memories that are due for review
    due_memories = session.exec(
        select(ProblemMemory).where(
            ProblemMemory.problem_id.in_(category_problem_ids),
            ProblemMemory.next_review_date <= now
        )
    ).all()

    due_problems_ids = [m.problem_id for m in due_memories]

    # Get the actual problems due for review
    due_problems = [p for p in category_problems if p.id in due_problems_ids]

    print(f'Due problems for category {category}: {[p.title for p in due_problems]}')
    return due_problems

# Return a list of unsolved problems in the category
def get_unsolved_problems(category: str, session: Session) -> List[Problem]:
    # Get all the problems in the category
    category_problems = session.exec(
        select(Problem).where(Problem.neetcode_category == category)
    ).all()

    # Get solved problem IDs
    solved_attempts = session.exec(
        select(UserAttempt).where(UserAttempt.solved == True)
    ).all()

    solved_problem_ids = set(a.problem_id for a in solved_attempts)

    # Filter out the unsolved problems
    unsolved_problems = [p for p in category_problems if p.id not in solved_problem_ids]
    
    print(f'Unsolved problems for category {category}: {[p.title for p in unsolved_problems]}')
    return unsolved_problems

# Fallback if no problems are available
def get_fallback_problem(session: Session) -> Problem:
    solved_attempts = session.exec(
        select(UserAttempt).where(UserAttempt.solved == True)
    ).all()

    solved_problem_ids = set(a.problem_id for a in solved_attempts)

    all_problems = session.exec(select(Problem)).all()
    unsolved_problems = [p for p in all_problems if p.id not in solved_problem_ids]

    if unsolved_problems:
        return random.choice(unsolved_problems)
    
    # If all problems are solved, return any problem
    if all_problems:
        return random.choice(all_problems)
    
    raise ValueError("No problems available in the database")

# Process a problem attempt, updating SM-2 category rewards.
def process_attempt(problem_id: int, solved: bool, time_taken: float) -> None:
    global INTERLEAVING_FACTOR, category_rewards, category_attempts, total_attempts

    # Update SM-2
    quality = estimate_quality(solved, time_taken)
    
    with Session(engine) as session:
        # Get the problem and its category
        problem = session.exec(select(Problem).where(Problem.id == problem_id)).first()
        if not problem:
            raise ValueError(f"Problem with ID {problem_id} not found")
        
        category = problem.neetcode_category

        # Update or create a memory record
        memory = session.exec(
            select(ProblemMemory).where(ProblemMemory.problem_id == problem_id)
        ).first()

        if not memory:
            memory = ProblemMemory(problem_id=problem_id)
            session.add(memory)
            session.commit()
            session.refresh(memory)

        # Update memory using SM-2
        updated_memory = update_sm2(memory, quality)
        session.add(updated_memory)
        session.commit()

        # Update category rewards (0-1)
        reward = quality / 5.0

        # Exponential moving average
        learning_rate = 0.1
        if category in category_rewards:
            category_rewards[category] = (1 - learning_rate) * category_rewards[category] + learning_rate * reward
        else:
            category_rewards[category] = reward
        
        # Update attempt counts
        category_attempts[category] = category_attempts.get(category, 0) + 1
        total_attempts += 1

        # Graudally increase interleaving factor
        INTERLEAVING_FACTOR = min(INTERLEAVING_FACTOR + INTERLEAVING_INCREASE_RATE, 0.7)

        print(f'Updated reward for {category}: {category_rewards[category]:.4f}')
        print(f'Interleaving factor: {INTERLEAVING_FACTOR:.4f}')

# Update the problem's memory using SM-2 algorithm
def update_sm2(memory: ProblemMemory, quality: int) -> ProblemMemory:
    easiness_factor = memory.easiness_factor
    repetitions = memory.repetitions
    last_interval = memory.last_interval

    # SM-2 easiness factor update
    easiness_factor += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    easiness_factor = max(easiness_factor, 1.3)

    if quality < 3:
        repetitions = 0
        interval_delays = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval_delays = 1
        elif repetitions == 2:
            interval_delays = 6
        else:
            interval_delays = round(last_interval * easiness_factor)

    memory.easiness_factor = easiness_factor
    memory.repetitions = repetitions
    memory.last_interval = interval_delays
    memory.last_attempt_date = datetime.now()
    memory.next_review_date = memory.last_attempt_date + timedelta(
        days=round(interval_delays)
    )

    return memory

# Estimate the quality score for SM-2 based on the success and time taken
def estimate_quality(solved: bool, time_taken: float) -> int:
    if not solved:
        return 2 if time_taken < 30 else 1
    else:
        return 5 if time_taken < 20 else 4
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
INTERLEAVING_FACTOR = 0.7
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
                # Select a random category with a slight preference for less reviewed categories
                categories = list(review_by_category.keys())
                category_counts = {categ: len(probs) for categ, probs in review_by_category.items()}
                # Inverse weighting: fewer problems = higher chances
                total = sum(1/count for count in category_counts.values())
                weights = [1/(category_counts[categ]*total) for categ in categories]
                selected_category = random.choices(categories, weights=weights)[0]
                print(f'Review interleaving: Selected category {selected_category} from {len(categories)} options')
                return random.choice(review_by_category[selected_category])
            else:
                print(f'Recommending review problem from {len(review_problems)} due.')
                return random.choice(review_problems)
            
        # Calculate progress for each category
        solved_counts_per_category = defaultdict(lambda: defaultdict(int))
        total_problems_per_category = defaultdict(lambda: defaultdict(int))

        # Count the number of solved problems and total problems in each category and difficulty
        for problem in problems:
            total_problems_per_category[problem.neetcode_category][problem.difficulty] += 1

        for attempt in attempts:
            if attempt.solved and attempt.problem_id in problem_map:
                problem = problem_map[attempt.problem_id]
                solved_counts_per_category[problem.neetcode_category][problem.difficulty] += 1

        # Find New problems
        new_problems = [
            p for p in problems
            if p.id not in completed_ids and p.id not in review_problem_ids
        ]

        # Find unlocked categories
        unlocked_categories = {}
        active_categories = []

        for indx, category in enumerate(CATEGORY_PRIORITY):
            # The first category is always unlocked
            if indx == 0:
                unlocked_categories[category] = True
                active_categories.append(category)
                continue

            # Check if previous category has enough experience to unlock the next one
            previous_category = CATEGORY_PRIORITY[indx - 1]
            easy_mastery_in_previous = solved_counts_per_category[previous_category]['Easy']
            total_easy_in_previous = total_problems_per_category[previous_category]['Easy']

            # Unlock current category if we have enough master or all easy problems are solved
            if (easy_mastery_in_previous >= MASTERY_THRESHOLD or (total_easy_in_previous > 0 and easy_mastery_in_previous == total_easy_in_previous)):
                unlocked_categories[category] = True
                active_categories.append(category)
            else:
                unlocked_categories[category] = False

        # Verify at least two categories are always active
        if len(active_categories) < 2 and len(CATEGORY_PRIORITY) > 1:
            for category in CATEGORY_PRIORITY:
                if category not in active_categories:
                    unlocked_categories[category] = True
                    active_categories.append(category)
                    print(f'Forcibly unlocked {category} to ensure interleaving')
                    break
        
        print(f'Unlocked categories: {[c for c in unlocked_categories if unlocked_categories[c]]}')
        print(f'Active categories for interleaving: {active_categories}')

        # Apply Dynamic Interleaving
        # FALL BACK FOR NO ACTIVE CATEGORIES
        if not active_categories:
            print('No active categories found. Using fallback')
            if new_problems:
                return random.choice(new_problems)
            elif problems:
                return random.choice(problems)
            else:
                raise ValueError('No problems found in the database')
            
        # Choose a set of categories to consider
        num_categories = min(MAX_INTERLEAVING_CATEGORIES, len(active_categories))
        num_categories = max(num_categories, 2) if len(active_categories) >= 2 else 1

        # Apply weithed selection for categories using:
        # Higher weight for position in earlier learning path (Master the first ones first)
        # Higher weight for the ones that haven't been solved as ofter
        category_weights = []

        for category in active_categories:
            # Weight for how early they appear
            position_indx = CATEGORY_PRIORITY.index(category)
            position_weight = math.exp(-0.2 * position_indx)

            # Weight for how many they have solved
            easy_solved = solved_counts_per_category[category]['Easy']
            easy_total = total_problems_per_category[category]['Easy']
            master_ratio = easy_solved / max(1, easy_total)
            master_weight = 1 - 0.7 * master_ratio

            combined_weight = position_weight * master_weight
            category_weights.append(combined_weight)

        # Select a diversified set of categories
        selected_categories = random.choices(
            active_categories,
            weights=category_weights,
            k=num_categories
        )

        print(f'Selected categories for interleaving: {selected_categories}')

        # Select Problems from interleaving categories
        candidate_problems = []

        for category in selected_categories:
            category_problems = [p for p in new_problems if p.neetcode_category == category]
            if not category_problems:
                continue

            # Determine eligible difficulties for this category
            easy_solved = solved_counts_per_category[category]['Easy']
            easy_total = total_problems_per_category[category]['Easy']

            eligible_diffic = []

            # Always consider easy problems
            easy_problems = [p for p in category_problems if p.difficulty == 'Easy']
            if easy_problems:
                eligible_diffic.append('Easy')

            # Consider Medium/Hard if enough Easy problems solved
            if easy_solved >= MASTERY_THRESHOLD or (easy_total > 0 and easy_solved == easy_total):
                medium_problems = [p for p in category_problems if p.difficulty == 'Medium']
                if medium_problems:
                    eligible_diffic.append('Medium')

                # Only consider hard if we ran out of easy/medium or if we have a high mastery level
                hard_problems = [p for p in category_problems if p.difficulty == 'Hard']
                if (not easy_problems and not medium_problems) or easy_solved >= 2*MASTERY_THRESHOLD:
                    if hard_problems:
                        eligible_diffic.append('Hard')

            # Select all the probelms of eligible difficulty
            for difficulty in eligible_diffic:
                difficulty_problems = [p for p in category_problems if p.difficulty == difficulty]
                if difficulty_problems:
                    candidate_problems.extend(difficulty_problems)

        # Make the final choice
        if candidate_problems:
            problems_by_category = defaultdict(list)
            for problem in candidate_problems:
                problems_by_category[problem.neetcode_category].append(problem)

            # Select a category with true interleaving (doesn't always choose the same category)
            categories = list(problems_by_category.keys())
            print(f'Final Categories: {categories}')

            if len(categories) > 1:
                selected_category = random.choice(categories)
                print(f'Final interleaving category: {selected_category}')
                return random.choice(problems_by_category[selected_category])
            else:
                print('Choosing random candidate problem')
                return random.choice(candidate_problems)
            
        # FALL BACK
        print('No active categories found. Using fallback')
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

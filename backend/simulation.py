# simulation.py (Additions/Modifications)
import random
import csv
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Import necessary components from your existing code
from sqlmodel import Session, create_engine, select, SQLModel
from models import Problem, UserAttempt, ProblemMemory # Assuming models.py is accessible
from recommender_v5 import (
    recommend_ordered_problem,
    process_attempt,
    CATEGORY_PRIORITY,
    DIFFICULTY_PRIORITY,
    engine, # Use the engine defined in recommender_v5
    # Import globals we need to reset/access
    category_rewards,
    category_attempts,
    total_attempts,
    INTERLEAVING_FACTOR,
    INTERLEAVING_INCREASE_RATE,
    EPSILON
)
import recommender_v5 # Import the module itself to modify its globals

# --- Simulation Constants ---
# LOG_FILE = 'simulation_log.csv' # We'll make this dynamic per run
NUM_SIMULATION_STEPS = 200 # How many problems the agent attempts
AGENT_LEARNING_RATE = 0.05 # How much skill increases on success
# AGENT_INITIAL_SKILL = 0.1 # Replaced by profiles

# --- Agent Profiles with per-category skill levels ---
AGENT_PROFILES = {
    "Freshman": {
        "Arrays and Hashing": 0.1,
        "Two Pointers": 0.05,
        "Stack": 0.05,
        "Sliding Window": 0.0,
        "Binary Search": 0.05,
        "Linked List": 0.0,
        "Tree": 0.0,
        "Tries": 0.0,
        "Heap / Priority Queue": 0.0,
        "Backtracking": 0.0,
        "Intervals": 0.0,
        "Greedy": 0.0,
        "Graph": 0.0,
        "Graph2": 0.0,
        "1DDP": 0.0,
        "2DDP": 0.0,
        "Bit Manipulation": 0.0,
        "Math": 0.1,
    },
    "Sophomore": {
        "Arrays and Hashing": 0.4,
        "Two Pointers": 0.4,
        "Stack": 0.4,
        "Sliding Window": 0.35,
        "Binary Search": 0.4,
        "Linked List": 0.3,
        "Tree": 0.25,
        "Tries": 0.2,
        "Heap / Priority Queue": 0.2,
        "Backtracking": 0.25,
        "Intervals": 0.3,
        "Greedy": 0.3,
        "Graph": 0.2,
        "Graph2": 0.2,
        "1DDP": 0.15,
        "2DDP": 0.1,
        "Bit Manipulation": 0.2,
        "Math": 0.3,
    },
    "Senior": {
        "Arrays and Hashing": 0.7,
        "Two Pointers": 0.7,
        "Stack": 0.65,
        "Sliding Window": 0.65,
        "Binary Search": 0.7,
        "Linked List": 0.6,
        "Tree": 0.6,
        "Tries": 0.55,
        "Heap / Priority Queue": 0.55,
        "Backtracking": 0.6,
        "Intervals": 0.6,
        "Greedy": 0.6,
        "Graph": 0.5,
        "Graph2": 0.5,
        "1DDP": 0.4,
        "2DDP": 0.35,
        "Bit Manipulation": 0.5,
        "Math": 0.6,
    },
    "Senior_Variant1": {
        "Arrays and Hashing": 0.7,
        "Two Pointers": 0.7,
        "Stack": 0.65,
        "Sliding Window": 0.65,
        "Binary Search": 0.7,
        "Linked List": 0.6,
        "Tree": 0.6,
        "Tries": 0.55,
        "Heap / Priority Queue": 0.55,
        "Backtracking": 0.2,  # Weak in Backtracking
        "Intervals": 0.6,
        "Greedy": 0.6,
        "Graph": 0.2,  # Weak in Graph
        "Graph2": 0.2,  # Weak in Graph2
        "1DDP": 0.1,  # Weak in 1DDP
        "2DDP": 0.35,
        "Bit Manipulation": 0.5,
        "Math": 0.6,
    },
    "Senior_Variant2": {
        "Arrays and Hashing": 0.7,
        "Two Pointers": 0.7,
        "Stack": 0.65,
        "Sliding Window": 0.2,  # Weak in Sliding Window
        "Binary Search": 0.2,  # Weak in Binary Search
        "Linked List": 0.6,
        "Tree": 0.6,
        "Tries": 0.15,  # Weak in Tries
        "Heap / Priority Queue": 0.55,
        "Backtracking": 0.6,
        "Intervals": 0.6,
        "Greedy": 0.6,
        "Graph": 0.5,
        "Graph2": 0.5,
        "1DDP": 0.4,
        "2DDP": 0.1,  # Weak in 2DDP
        "Bit Manipulation": 0.5,
        "Math": 0.6,
    },
    "Senior_Variant3": {
        "Arrays and Hashing": 0.7,
        "Two Pointers": 0.7,
        "Stack": 0.65,
        "Sliding Window": 0.65,
        "Binary Search": 0.7,
        "Linked List": 0.6,
        "Tree": 0.15,  # Weak in Tree
        "Tries": 0.55,
        "Heap / Priority Queue": 0.15,  # Weak in Heap / Priority Queue
        "Backtracking": 0.6,
        "Intervals": 0.2,  # Weak in Intervals
        "Greedy": 0.6,
        "Graph": 0.5,
        "Graph2": 0.5,
        "1DDP": 0.4,
        "2DDP": 0.35,
        "Bit Manipulation": 0.2,  # Weak in Bit Manipulation
        "Math": 0.6,
    }
}


# Map difficulty strings to numerical values for comparison with skill
DIFFICULTY_VALUES = {
    "Easy": 0.3,
    "Medium": 0.6,
    "Hard": 0.9
}

# Define categories for the agent based on the recommender's priorities
AGENT_CATEGORIES = list(CATEGORY_PRIORITY.keys())

# --- Utility Functions ---

def map_difficulty_to_value(difficulty: str) -> float:
    """Maps difficulty string to a numerical value."""
    return DIFFICULTY_VALUES.get(difficulty, 0.5) # Default to medium if unknown

def reset_recommender_state():
    """Resets the global state variables in recommender_v5."""
    print("Resetting recommender global state...")
    recommender_v5.category_rewards = {cat: 1.0 for cat in AGENT_CATEGORIES}
    recommender_v5.category_attempts = {cat: 0 for cat in AGENT_CATEGORIES}
    recommender_v5.total_attempts = 0
    recommender_v5.INTERLEAVING_FACTOR = 0.3 # Reset to initial value
    # Note: EPSILON and INTERLEAVING_INCREASE_RATE are constants, no need to reset

def reset_simulation_db_state(engine):
    """Clears UserAttempt and ProblemMemory tables for a fresh simulation."""
    print("Resetting simulation database state (UserAttempt, ProblemMemory)...")
    # Ensure tables exist before trying to delete from them
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        # Delete records carefully
        session.query(UserAttempt).delete()
        session.query(ProblemMemory).delete()
        session.commit()
    print("Database state reset.")

def get_all_unsolved_problems(session: Session) -> List[Problem]:
    """Gets all problems not marked as solved in UserAttempt for this session run."""
    # Note: Since we reset the DB for each run, this effectively gets problems
    # not solved *within the current simulation run*.
    solved_attempts = session.exec(
        select(UserAttempt).where(UserAttempt.solved == True)
    ).all()
    solved_problem_ids = set(a.problem_id for a in solved_attempts)

    all_problems = session.exec(select(Problem)).all()
    unsolved_problems = [p for p in all_problems if p.id not in solved_problem_ids]
    return unsolved_problems

def recommend_random_problem(session: Session) -> Optional[Problem]:
    """Selects a random unsolved problem from the entire database."""
    unsolved_problems = get_all_unsolved_problems(session)

    if unsolved_problems:
        chosen = random.choice(unsolved_problems)
        print(f"Random Baseline selected: {chosen.title} (ID: {chosen.id})")
        return chosen
    else:
        # If all problems are somehow solved (unlikely in short simulation), pick any
        all_problems = session.exec(select(Problem)).all()
        if all_problems:
            chosen = random.choice(all_problems)
            print(f"Random Baseline (all solved): {chosen.title} (ID: {chosen.id})")
            return chosen
        else:
            print("Random Baseline: No problems found in DB.")
            return None # Should not happen if DB is seeded


# simulation.py (Agent class modifications)

class Agent:
    """
    Represents a simulated LeetCode learner with different profiles.
    """
    def __init__(self, agent_id: int, profile_name: str, learning_rate: float = AGENT_LEARNING_RATE):
        self.agent_id = agent_id
        self.profile = profile_name
        if profile_name not in AGENT_PROFILES:
            raise ValueError(f"Unknown agent profile: {profile_name}. Valid profiles: {list(AGENT_PROFILES.keys())}")

        profile_skill = AGENT_PROFILES[profile_name]
        # Simple knowledge: skill level (0.0 to 1.0) per category
        self.skills: Dict[str, float] = {
            category: profile_skill.get(category, 0.0) for category in AGENT_CATEGORIES
        }
        self.learning_rate = learning_rate
        self.problems_solved = 0
        self.problems_attempted = 0
        # Track solved problems internally for the random baseline logic if needed,
        # but relying on DB reset and query is simpler for now.
        # self.solved_problem_ids = set()

    def get_skill(self, category: str) -> float:
        """Gets skill for a category, defaulting based on profile."""
        return self.skills.get(category, AGENT_PROFILES[self.profile])

    def attempt_problem(self, problem: Problem) -> Tuple[bool, float]:
        """
        Simulates the agent attempting a problem.
        Returns (solved: bool, time_taken_minutes: float).
        (Keep the existing stochastic logic from the previous version)
        """
        self.problems_attempted += 1
        category = problem.neetcode_category
        difficulty_value = map_difficulty_to_value(problem.difficulty)
        agent_skill = self.get_skill(category)

        # --- Stochastic Solving Logic ---
        solve_chance = agent_skill - difficulty_value + 0.5 + random.uniform(-0.2, 0.2)
        solve_chance = max(0.0, min(1.0, solve_chance)) # Clamp between 0 and 1
        solved = random.random() < solve_chance

        # --- Simulate Time Taken ---
        base_time = {"Easy": 15, "Medium": 30, "Hard": 45}.get(problem.difficulty, 30)
        time_taken = base_time

        # Adjust time based on solve_chance (same logic as before)
        if solved:
            self.problems_solved += 1
            if solve_chance > 0.75: time_taken *= random.uniform(0.5, 0.8)
            else: time_taken *= random.uniform(0.8, 1.2)
            time_taken = max(5, time_taken) # Minimum 5 minutes
            # self.solved_problem_ids.add(problem.id) # Track if needed internally
            print(f"Agent {self.agent_id} ({self.profile}): SOLVED {problem.title} (Cat: {category}, Diff: {problem.difficulty}, Skill: {agent_skill:.2f}, Chance: {solve_chance:.2f}) -> Time: {time_taken:.1f} min")
        else:
            if solve_chance < 0.25: time_taken *= random.uniform(1.2, 1.6)
            else: time_taken *= random.uniform(0.9, 1.3)
            time_taken = min(60, time_taken) # Max 60 minutes
            print(f"Agent {self.agent_id} ({self.profile}): FAILED {problem.title} (Cat: {category}, Diff: {problem.difficulty}, Skill: {agent_skill:.2f}, Chance: {solve_chance:.2f}) -> Time: {time_taken:.1f} min")

        return solved, round(time_taken, 2)


    def update_skill(self, category: str, solved: bool):
        """
        Updates the agent's skill in a category based on attempt outcome.
        (Keep the existing logic)
        """
        if category not in self.skills:
             print(f"Warning: Category '{category}' not found in agent's known categories. Initializing using profile default.")
             self.skills[category] = AGENT_PROFILES[self.profile] # Use profile default

        if solved:
            old_skill = self.skills[category]
            self.skills[category] = min(1.0, self.skills[category] + self.learning_rate)
            # print(f"Agent {self.agent_id} ({self.profile}): Skill for {category} increased from {old_skill:.3f} to {self.skills[category]:.3f}")
        # No change on failure for simplicity

    def get_knowledge_state(self) -> Dict[str, float]:
        """Returns the agent's current skill levels."""
        return self.skills.copy()


# simulation.py (run_simulation modifications)

def run_simulation(
    num_steps: int,
    agent_profile: str,
    agent_type: str, # "recommender" or "random_baseline"
    log_filename: str,
    agent_id: int = 0, # Keep simple with one agent per run for now
):
    """
    Runs the recommendation and agent interaction simulation for a specific configuration.
    """
    print(f"\n===== Starting Simulation: Profile={agent_profile}, Type={agent_type} =====")

    # --- Setup ---
    # Crucially, reset state BEFORE each simulation run
    reset_simulation_db_state(engine) # Clear UserAttempt, ProblemMemory
    if agent_type == "recommender":
        reset_recommender_state() # Reset rewards ONLY for recommender runs

    agent = Agent(agent_id=agent_id, profile_name=agent_profile)

    # Prepare log file (append mode in case run_all_simulations uses one file)
    log_file_exists = os.path.exists(log_filename)
    with open(log_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'step', 'agent_id', 'agent_profile', 'agent_type',
            'problem_id', 'problem_title', 'problem_category', 'problem_difficulty',
            'agent_skill_before', 'solved', 'time_taken', 'agent_skill_after',
            'recommender_reward', 'interleaving_factor', 'total_attempts_recommender' # Note: these are only relevant for agent_type='recommender'
        ]
        for cat in AGENT_CATEGORIES: fieldnames.append(f'agent_skill_{cat.replace(" ", "_")}')
        for cat in AGENT_CATEGORIES: fieldnames.append(f'recommender_reward_{cat.replace(" ", "_")}') # Logged even for random, but won't change

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header only if file is new or empty
        if not log_file_exists or csvfile.tell() == 0:
            writer.writeheader()

        # --- Simulation Loop ---
        with Session(engine) as session: # Keep session open for the loop
            for step in range(num_steps):
                # print(f"--- Sim Step {step + 1}/{num_steps} ({agent_profile}/{agent_type}) ---") # Verbose

                # 1. Get recommendation (depends on agent_type)
                recommended_problem: Optional[Problem] = None
                if agent_type == "recommender":
                    recommended_problem = recommend_ordered_problem() # Uses the global recommender state
                elif agent_type == "random_baseline":
                    recommended_problem = recommend_random_problem(session) # Uses session to find unsolved
                else:
                    raise ValueError(f"Unknown agent_type: {agent_type}")

                if not recommended_problem:
                    print(f"Warning ({agent_profile}/{agent_type}): No problem recommended/found. Skipping step {step + 1}.")
                    continue

                # Ensure problem attributes are valid before proceeding
                if not all([recommended_problem.id, recommended_problem.title,
                            recommended_problem.neetcode_category, recommended_problem.difficulty]):
                     print(f"Warning ({agent_profile}/{agent_type}): Recommended problem missing essential data (ID: {recommended_problem.id}). Skipping.")
                     continue

                problem_id = recommended_problem.id
                category = recommended_problem.neetcode_category
                difficulty = recommended_problem.difficulty
                problem_title = recommended_problem.title

                # Log state *before* attempt
                skill_before = agent.get_skill(category)
                # Get recommender state (will be stale/initial for random agent)
                current_recommender_reward = recommender_v5.category_rewards.get(category, 0.0)
                current_interleaving = recommender_v5.INTERLEAVING_FACTOR
                current_total_attempts = recommender_v5.total_attempts

                # 2. Agent attempts the problem
                solved, time_taken = agent.attempt_problem(recommended_problem)

                # 3. Report attempt *only* if using the recommender
                # Also save the attempt to DB for *both* types (to track solved status for random agent)
                attempt = UserAttempt(
                    problem_id=problem_id,
                    solved=solved,
                    time_taken_minutes=time_taken,
                    # Add other fields if needed, defaults are likely okay
                )
                session.add(attempt)
                session.commit() # Commit attempt so random selector sees it as solved next time

                if agent_type == "recommender":
                    # This updates recommender globals (rewards, interleaving) and ProblemMemory
                    process_attempt(problem_id=problem_id, solved=solved, time_taken=time_taken)
                # else: Baseline agent does not interact with recommender state or ProblemMemory

                # 4. Agent updates its internal skill
                agent.update_skill(category, solved)
                skill_after = agent.get_skill(category)

                # 5. Log results
                log_data = {
                    'step': step + 1,
                    'agent_id': agent.agent_id,
                    'agent_profile': agent_profile,
                    'agent_type': agent_type,
                    'problem_id': problem_id,
                    'problem_title': problem_title,
                    'problem_category': category,
                    'problem_difficulty': difficulty,
                    'agent_skill_before': round(skill_before, 4),
                    'solved': solved,
                    'time_taken': time_taken,
                    'agent_skill_after': round(skill_after, 4),
                    # Log recommender state; will reflect updates only for 'recommender' type
                    'recommender_reward': round(recommender_v5.category_rewards.get(category, 0.0), 4),
                    'interleaving_factor': round(recommender_v5.INTERLEAVING_FACTOR, 4),
                    'total_attempts_recommender': recommender_v5.total_attempts
                }
                agent_skills_snapshot = agent.get_knowledge_state()
                # Use a copy of rewards at this point in time
                recommender_rewards_snapshot = recommender_v5.category_rewards.copy()
                for cat in AGENT_CATEGORIES:
                    log_data[f'agent_skill_{cat.replace(" ", "_")}'] = round(agent_skills_snapshot.get(cat, AGENT_PROFILES[agent_profile]), 4)
                    # Log the state of recommender rewards, even if it wasn't updated this step (for random baseline)
                    log_data[f'recommender_reward_{cat.replace(" ", "_")}'] = round(recommender_rewards_snapshot.get(cat, 1.0), 4) # Default reward is 1.0

                writer.writerow(log_data)
                csvfile.flush() # Ensure data is written immediately

    print(f"===== Simulation Complete: Profile={agent_profile}, Type={agent_type} =====")
    print(f"Log saved to {log_filename}")
    final_agent_state = agent.get_knowledge_state()
    print(f"Final Agent Skills ({agent_profile}/{agent_type}):")
    # for cat, skill in sorted(final_agent_state.items()):
    #     print(f"  {cat}: {skill:.3f}")
    print(f"  Problems Solved: {agent.problems_solved} / {agent.problems_attempted}")

    # if agent_type == 'recommender':
    #     final_recommender_rewards = recommender_v5.category_rewards
    #     print("Final Recommender Category Rewards:")
    #     for cat, reward in sorted(final_recommender_rewards.items()):
    #         print(f"  {cat}: {reward:.3f}")


# simulation.py (New Orchestration Function)

def run_all_simulations(num_steps: int, base_log_filename: str = "simulation_log"):
    """Runs simulations for all profiles and the baseline."""

    # Define configurations to run
    configurations = []
    # Recommender runs
    for profile in AGENT_PROFILES.keys():
        configurations.append({
            "profile": profile,
            "type": "recommender",
            "logfile": f"{base_log_filename}_{profile}_recommender.csv"
        })
    # Baseline runs (run for each profile for direct comparison)
    for profile in AGENT_PROFILES.keys():
        configurations.append({
            "profile": profile,
            "type": "random_baseline",
            "logfile": f"{base_log_filename}_{profile}_random.csv"
        })

    # Or, a simpler baseline comparison:
    # configurations.append({
    #     "profile": "Beginner", # Use one profile for baseline usually
    #     "type": "random_baseline",
    #     "logfile": f"{base_log_filename}_Beginner_random.csv"
    # })

    # --- Database Pre-check ---
    if not os.path.exists("./database.db"):
        print("Error: database.db not found. Please create and seed it first.")
        # Example:
        # print("Attempting to create and seed database...")
        # import db # Assuming db.py with create_db_and_tables, seed_database
        # db.create_db_and_tables()
        # db.seed_database()
        # print("Database created and seeded.")
        # if not os.path.exists("./database.db"): # Check again
        #      exit()
        exit() # Exit if DB still not found


    # --- Run Simulations ---
    all_log_files = []
    for config in configurations:
        log_file = config["logfile"]
        all_log_files.append(log_file)
        # Clear the specific log file for this run
        if os.path.exists(log_file):
            print(f"Removing old log file: {log_file}")
            os.remove(log_file)

        run_simulation(
            num_steps=num_steps,
            agent_profile=config["profile"],
            agent_type=config["type"],
            log_filename=log_file
        )
        print("-" * 30) # Separator between runs

    print("\n======= All Simulations Complete =======")
    print("Logs generated:")
    for fname in all_log_files:
        print(f" - {fname}")

    # --- Optional: Combined Analysis ---
    # You could load all CSVs into pandas here for a combined analysis
    try:
        import pandas as pd
        all_dfs = []
        for fname in all_log_files:
             if os.path.exists(fname):
                 df = pd.read_csv(fname)
                 all_dfs.append(df)
             else:
                 print(f"Warning: Log file {fname} not found for analysis.")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print("\n--- Basic Combined Analysis ---")
            print(f"Total steps logged across all runs: {len(combined_df)}")

            # Compare solve rates
            solve_summary = combined_df.groupby(['agent_profile', 'agent_type'])['solved'].agg(['mean', 'sum', 'count'])
            print("\nSolve Rate Summary:")
            print(solve_summary)

            # Example: Plot average skill progression
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(12, 7))
            # for name, group in combined_df.groupby(['agent_profile', 'agent_type']):
            #     # Calculate average skill across all categories per step
            #     skill_cols = [f'agent_skill_{cat.replace(" ", "_")}' for cat in AGENT_CATEGORIES]
            #     group['average_skill'] = group[skill_cols].mean(axis=1)
            #     group.groupby('step')['average_skill'].mean().plot(ax=ax, label=f'{name[0]} ({name[1]})')

            # ax.set_title('Average Agent Skill Progression Over Time')
            # ax.set_xlabel('Simulation Step')
            # ax.set_ylabel('Average Skill Level')
            # ax.legend()
            # ax.grid(True)
            # plt.tight_layout()
            # plt.show()

    except ImportError:
        print("\nInstall pandas ('pip install pandas') and matplotlib ('pip install matplotlib') for combined analysis and plotting.")
    except Exception as e:
         print(f"\nError during analysis: {e}")


# simulation.py (main execution block update)

if __name__ == "__main__":
    # Run all simulation configurations
    run_all_simulations(
        num_steps=NUM_SIMULATION_STEPS,
        base_log_filename="simulation_log" # Files like simulation_log_Beginner_recommender.csv will be created
    )

    # The analysis part is now inside run_all_simulations
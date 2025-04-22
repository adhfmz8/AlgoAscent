#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
LOG_DIR = '.'  # Directory where the CSV log files are located
FILE_PATTERN = 'simulation_log_*.csv' # Pattern to find the log files

# Define expected profiles and types for consistent reporting
# (Should match those used in the simulation)
PROFILES = ['Beginner', 'Moderate', 'Expert']
AGENT_TYPES = ['recommender', 'random_baseline']

# Initial skill levels for each profile (used for calculating skill change)
# (Should match AGENT_PROFILES in simulation.py)
AGENT_INITIAL_SKILLS = {
    "Beginner": 0.1,
    "Moderate": 0.35,
    "Expert": 0.6,
}

# List of categories used for average skill calculation
# (Should match AGENT_CATEGORIES in simulation.py)
# Define manually if simulation.py is not easily importable
AGENT_CATEGORIES = [
    "Arrays and Hashing", "Two Pointers", "Stack", "Sliding Window", "Binary Search",
    "Linked List", "Tree", "Tries", "Heap / Priority Queue", "Backtracking",
    "Intervals", "Greedy", "Graph", "Graph2", "1DDP", "2DDP",
    "Bit Manipulation", "Math"
]
SKILL_COLS = [f'agent_skill_{cat.replace(" ", "_")}' for cat in AGENT_CATEGORIES]
# --- End Configuration ---


def analyze_and_print_results(log_dir, file_pattern):
    """
    Loads simulation data, analyzes it, and prints key results in plain text.
    """
    print("--- Starting Simulation Analysis ---")

    # 1. Load Data
    all_files = glob.glob(os.path.join(log_dir, file_pattern))
    if not all_files:
        print(f"\nERROR: No files found matching pattern '{file_pattern}' in directory '{log_dir}'")
        return

    print(f"Found {len(all_files)} log files:")
    # for f in all_files: print(f" - {os.path.basename(f)}")

    li = []
    for filename in all_files:
        try:
            df_temp = pd.read_csv(filename)
            # Basic check for essential columns
            if not all(col in df_temp.columns for col in ['step', 'agent_profile', 'agent_type', 'solved']):
                 print(f"WARNING: Skipping file {os.path.basename(filename)} - missing essential columns.")
                 continue
            li.append(df_temp)
        except Exception as e:
            print(f"WARNING: Error loading {os.path.basename(filename)}: {e}. Skipping.")

    if not li:
        print("\nERROR: No data loaded successfully. Exiting.")
        return

    df = pd.concat(li, axis=0, ignore_index=True)
    print(f"\nSuccessfully loaded data from {len(li)} files. Total records: {len(df)}")
    print("Profiles and Types found:", df[['agent_profile', 'agent_type']].value_counts().index.tolist())

    # 2. Data Preparation
    if df['solved'].dtype != bool:
        df['solved'] = df['solved'].astype(bool)

    # Calculate average agent skill per row
    existing_skill_cols = [col for col in SKILL_COLS if col in df.columns]
    if not existing_skill_cols:
        print("\nWARNING: No skill columns found in the DataFrame. Cannot calculate average skill.")
        df['average_skill'] = np.nan
    else:
        if len(existing_skill_cols) < len(SKILL_COLS):
            print(f"\nWARNING: Calculating average skill based on {len(existing_skill_cols)} found columns (expected {len(SKILL_COLS)}).")
        df['average_skill'] = df[existing_skill_cols].mean(axis=1)

    # Determine the number of simulation steps per run (assuming it's consistent)
    try:
        num_steps = df.groupby(['agent_id', 'agent_profile', 'agent_type'])['step'].max().mode()[0]
        print(f"Detected simulation length: {num_steps} steps per run.")
    except Exception:
        num_steps = df['step'].max()
        print(f"Could not reliably detect steps per run, using overall max steps: {num_steps}.")


    # 3. Calculate Metrics and Print Results

    print("\n\n" + "=" * 50)
    print("       OVERALL PERFORMANCE SUMMARY")
    print("=" * 50 + "\n")

    # --- Solve Rate ---
    print("--- Overall Solve Rate (% of attempts solved) ---")
    solve_rate_summary = df.groupby(['agent_profile', 'agent_type'])['solved'].mean()
    # Ensure all combinations are present, fill missing with NaN
    idx = pd.MultiIndex.from_product([PROFILES, AGENT_TYPES], names=['agent_profile', 'agent_type'])
    solve_rate_summary = solve_rate_summary.reindex(idx)
    print(solve_rate_summary.map('{:.1%}'.format).unstack().to_string(na_rep='N/A'))
    print("-" * 50)

    # --- Total Solved ---
    print(f"--- Total Problems Solved (out of {num_steps} attempts) ---")
    total_solved_summary = df.groupby(['agent_profile', 'agent_type'])['solved'].sum()
    total_solved_summary = total_solved_summary.reindex(idx)
    print(total_solved_summary.unstack().to_string(na_rep='N/A', float_format='%.0f'))
    print("-" * 50)


    print("\n\n" + "=" * 50)
    print("         AGENT SKILL ANALYSIS")
    print("=" * 50 + "\n")

    # --- Final Average Skill ---
    if 'average_skill' in df.columns and not df['average_skill'].isna().all():
        print("--- Final Average Skill Level (at end of simulation) ---")
        # Get the average skill from the last step for each run
        final_skill_rows = df.loc[df.groupby(['agent_id', 'agent_profile', 'agent_type'])['step'].idxmax()]
        final_skill_summary = final_skill_rows.groupby(['agent_profile', 'agent_type'])['average_skill'].mean()
        final_skill_summary = final_skill_summary.reindex(idx)
        print(final_skill_summary.map('{:.3f}'.format).unstack().to_string(na_rep='N/A'))
        print("-" * 50)

        # --- Average Skill Change ---
        print("--- Average Skill Change (Final Skill - Initial Skill) ---")
        # Map initial skills to the summary index for calculation
        initial_skills_series = pd.Series(AGENT_INITIAL_SKILLS, name='initial_skill')
        initial_skills_series.index.name = 'agent_profile'
        # Align initial skills with final skills summary (needs careful index handling)
        final_skill_df = final_skill_summary.reset_index()
        final_skill_df = final_skill_df.merge(initial_skills_series, on='agent_profile', how='left')
        final_skill_df['skill_change'] = final_skill_df['average_skill'] - final_skill_df['initial_skill']
        # Pivot back for display
        skill_change_summary = final_skill_df.pivot(index='agent_profile', columns='agent_type', values='skill_change')
        skill_change_summary = skill_change_summary.reindex(index=PROFILES, columns=AGENT_TYPES) # Ensure order
        print(skill_change_summary.map('{:+.3f}'.format).unstack().unstack().to_string(na_rep='N/A')) # Use unstack twice to match format
        print("-" * 50)

    else:
        print("--- Skill Analysis Skipped (No valid 'average_skill' data) ---")


    print("\n\n" + "=" * 50)
    print("         PROBLEM SELECTION ANALYSIS")
    print("=" * 50 + "\n")

    # --- Difficulty Distribution ---
    if 'problem_difficulty' in df.columns:
        print("--- Distribution of Problem Difficulty Attempted (%) ---")
        difficulty_dist = pd.crosstab(
            [df['agent_profile'], df['agent_type']],
            df['problem_difficulty'],
            normalize='index' # Calculate percentage within each group
        ) * 100
        # Ensure all difficulties and combinations are present
        expected_difficulties = df['problem_difficulty'].unique() # Or define explicitly ['Easy', 'Medium', 'Hard']
        difficulty_dist = difficulty_dist.reindex(index=idx, columns=expected_difficulties, fill_value=0)
        print(difficulty_dist.stack().unstack(level=[1,2]).map('{:.1f}%'.format).to_string(na_rep='N/A'))
        print("-" * 50)
    else:
        print("--- Difficulty Distribution Skipped ('problem_difficulty' column not found) ---")

    # --- Category Distribution (Optional - can be very long) ---
    # print("\n--- Distribution of Categories Attempted (Top 5) (%) ---")
    # if 'problem_category' in df.columns:
    #     category_dist = pd.crosstab(
    #         [df['agent_profile'], df['agent_type']],
    #         df['problem_category'],
    #         normalize='index'
    #     ) * 100
    #     category_dist = category_dist.reindex(idx, fill_value=0)
    #     # Get top 5 most frequent categories overall
    #     top_5_cats = df['problem_category'].value_counts().nlargest(5).index.tolist()
    #     print(category_dist[top_5_cats].stack().unstack(level=[1,2]).map('{:.1f}%'.format).to_string(na_rep='N/A'))
    #     print("-" * 50)
    # else:
    #      print("--- Category Distribution Skipped ('problem_category' column not found) ---")


    print("\n\n" + "=" * 50)
    print("         ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    analyze_and_print_results(LOG_DIR, FILE_PATTERN)
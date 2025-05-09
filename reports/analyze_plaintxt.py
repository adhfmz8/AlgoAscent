#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# --- Configuration ---
LOG_DIR = '.'  # Directory where the CSV log files are located
FILE_PATTERN = './test1/simulation_log_*.csv'  # Pattern to find the log files

# Define expected profiles and types for consistent reporting
PROFILES = ['Freshman', 'Sophomore', 'Senior']
AGENT_TYPES = ['recommender', 'random_baseline']

# Initial skill levels for each profile (used for calculating skill change)
# Note: This matches AGENT_PROFILES in simulation.py
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
    }
}

# List of categories used for average skill calculation
AGENT_CATEGORIES = [
    "Arrays and Hashing", "Two Pointers", "Stack", "Sliding Window", "Binary Search",
    "Linked List", "Tree", "Tries", "Heap / Priority Queue", "Backtracking",
    "Intervals", "Greedy", "Graph", "Graph2", "1DDP", "2DDP",
    "Bit Manipulation", "Math"
]

# Create column names for agent skill and recommender reward
SKILL_COLS = [f'agent_skill_{cat.replace(" ", "_")}' for cat in AGENT_CATEGORIES]
REWARD_COLS = [f'recommender_reward_{cat.replace(" ", "_")}' for cat in AGENT_CATEGORIES]

# --- Utility Functions ---

def calculate_initial_average_skill(profile: str) -> float:
    """Calculate the initial average skill across all categories for a profile."""
    if profile in AGENT_PROFILES:
        profile_skills = AGENT_PROFILES[profile]
        return sum(profile_skills.values()) / len(profile_skills)
    return 0.0

def calculate_initial_skills_df() -> pd.DataFrame:
    """Create a DataFrame with initial skills for all profiles."""
    data = []
    for profile in PROFILES:
        row = {'agent_profile': profile}
        # Add average initial skill
        row['initial_average_skill'] = calculate_initial_average_skill(profile)
        # Add individual category skills
        for cat in AGENT_CATEGORIES:
            col_name = f'initial_skill_{cat.replace(" ", "_")}'
            row[col_name] = AGENT_PROFILES.get(profile, {}).get(cat, 0.0)
        data.append(row)
    return pd.DataFrame(data)

def generate_plots(df: pd.DataFrame, output_dir: str = '.'):
    """Generate visualization plots from the simulation data."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Plot Solve Rate Over Time
        plt.figure(figsize=(12, 8))
        # Calculate rolling solve rate with a window of 10 steps
        for (profile, agent_type), group in df.groupby(['agent_profile', 'agent_type']):
            group = group.sort_values('step')
            rolling_solve = group['solved'].rolling(window=10, min_periods=1).mean()
            plt.plot(group['step'], rolling_solve, label=f"{profile} - {agent_type}", linewidth=2)
        
        plt.title('Solve Rate Over Time (10-Problem Moving Average)', fontsize=14)
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Solve Rate (Moving Average)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'solve_rate_over_time.png'))
        
        # 2. Plot Average Skill Progression
        plt.figure(figsize=(12, 8))
        for (profile, agent_type), group in df.groupby(['agent_profile', 'agent_type']):
            group = group.sort_values('step')
            plt.plot(group['step'], group['average_skill'], label=f"{profile} - {agent_type}", linewidth=2)
        
        plt.title('Average Skill Progression Over Time', fontsize=14)
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Average Skill Level', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_skill_progression.png'))
        
        # 3. Plot Skill Change by Category (only for recommender)
        # Get profiles with recommender data
        recommender_profiles = df[df['agent_type'] == 'recommender']['agent_profile'].unique()
        
        for profile in recommender_profiles:
            profile_data = df[(df['agent_profile'] == profile) & (df['agent_type'] == 'recommender')]
            if len(profile_data) > 0:
                # Get first and last step data
                first_step = profile_data[profile_data['step'] == profile_data['step'].min()]
                last_step = profile_data[profile_data['step'] == profile_data['step'].max()]
                
                # Calculate skill change for each category
                skill_changes = {}
                for cat in AGENT_CATEGORIES:
                    col_name = f'agent_skill_{cat.replace(" ", "_")}'
                    if col_name in first_step.columns and col_name in last_step.columns:
                        initial = first_step[col_name].values[0]
                        final = last_step[col_name].values[0]
                        skill_changes[cat] = final - initial
                
                # Sort categories by skill change
                sorted_cats = sorted(skill_changes.items(), key=lambda x: x[1], reverse=True)
                cats = [item[0] for item in sorted_cats]
                changes = [item[1] for item in sorted_cats]
                
                plt.figure(figsize=(14, 10))
                bars = plt.barh(cats, changes)
                
                # Color bars based on positive/negative change
                for i, bar in enumerate(bars):
                    if changes[i] > 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
                
                plt.title(f'Skill Change by Category for {profile} (Recommender)', fontsize=14)
                plt.xlabel('Skill Change (Final - Initial)', fontsize=12)
                plt.ylabel('Category', fontsize=12)
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'skill_change_{profile}_recommender.png'))
        
        print(f"Plots saved to {output_dir}")
        return True
    except Exception as e:
        print(f"Error generating plots: {e}")
        return False


def analyze_and_print_results(log_dir: str, file_pattern: str, generate_visualizations: bool = True):
    """
    Loads simulation data, analyzes it, and prints key results in plain text.
    Optionally generates visualizations.
    """
    print("--- Starting Simulation Analysis ---")

    # 1. Load Data
    all_files = glob.glob(os.path.join(log_dir, file_pattern))
    if not all_files:
        print(f"\nERROR: No files found matching pattern '{file_pattern}' in directory '{log_dir}'")
        return

    print(f"Found {len(all_files)} log files:")
    for f in all_files:
        print(f" - {os.path.basename(f)}")

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
    
    # List available unique combinations
    profile_type_combos = df[['agent_profile', 'agent_type']].drop_duplicates()
    print("Available profile/type combinations:")
    for _, row in profile_type_combos.iterrows():
        print(f" - {row['agent_profile']} / {row['agent_type']}")

    # 2. Data Preparation
    # Convert solved to boolean if it's not already
    if df['solved'].dtype != bool:
        df['solved'] = df['solved'].astype(bool)

    # Calculate average agent skill per row using available skill columns
    existing_skill_cols = [col for col in SKILL_COLS if col in df.columns]
    if not existing_skill_cols:
        print("\nWARNING: No skill columns found in the DataFrame. Cannot calculate average skill.")
        df['average_skill'] = np.nan
    else:
        if len(existing_skill_cols) < len(SKILL_COLS):
            print(f"\nWARNING: Calculating average skill based on {len(existing_skill_cols)} found columns (expected {len(SKILL_COLS)}).")
        df['average_skill'] = df[existing_skill_cols].mean(axis=1)

    # Determine the number of simulation steps per run
    try:
        steps_by_group = df.groupby(['agent_profile', 'agent_type'])['step'].max()
        num_steps = steps_by_group.mode()[0]
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
    
    # Format as percentages for display
    solve_rate_table = solve_rate_summary.unstack()
    print(solve_rate_table.map('{:.1%}'.format).to_string(na_rep='N/A'))
    
    # Calculate improvement of recommender over random baseline
    if all(agent_type in solve_rate_table.columns for agent_type in AGENT_TYPES):
        print("\n--- Recommender Improvement Over Random Baseline ---")
        improvement = solve_rate_table['recommender'] - solve_rate_table['random_baseline']
        improvement_pct = improvement / solve_rate_table['random_baseline'] * 100
        improvement_df = pd.DataFrame({
            'Absolute Difference': improvement.map('{:+.1%}'.format),
            'Relative Improvement': improvement_pct.map('{:+.1f}%'.format)
        })
        print(improvement_df.to_string(na_rep='N/A'))
    
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
        # Get the last step for each profile/type combination
        final_steps = df.groupby(['agent_profile', 'agent_type'])['step'].max().reset_index()
        final_skill_rows = pd.merge(
            df, final_steps, 
            on=['agent_profile', 'agent_type', 'step'],
            how='inner'
        )
        
        final_skill_summary = final_skill_rows.groupby(['agent_profile', 'agent_type'])['average_skill'].mean()
        final_skill_summary = final_skill_summary.reindex(idx)
        print(final_skill_summary.unstack().map('{:.3f}'.format).to_string(na_rep='N/A'))
        
        # --- Skill Change Analysis ---
        print("\n--- Average Skill Change (Final - Initial) ---")
        # Get initial skills for each profile
        initial_skills_df = calculate_initial_skills_df()
        
        # For each profile and agent type
        skill_change_data = []
        for profile in PROFILES:
            initial_avg_skill = initial_skills_df[initial_skills_df['agent_profile'] == profile]['initial_average_skill'].values
            if len(initial_avg_skill) > 0:
                initial_avg_skill = initial_avg_skill[0]
                
                for agent_type in AGENT_TYPES:
                    final_skill = final_skill_summary.get((profile, agent_type), np.nan)
                    if not pd.isna(final_skill):
                        skill_change = final_skill - initial_avg_skill
                        skill_change_data.append({
                            'agent_profile': profile,
                            'agent_type': agent_type,
                            'initial_skill': initial_avg_skill,
                            'final_skill': final_skill,
                            'skill_change': skill_change
                        })
        
        if skill_change_data:
            skill_change_df = pd.DataFrame(skill_change_data)
            skill_change_pivot = skill_change_df.pivot(
                index='agent_profile', 
                columns='agent_type', 
                values='skill_change'
            )
            print(skill_change_pivot.map('{:+.3f}'.format).to_string(na_rep='N/A'))
            
            # Calculate improvement of recommender over random for skill gain
            if all(agent_type in skill_change_pivot.columns for agent_type in AGENT_TYPES):
                print("\n--- Recommender vs. Random Skill Gain Comparison ---")
                skill_gain_diff = skill_change_pivot['recommender'] - skill_change_pivot['random_baseline']
                skill_gain_ratio = skill_change_pivot['recommender'] / skill_change_pivot['random_baseline']
                skill_gain_comparison = pd.DataFrame({
                    'Difference in Gain': skill_gain_diff.map('{:+.3f}'.format),
                    'Ratio of Gains': skill_gain_ratio.map('{:.2f}x'.format)
                })
                print(skill_gain_comparison.to_string(na_rep='N/A'))
        
        print("-" * 50)
    else:
        print("WARNING: Skill analysis skipped (no valid 'average_skill' data)")

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
        expected_difficulties = sorted(df['problem_difficulty'].unique())
        difficulty_dist = difficulty_dist.reindex(index=idx, columns=expected_difficulties, fill_value=0)
        
        # Format nicely
        print(difficulty_dist.unstack().unstack().to_string(float_format='%.1f%%', na_rep='N/A'))
        print("-" * 50)
    else:
        print("WARNING: Difficulty distribution analysis skipped ('problem_difficulty' column not found)")

    # --- Category Distribution ---
    if 'problem_category' in df.columns:
        print("--- Top 5 Categories Selected by Each Agent Type (%) ---")
        for profile in PROFILES:
            print(f"\nProfile: {profile}")
            for agent_type in AGENT_TYPES:
                subset = df[(df['agent_profile'] == profile) & (df['agent_type'] == agent_type)]
                if len(subset) > 0:
                    cat_counts = subset['problem_category'].value_counts(normalize=True) * 100
                    print(f"\n  {agent_type.capitalize()}:")
                    for cat, pct in cat_counts.nlargest(5).items():
                        print(f"    {cat}: {pct:.1f}%")
                else:
                    print(f"\n  {agent_type.capitalize()}: No data available")
        print("-" * 50)
    else:
        print("WARNING: Category distribution analysis skipped ('problem_category' column not found)")

    # --- Recommender Reward Analysis (if applicable) ---
    existing_reward_cols = [col for col in REWARD_COLS if col in df.columns]
    if existing_reward_cols and 'recommender_reward' in df.columns:
        print("\n\n" + "=" * 50)
        print("         RECOMMENDER REWARD ANALYSIS")
        print("=" * 50 + "\n")
        
        print("--- Final Recommender Rewards by Category ---")
        # Get final rewards for recommender runs
        recommender_data = df[df['agent_type'] == 'recommender']
        if len(recommender_data) > 0:
            final_steps = recommender_data.groupby(['agent_profile'])['step'].max().reset_index()
            final_reward_rows = pd.merge(
                recommender_data, final_steps, 
                on=['agent_profile', 'step'],
                how='inner'
            )
            
            # Display rewards for top 5 and bottom 5 categories (to keep output manageable)
            for profile in PROFILES:
                profile_data = final_reward_rows[final_reward_rows['agent_profile'] == profile]
                if len(profile_data) > 0:
                    # Get the first row (should be only one per profile at final step)
                    row = profile_data.iloc[0]
                    
                    # Extract category rewards
                    category_rewards = {}
                    for cat in AGENT_CATEGORIES:
                        col_name = f'recommender_reward_{cat.replace(" ", "_")}'
                        if col_name in row:
                            category_rewards[cat] = row[col_name]
                    
                    if category_rewards:
                        # Sort and display
                        sorted_rewards = sorted(category_rewards.items(), key=lambda x: x[1], reverse=True)
                        
                        print(f"\nProfile: {profile}")
                        print("  Top 5 Highest Reward Categories:")
                        for cat, reward in sorted_rewards[:5]:
                            print(f"    {cat}: {reward:.3f}")
                        
                        print("  Bottom 5 Lowest Reward Categories:")
                        for cat, reward in sorted_rewards[-5:]:
                            print(f"    {cat}: {reward:.3f}")
            
            # Average reward trend over time
            print("\n--- Average Recommender Reward Trend ---")
            reward_trend = recommender_data.groupby('step')['recommender_reward'].mean().reset_index()
            start_reward = reward_trend.iloc[0]['recommender_reward']
            end_reward = reward_trend.iloc[-1]['recommender_reward']
            print(f"  Starting Average Reward: {start_reward:.3f}")
            print(f"  Ending Average Reward: {end_reward:.3f}")
            print(f"  Change: {end_reward - start_reward:+.3f}")
        else:
            print("No recommender data available for reward analysis")
    
    # 4. Generate Visualizations (if requested)
    if generate_visualizations:
        print("\n\n" + "=" * 50)
        print("         GENERATING VISUALIZATIONS")
        print("=" * 50 + "\n")
        
        vis_success = generate_plots(df, output_dir='simulation_plots')
        if not vis_success:
            print("WARNING: Failed to generate some or all visualizations")

    print("\n\n" + "=" * 50)
    print("         ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    analyze_and_print_results(LOG_DIR, FILE_PATTERN, generate_visualizations=True)
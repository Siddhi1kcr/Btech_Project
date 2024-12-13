import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import openai
import groq
import random
import csv
import ast
import heapq
from sklearn.model_selection import train_test_split

df = pd.read_csv('Participatory_Budgeting_Dataset.csv')

df_renamed = df.rename(columns={
    'Votes': 'Vote Count',
    'Vote Year': 'Year',
    'BP Funding': 'Funding Amount'
})

columns_to_keep = ['Title', 'Funding Amount', 'Vote Count', 'Year', 'Category', 'Winner', 'Cost']
df_filtered = df_renamed.loc[:, columns_to_keep]

# Standardize text data
df_filtered['Title'] = df_filtered['Title'].str.strip().str.title()
df_filtered['Category'] = df_filtered['Category'].str.strip().str.title()

# Convert numeric columns and handle commas
numeric_columns = ['Funding Amount', 'Vote Count', 'Cost']
for col in numeric_columns:
    df_filtered.loc[:, col] = pd.to_numeric(df_filtered[col].replace(',', '', regex=True), errors='coerce')

# Fill missing numeric values with median
df_filtered.fillna(df_filtered.median(numeric_only=True), inplace=True)

# Fill missing categorical values with 'Unknown'
df_filtered[['Title', 'Category']] = df_filtered[['Title', 'Category']].fillna('Unknown')

# Drop duplicates if any
df_final = df_filtered.drop_duplicates()


"""### Synthetic data creation"""

data = pd.read_csv("Participatory_Budgeting_Dataset.csv")

# Define a function to generate synthetic voter profiles
def generate_voter_profile(project_category):
    # Assign synthetic age, gender, and preferences based on project category
    age = random.randint(18, 65)
    gender = random.choice(["Male", "Female", "Other"])
    preferences = {
        "Category": project_category,
        "Cost Sensitivity": random.uniform(0.1, 1.0),
        "Impact Preference": random.uniform(0.1, 1.0),
    }
    return {"Age": age, "Gender": gender, "Preferences": preferences}

# Add synthetic voters for each project
synthetic_voters = []
for _, row in data.iterrows():
    project_category = row["Category"]
    for _ in range(random.randint(5, 20)):  # Simulate 5–20 voters per project
        voter_profile = generate_voter_profile(project_category)
        synthetic_voters.append({
            "Project": row["Title"],
            "Category": project_category,
            "Voter Profile": voter_profile,
        })

# Convert synthetic voters to a DataFrame
voters_df = pd.DataFrame(synthetic_voters)

# Save synthetic voters to a CSV
voters_df.to_csv("Synthetic_Voter_Profiles.csv", index=False)

# Display a sample of synthetic voters
# voters_df.head()

"""### Creating AI persona(GPT-4o and Llama-3.3-70B)"""

os.environ["GROQ_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

# Function to generate AI persona ballots using GPT models
def generate_gpt_persona(voter_profile, model="gpt-4o"):
    """Generates an AI persona based on the voter profile using OpenAI API."""
    prompt = f"""
    You are a highly informed citizen participating in a participatory budgeting project.
    Your preferences are:
    - Age: {voter_profile['Age']}
    - Gender: {voter_profile['Gender']}
    - Primary Interests: {voter_profile['Preferences']}

    The following projects are available for funding:
    {voter_profile['Projects']}

    Based on your preferences, rank the projects from most to least preferred. Provide a reason for each ranking.
    """

    openai.api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable for API key
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating persona with GPT: {e}")
        return None

# Function to generate AI persona ballots using LLaMA 3.3
def generate_llama_persona(voter_profile, model="llama-3.3-70b-versatile"):
    """Generates an AI persona based on the voter profile using LLaMA 3.3."""
    prompt = f"""
    You are a highly informed citizen participating in a participatory budgeting project.
    Your preferences are:
    - Age: {voter_profile['Age']}
    - Gender: {voter_profile['Gender']}
    - Primary Interests: {voter_profile['Preferences']}

    The following projects are available for funding:
    {voter_profile['Projects']}

    Based on your preferences, rank the projects from most to least preferred. Provide a reason for each ranking.
    """

    # Simulated LLaMA response - replace with actual API implementation
    try:
        # Simulated response for now
        response = {"text": f"Simulated LLaMA AI Response for {voter_profile['Age']} year old voter."}
        return response["text"]
    except Exception as e:
        print(f"Error generating persona with LLaMA: {e}")
        return None

# Load the NYC projects dataset
nyc_projects = pd.read_csv('Participatory_Budgeting_Dataset.csv')
nyc_projects['Cost'] = pd.to_numeric(nyc_projects['Cost'], errors='coerce').fillna(0)

# Randomly select 100 different projects for AI and Human ballots
ai_projects = nyc_projects.sample(n=100, random_state=42).reset_index(drop=True)
remaining_projects = nyc_projects.drop(ai_projects.index)
human_projects = remaining_projects.sample(n=100, random_state=43).reset_index(drop=True)

# Load the synthetic voter profiles dataset
voter_profiles = pd.read_csv('Synthetic_Voter_Profiles.csv')

def parse_voter_profile(profile_str):
    try:
        if 'nan' in profile_str:
            profile_str = profile_str.replace('nan', 'None')
        return ast.literal_eval(profile_str)
    except (SyntaxError, ValueError):
        profile_str_clean = profile_str.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
        if 'nan' in profile_str_clean:
            profile_str_clean = profile_str_clean.replace('nan', 'null')
        try:
            return json.loads(profile_str_clean)
        except json.JSONDecodeError as e:
            print("Error parsing profile string:", profile_str)
            raise e

def generate_ballots(voter_profiles_subset, projects_subset, output_path, ballot_type):
    output_rows = []

    for i, row in voter_profiles_subset.iterrows():
        voter_profile = parse_voter_profile(row['Voter Profile'])
        age = voter_profile.get('Age', None)
        gender = voter_profile.get('Gender', None)
        preferences = voter_profile.get('Preferences', {})

        preferred_category = preferences.get('Category', None)
        cost_sensitivity = preferences.get('Cost Sensitivity', 0.5)
        impact_preference = preferences.get('Impact Preference', 0.5)

        voter_id = i + 1

        if 'Title' in projects_subset.columns:
            projects_subset.rename(columns={'Title': 'Project Title'}, inplace=True)

        proj_df = projects_subset.copy()
        proj_df['category_bonus'] = 0.0
        if preferred_category is not None:
            proj_df.loc[proj_df['Category'] == preferred_category, 'category_bonus'] = 1.0

        proj_df['score'] = proj_df['category_bonus'] + (proj_df['Cost'] * impact_preference) - (proj_df['Cost'] * cost_sensitivity)
        proj_df = proj_df.sort_values('score', ascending=False)

        ballot = ",".join(proj_df['Project Title'].astype(str).tolist())
        project_details = dict(zip(proj_df['Project Title'], proj_df['Cost']))
        project_details_json = json.dumps(project_details)
        preferences_json = json.dumps(preferences)

        output_rows.append({
            'VoterID': voter_id,
            'Age': age,
            'Gender': gender,
            'Preferences': preferences_json,
            f'{ballot_type} Ballot': ballot,
            'Project Details': project_details_json
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"{ballot_type} ballots saved to {output_path}")

# Split the voter profiles into two subsets (50% each)
ai_profiles, human_profiles = train_test_split(voter_profiles, test_size=0.5, random_state=42)

# Generate ballots for AI and Human projects
generate_ballots(ai_profiles, ai_projects, 'ai_ballots.csv', 'AI')
generate_ballots(human_profiles, human_projects, 'human_ballots.csv', 'Human')

# df = pd.read_csv('ai_ballots.csv')
# df.head()

# df = pd.read_csv('human_ballots.csv')
# df.head()

def load_projects_from_ballots(ballots_path, chunksize=1000):
    """Load projects with their votes and costs from ballots in chunks."""
    for chunk in pd.read_csv(ballots_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            project_details = json.loads(row["Project Details"])
            ai_or_human_ballot = row["AI Ballot"] if "AI Ballot" in row else row["Human Ballot"]
            ranked_projects = ai_or_human_ballot.split(",")

            for rank, project in enumerate(ranked_projects):
                if project in project_details:
                    yield {
                        "Project": project,
                        "Votes": len(ranked_projects) - rank,
                        "Cost": project_details[project]
                    }

def utilitarian_greedy(projects_gen, total_budget):
    """Utilitarian Greedy allocation using a generator."""
    projects_sorted = heapq.nlargest(10000, projects_gen, key=lambda x: x['Votes'])
    remaining_budget = total_budget
    funded_projects = []

    for proj in projects_sorted:
        cost = proj['Cost']
        if cost <= remaining_budget:
            funded_projects.append({
                'Project': proj['Project'],
                'Votes': proj['Votes'],
                'Allocated': cost
            })
            remaining_budget -= cost

    return funded_projects, remaining_budget

def equal_shares(projects_gen, total_budget):
    """Equal Shares allocation using a generator."""
    projects_data = [
        {
            'Project': p['Project'],
            'Votes': p['Votes'],
            'Cost': p['Cost'],
            'Allocated': 0.0,  # Initialize Allocated field
            'Funded': False   # Initialize Funded field
        }
        for p in projects_gen
    ]
    total_votes = sum(p['Votes'] for p in projects_data)
    if total_votes == 0:
        return [], total_budget

    leftover_budget = total_budget
    unfunded_projects = projects_data[:]

    while leftover_budget > 0 and len(unfunded_projects) > 0:
        current_votes = sum(p['Votes'] for p in unfunded_projects)
        if current_votes == 0:
            break

        for p in unfunded_projects:
            p['Share'] = (p['Votes'] / current_votes) * leftover_budget

        fully_funded_projects = []
        redistributed_amount = 0.0

        for p in unfunded_projects:
            cost_needed = p['Cost'] - p['Allocated']
            if p['Share'] >= cost_needed:
                p['Allocated'] += cost_needed
                p['Funded'] = True
                excess = p['Share'] - cost_needed
                redistributed_amount += excess
                leftover_budget -= cost_needed
                fully_funded_projects.append(p)

        if len(fully_funded_projects) == 0:
            for p in unfunded_projects:
                cost_needed = p['Cost'] - p['Allocated']
                allocation = min(p['Share'], cost_needed)
                p['Allocated'] += allocation
                leftover_budget -= allocation
            break
        else:
            for fp in fully_funded_projects:
                unfunded_projects.remove(fp)
            leftover_budget += redistributed_amount

    funded_projects = [
        {
            'Project': p['Project'],
            'Votes': p['Votes'],
            'Allocated': p['Allocated']
        }
        for p in projects_data if p['Allocated'] > 0
    ]

    return funded_projects, leftover_budget

if __name__ == "__main__":
    total_budget = 50000 #you can change this according to your needs.

    # Apply to AI Ballots
    ai_projects_gen = load_projects_from_ballots("ai_ballots.csv")
    funded_ug_ai, remaining_ug_ai = utilitarian_greedy(ai_projects_gen, total_budget)
    ai_projects_gen = load_projects_from_ballots("ai_ballots.csv")  # Reload generator
    funded_es_ai, remaining_es_ai = equal_shares(ai_projects_gen, total_budget)

    pd.DataFrame(funded_ug_ai).to_csv("outputs/utilitarian_greedy_ai.csv", index=False)
    pd.DataFrame(funded_es_ai).to_csv("outputs/equal_shares_ai.csv", index=False)

    # Apply to Human Ballots
    human_projects_gen = load_projects_from_ballots("human_ballots.csv")
    funded_ug_human, remaining_ug_human = utilitarian_greedy(human_projects_gen, total_budget)
    human_projects_gen = load_projects_from_ballots("human_ballots.csv")  # Reload generator
    funded_es_human, remaining_es_human = equal_shares(human_projects_gen, total_budget)

    pd.DataFrame(funded_ug_human).to_csv("outputs/utilitarian_greedy_human.csv", index=False)
    pd.DataFrame(funded_es_human).to_csv("outputs/equal_shares_human.csv", index=False)

    print("Utilitarian Greedy and Equal Shares allocations completed for AI and Human Ballots.")

def load_ballots(ballots_path):
    """Load ballots and extract rankings."""
    ballots = pd.read_csv(ballots_path)
    results = []
    for _, row in ballots.iterrows():
        projects = row["AI Ballot"] if "AI Ballot" in row else row["Human Ballot"]
        project_details = json.loads(row["Project Details"])
        rankings = projects.split(",")
        results.append({"Rankings": rankings, "Details": project_details})
    return results

def calculate_overlap(ai_rankings, human_rankings):
    """Calculate overlap percentage between AI and human rankings."""
    ai_projects = set(ai_rankings)
    human_projects = set(human_rankings)
    overlap = len(ai_projects.intersection(human_projects))
    total_projects = len(human_projects)
    return (overlap / total_projects) * 100 if total_projects > 0 else 0

def calculate_spearman(ai_rankings, human_rankings):
    """Calculate Spearman rank correlation between AI and human rankings."""
    common_projects = set(ai_rankings).intersection(set(human_rankings))
    if not common_projects:
        return None  # No common projects, cannot compute correlation

    ai_ranks = {proj: rank for rank, proj in enumerate(ai_rankings)}
    human_ranks = {proj: rank for rank, proj in enumerate(human_rankings)}

    ai_scores = [ai_ranks[proj] for proj in common_projects]
    human_scores = [human_ranks[proj] for proj in common_projects]
    correlation, _ = spearmanr(ai_scores, human_scores)
    return correlation

def visualize_bias_analysis(ai_ballots_path, human_ballots_path):
    """Perform bias analysis and visualize the results."""
    # Load ballots
    ai_ballots = load_ballots(ai_ballots_path)
    human_ballots = load_ballots(human_ballots_path)

    overlaps = []
    correlations = []

    for ai, human in zip(ai_ballots, human_ballots):
        overlap = calculate_overlap(ai["Rankings"], human["Rankings"])
        spearman = calculate_spearman(ai["Rankings"], human["Rankings"])
        overlaps.append(overlap)
        correlations.append(spearman if spearman is not None else 0)

    # Visualization 1: Overlap Percentage Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(overlaps, bins=10, kde=True, color="skyblue")
    plt.title("Distribution of Overlap Percentage Between AI and Human Ballots")
    plt.xlabel("Overlap Percentage")
    plt.ylabel("Frequency")
    plt.show()

    # Visualization 2: Spearman Correlation Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(correlations, bins=10, kde=True, color="orange")
    plt.title("Distribution of Spearman Rank Correlation Between AI and Human Ballots")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.show()

    # Visualization 3: Scatterplot of Overlap vs Spearman Correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(overlaps, correlations, alpha=0.6, color="green")
    plt.title("Overlap vs Spearman Correlation Between AI and Human Ballots")
    plt.xlabel("Overlap Percentage")
    plt.ylabel("Spearman Correlation")
    plt.grid()
    plt.show()

    # Summary Statistics
    print("Bias Analysis Summary:")
    print(f"Average Overlap Percentage: {sum(overlaps) / len(overlaps):.2f}%")
    print(f"Average Spearman Correlation: {sum(correlations) / len(correlations):.2f}")
    print(f"Number of Ballots Analyzed: {len(overlaps)}")

if __name__ == "__main__":
    visualize_bias_analysis("ai_ballots.csv", "human_ballots.csv")

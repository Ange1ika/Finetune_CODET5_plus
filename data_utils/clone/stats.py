import json
import pandas as pd
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
import seaborn as sns

# Установка стиля для графиков
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def read_jsonl_file(file_path):
    """
    Reads a .jsonl file and returns a list of dictionaries.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{file_path}': {e}")
        return None
    return data

def calculate_normalized_levenshtein(source, target):
    """
    Calculates the normalized Levenshtein distance between two strings.
    Normalized by the length of the longer string. Returns 0 if one string is missing.
    """
    if not source or not target:
        return 0.0 # Return 0 for missing strings
    
    dist = levenshtein_distance(source, target)
    max_len = max(len(source), len(target))
    
    if max_len == 0: # Avoid division by zero if both strings are empty
        return 0.0
    
    return dist / max_len

# --- Main Script ---
file_name = '/home/anzhelika/Desktop/CodeT5/better_dataset.jsonl' # Specify your .jsonl file name here
dataset_raw = read_jsonl_file(file_name)

if dataset_raw is None or not dataset_raw:
    print("Could not read data or file is empty.")
else:
    print(f"Successfully read {len(dataset_raw)} records from {file_name}")

    # Create DataFrame from raw data
    df = pd.DataFrame(dataset_raw)

    # Calculate normalized Levenshtein distance for each record
    df['normalized_levenshtein_distance'] = df.apply(
        lambda row: calculate_normalized_levenshtein(row.get('source'), row.get('target')),
        axis=1
    )

    # --- General Statistics ---
    print("\n--- General Statistics ---")
    print(f"Total number of records: {len(df)}")
    print("\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print("\nLabel Distribution Percentages:")
    label_percentages = df['label'].value_counts(normalize=True) * 100
    print(label_percentages.round(2))

    print("\nStatistics for Normalized Levenshtein Distance:")
    print(df['normalized_levenshtein_distance'].describe())

    # --- Visualization with Matplotlib and Seaborn ---
    plt.figure(figsize=(16, 6))

    # Plot 1: Label Distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Label Distribution in Dataset')
    plt.xlabel('Label')
    plt.ylabel('Number of Records')
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width() / 2, height + 0.1,
                       f'{int(height)} ({height/len(df):.1%})',
                       ha='center', va='bottom')

    # Plot 2: Histogram of Normalized Levenshtein Distance
    plt.subplot(1, 2, 2)
    sns.histplot(df['normalized_levenshtein_distance'], kde=True, bins=20, color='skyblue')
    plt.title('Distribution of Normalized Levenshtein Distance')
    plt.xlabel('Normalized Levenshtein Distance')
    plt.ylabel('Number of Records')

    plt.tight_layout()
    plt.show()

    # Additional plot: Box plot for Normalized Levenshtein by Label
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='normalized_levenshtein_distance', data=df, palette='muted')
    plt.title('Normalized Levenshtein Distance by Label')
    plt.xlabel('Label')
    plt.ylabel('Normalized Levenshtein Distance')
    plt.tight_layout()
    plt.show()
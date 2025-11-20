import json
from collections import Counter
import statistics
from pathlib import Path

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# === LOAD ===
train = load_jsonl("testgen_train.jsonl")
val   = load_jsonl("testgen_validation.jsonl")
test  = load_jsonl("testgen_test.jsonl")

all_data = train + val + test

print("=== Dataset Sizes ===")
print(f"Train:      {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test:       {len(test)}")
print(f"Total:      {len(all_data)}")


# === REPOS ===
repos = Counter([d["source_repo"] for d in all_data])
print("\n=== Repositories (top) ===")
for k, v in repos.most_common():
    print(f"{k}: {v}")


# === Length statistics ===
def text_len(s):
    return len(s.split())

source_lens = [text_len(d["source"]) for d in all_data]
target_lens = [text_len(d["target"]) for d in all_data]

print("\n=== Length Statistics ===")

def stats(arr, name):
    print(f"{name}:")
    print(f"  Mean:    {statistics.mean(arr):.2f}")
    print(f"  Median:  {statistics.median(arr)}")
    print(f"  Min:     {min(arr)}")
    print(f"  Max:     {max(arr)}")
    print(f"  Std:     {statistics.stdev(arr):.2f}")
    print("")

stats(source_lens, "SOURCE (functions)")
stats(target_lens, "TARGET (tests)")


# === Check duplicates ===
pairs = Counter((d["source"], d["target"]) for d in all_data)
duplicate_count = sum(1 for p, cnt in pairs.items() if cnt > 1)
print(f"\n=== Duplicate pairs: {duplicate_count}")


# === Print sample ===
print("\n=== Example Sample ===")
ex = all_data[0]
print("Source repo:", ex["source_repo"])
print("\nSOURCE:\n", ex["source"])
print("\nTARGET (test):\n", ex["target"])
print("\n=====================\n")

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(source_lens, bins=50)
plt.title("Source function length distribution")
plt.grid(True)
plt.show()


sns.histplot(target_lens, bins=50)
plt.title("Test function length distribution")
plt.grid(True)
plt.show()

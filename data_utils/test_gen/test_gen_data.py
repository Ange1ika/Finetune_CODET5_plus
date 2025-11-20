import os, requests, ast, json, re, random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# === РЕПОЗИТОРИИ ДЛЯ СКАЧИВАНИЯ ===
REPOS = [
    "pydantic/pydantic",
    "pytorch/pytorch",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas"
]

MAX_FILES_PER_REPO = 300
THREADS_IO = 16


# ============================================================
#     UTILS
# ============================================================

def fetch_tree(repo):
    """Получаем дерево файлов репо."""
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        # fallback master
        url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
        r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("tree", [])


def fetch_raw(repo, path):
    """Загрузка raw-контента Python файла."""
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        r = requests.get(url)
        if r.status_code == 200:
            return r.text
    return None


def clean_code(code: str):
    """Чистим docstring + комментарии."""
    code = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', '', code)
    code = re.sub(r'#.*', '', code)
    return "\n".join(l.rstrip() for l in code.splitlines() if l.strip())


def extract_functions(code):
    """Извлекаем обычные функции."""
    try:
        tree = ast.parse(code)
    except:
        return []

    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__"):
                continue
            src = ast.get_source_segment(code, node)
            if src and len(src) > 40:
                out.append((node.name, src))
    return out


def extract_test_functions(code):
    try:
        tree = ast.parse(code)
    except:
        return []

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            src = ast.get_source_segment(code, node)
            if src and len(src) > 20:
                out.append((node.name, src))
    return out

dataset = []

for repo in REPOS:
    print(f"\n📦 Scanning {repo}")
    tree = fetch_tree(repo)

    py_files = [t["path"] for t in tree if t["path"].endswith(".py")]
    py_files = py_files[:MAX_FILES_PER_REPO]

    # Список тестовых файлов
    test_files = [p for p in py_files if "test" in p.lower() or "/tests/" in p]
    
    functions_by_file = {}

    with ThreadPoolExecutor(max_workers=THREADS_IO) as ex:
        futures = {ex.submit(fetch_raw, repo, p): p for p in py_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Loading {repo}"):
            path = futures[fut]
            raw = fut.result()
            if not raw:
                continue
            code = clean_code(raw)

            if path in test_files:
                test_funcs = extract_test_functions(code)
                if test_funcs:
                    functions_by_file[path] = test_funcs
            else:
                funcs = extract_functions(code)
                if funcs:
                    functions_by_file[path] = funcs

    for src_path, funcs in functions_by_file.items():
        if "test" in src_path:
            continue
        
        module_name = Path(src_path).stem
        candidates = [
            p for p in test_files
            if module_name in Path(p).stem
        ]

        if not candidates:
            continue

        all_tests = []
        for tpath in candidates:
            if tpath in functions_by_file:
                all_tests.extend(functions_by_file[tpath])

        if not all_tests:
            continue

        # формируем пары
        for (fname, fcode) in funcs:
            for (tname, tcode) in all_tests:
                dataset.append({
                    "source_repo": repo,
                    "source": fcode,
                    "target": tcode,
                })

random.shuffle(dataset)

for i, item in enumerate(dataset):
    item["example_id"] = f"{i:06d}"

n = len(dataset)
train = dataset[: int(0.8*n)]
val   = dataset[int(0.8*n): int(0.9*n)]
test  = dataset[int(0.9*n):]

def wrap(items, split):
    return [{
        "task": "TEST_GENERATION",
        "source_repo": it["source_repo"],
        "data_split": split,
        "example_id": it["example_id"],
        "source": it["source"],
        "target": it["target"],
    } for it in items]


train_out = wrap(train, "train")
val_out   = wrap(val, "validation")
test_out  = wrap(test, "test")

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

save_jsonl("testgen_train.jsonl", train_out)
save_jsonl("testgen_validation.jsonl", val_out)
save_jsonl("testgen_test.jsonl", test_out)

print("\n🎉 DONE!")
print(f"Train:      {len(train_out)}")
print(f"Validation: {len(val_out)}")
print(f"Test:       {len(test_out)}")

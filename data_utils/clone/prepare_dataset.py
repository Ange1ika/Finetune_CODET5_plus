import os, requests, ast, json, random, re, itertools
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from rapidfuzz.distance import Levenshtein  # pip install rapidfuzz

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "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")  # export GITHUB_TOKEN=...
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
REPOS = ["pydantic/pydantic", "scikit-learn/scikit-learn", "opencv/opencv", "pytorch/pytorch"]
OUTPUT = Path("better_dataset.jsonl")

# ==== ПАРАМЕТРЫ ====
MAX_FILES_PER_REPO = 200
MAX_FUNCS_PER_NAME = 12
MAX_PAIRS_PER_NAME = 40
MAX_TOKENS = 512
JACCARD_MIN = 0.30           # дешёвый предфильтр
LEV_NORM_THR = 0.70          # нормализованная дистанция (меньше — ближе)
THREADS_IO = 16
PROCESSES_CPU = max(1, os.cpu_count() // 2)

def fetch_py_files(repo, branch=None, max_files=MAX_FILES_PER_REPO):
    if branch is None:
        r = requests.get(f"https://api.github.com/repos/{repo}", headers=HEADERS); r.raise_for_status()
        branch = r.json().get("default_branch", "main")
    r = requests.get(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1", headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    return [f["path"] for f in data.get("tree", []) if f["path"].endswith(".py")][:max_files]

def clean_code_comments(code: str) -> str:
    code = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', '', code)  # docstrings
    code = re.sub(r'#.*', '', code)                       # inline comments
    # укорачиваем подряд идущие пустые строки
    return "\n".join([l.rstrip() for l in code.splitlines() if l.strip()])

def get_file_content_raw(repo, path):
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        r = requests.get(url)
        if r.status_code == 200:
            return r.text
    return None

def get_file_content(repo, path):
    raw = get_file_content_raw(repo, path)
    return clean_code_comments(raw) if raw else None

def extract_functions_from_code(cleaned_code):
    # cleaned_code уже без комментариев → не чистим ещё раз
    try:
        tree = ast.parse(cleaned_code)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            src = ast.get_source_segment(cleaned_code, node)
            if not src:
                continue
            if len(src) <= 30 or len(src.split()) <= 5:
                continue
            out.append({"name": name, "code": src})
    return out

def quick_tokens(s: str):
    # очень дёшево: разобьём по неалфанумам
    return set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|->|::|:=|\.|,|\+|-|\*|/|%", s))

def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    union = len(a) + len(b) - inter
    return inter / union

# ==== Сбор функций (параллельная загрузка файлов) ====
all_funcs = []
for repo in REPOS:
    print(f"📦 Scanning {repo}")
    paths = fetch_py_files(repo)
    with ThreadPoolExecutor(max_workers=THREADS_IO) as ex:
        futs = {ex.submit(get_file_content, repo, p): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=repo):
            code = fut.result()
            if not code: continue
            funcs = extract_functions_from_code(code)
            for f in funcs:
                f["repo"] = repo
                f["path"] = futs[fut]
                # заранее считаем “быстрые” признаки
                f["len_tokens"] = len(f["code"].split())
                f["lex"] = quick_tokens(f["code"])
                all_funcs.append(f)

# ==== Группировка по имени ====
same_name = defaultdict(list)
for f in all_funcs:
    same_name[(f["repo"], f["name"])].append(f)

dataset = []

def _pair_wrapper(pair):
    f1, f2 = pair
    return pos_pair_ok(f1, f2)

# ==== Позитивные пары (сэмплинг + быстрый предфильтр + параллельный Levenshtein) ====
def pos_pair_ok(f1, f2):
    c1, c2 = f1["code"], f2["code"]
    # длина
    if f1["len_tokens"] > MAX_TOKENS or f2["len_tokens"] > MAX_TOKENS:
        return None
    # быстрый Jaccard
    if jaccard(f1["lex"], f2["lex"]) < JACCARD_MIN:
        return None
    # быстрый нормализованный Levenshtein
    norm = Levenshtein.normalized_distance(c1, c2)
    if norm < LEV_NORM_THR:
        return {"source": c1, "target": c2, "label": 1}
    return None

for (repo, name), funcs in tqdm(same_name.items(), desc="Pairing"):
    if len(funcs) < 2: 
        continue
    # ограничиваем репрезентативную выборку
    funcs = funcs[:MAX_FUNCS_PER_NAME]
    pairs = list(itertools.combinations(funcs, 2))
    if len(pairs) > MAX_PAIRS_PER_NAME:
        pairs = random.sample(pairs, MAX_PAIRS_PER_NAME)

    # параллельно считаем расстояния
    with ProcessPoolExecutor(max_workers=PROCESSES_CPU) as pex:
        results = list(pex.map(_pair_wrapper, pairs))
    dataset.extend([r for r in results if r])

# ==== Негативные пары (ровно столько же, сколько позитивов) ====
pos_count = sum(1 for d in dataset if d["label"] == 1)
neg_needed, neg_added = pos_count, 0
while neg_added < neg_needed:
    f1, f2 = random.sample(all_funcs, 2)
    if f1["name"] == f2["name"]:
        continue
    if f1["len_tokens"] > MAX_TOKENS or f2["len_tokens"] > MAX_TOKENS:
        continue
    # подстраховка: слишком похожие — пропускаем, чтобы не замусорить негативы
    if jaccard(f1["lex"], f2["lex"]) > 0.5:
        continue
    dataset.append({"source": f1["code"], "target": f2["code"], "label": 0})
    neg_added += 1

random.shuffle(dataset)

# ==== Сохранение ====
with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(dataset)} pairs (pos={pos_count}, neg={neg_needed}) to {OUTPUT}")

Here’s a compact, practical blueprint you can drop into a repo. It shows how to wire a **code-aware RAG** loop in **LangGraph** to help with coding tasks (read/understand files, plan a change, propose a patch, run tests, iterate).

---

# What “good” RAG for coding looks like

**Indexing (offline)**

* Parse repo with Tree-sitter (or `ast` for Python) → extract **symbols** (functions, classes), **call/import graph**, and **docstrings**.
* Chunk by **symbol boundaries** (one function/class per chunk) + attach **metadata**: `path`, `symbol`, `start_line`, `end_line`, `imports`, `callers`.
* Build **hybrid retrieval**:

  * Dense: code/text embedding model (e.g., all-MiniLM/`e5`/OpenAI) on code and docs.
  * Sparse: BM25 / ripgrep hits (term, symbol, file names).
  * Structural: re-rank by “same package/module”, import distance, and call-graph proximity.
* Store in FAISS/LanceDB/PGVector with an inverted index (or just ripgrep at runtime).

**At query time**

* Expand user query with synonyms (API names, file globs), normalize to symbols.
* Retrieve K=30 (dense+sparse), **re-rank** by structure, keep top 8–12, and **attach line spans**.
* Show provenance (paths+line ranges) in every answer and propose **unified diff** patches.

**In the graph**

* Router → Retriever → Planner → Code-Actions → Tester → Critic → Done/Loop.
* Tools: `ripgrep`, `read_file_span`, `apply_patch`, `run_pytest` (or `npm test`), `mypy/ruff` etc.

---

# Minimal LangGraph example (Python)

> Assumes: `langgraph`, `langchain-core`, a vector store (`faiss` example), and some simple local tools. Replace `llm_call()` with your provider.

```python
# pyproject: langgraph, langchain-core, faiss-cpu, pydantic, rapidfuzz, rich
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# State
# -----------------------------
class FileSpan(BaseModel):
    path: str
    start: int
    end: int
    symbol: Optional[str] = None
    score: float = 0.0
    preview: Optional[str] = None

class Plan(BaseModel):
    goal: str
    steps: List[str] = []
    target_files: List[str] = []

class Patch(BaseModel):
    diff_unified: str
    affected_files: List[str]

class RunResult(BaseModel):
    command: str
    stdout: str
    stderr: str
    exit_code: int

class GraphState(BaseModel):
    query: str
    mode: str = "auto"  # auto|read|refactor|bugfix|test
    retrieved: List[FileSpan] = Field(default_factory=list)
    plan: Optional[Plan] = None
    answer: Optional[str] = None
    patch: Optional[Patch] = None
    test_result: Optional[RunResult] = None
    iterations: int = 0
    done: bool = False

# -----------------------------
# Tools (sketches – implement for your stack)
# -----------------------------
import subprocess, textwrap, difflib, pathlib, re

def ripgrep(query: str, root: str = ".") -> List[FileSpan]:
    # Very simple sparse retrieval fallback
    try:
        out = subprocess.run(
            ["rg", "--vimgrep", "-n", "-H", "-S", query, root],
            capture_output=True, text=True, check=False
        ).stdout.splitlines()
    except FileNotFoundError:
        return []

    hits = []
    for line in out[:200]:
        # file:line:col:content
        m = re.match(r"(.+?):(\d+):(\d+):(.*)", line)
        if not m: continue
        path, ln, _, content = m.groups()
        start = max(int(ln) - 15, 1)
        end = int(ln) + 15
        preview = content.strip()[:200]
        hits.append(FileSpan(path=path, start=start, end=end, preview=preview, score=0.3))
    return hits

# Simple vector search wrapper (replace with FAISS/LanceDB)
class DenseIndex:
    def __init__(self): ...
    def search(self, query: str, k: int = 20) -> List[FileSpan]:
        # Implement with your embedding store; return FileSpan with path/start/end/symbol/score
        return []

VECTOR = DenseIndex()

def read_span(span: FileSpan) -> FileSpan:
    try:
        lines = pathlib.Path(span.path).read_text(encoding="utf-8", errors="ignore").splitlines()
        excerpt = "\n".join(lines[span.start-1:span.end])
        span.preview = excerpt[:7000]
        return span
    except Exception:
        return span

def apply_unified_diff(diff_text: str) -> List[str]:
    # For safety, write your own patch applier or call `git apply --index -p0`
    patch_file = ".rag_patch.diff"
    pathlib.Path(patch_file).write_text(diff_text, encoding="utf-8")
    p = subprocess.run(["git", "apply", "--index", patch_file], capture_output=True, text=True)
    if p.returncode != 0:
        return [f"git apply failed: {p.stderr[:1000]}"]
    return ["patch applied"]

def run_cmd(cmd: List[str]) -> RunResult:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return RunResult(command=" ".join(cmd), stdout=p.stdout[-5000:], stderr=p.stderr[-5000:], exit_code=p.returncode)

# -----------------------------
# LLM helpers (replace as needed)
# -----------------------------
def llm_call(system: str, user: str, context: str = "") -> str:
    # Plug your LLM here. Ensure it cites file:line and outputs diffs when asked.
    return "..."

# -----------------------------
# Nodes
# -----------------------------
def route_node(state: GraphState) -> GraphState:
    q = state.query.lower()
    if any(k in q for k in ["bug", "exception", "traceback", "fix"]):
        state.mode = "bugfix"
    elif any(k in q for k in ["refactor", "rename", "extract"]):
        state.mode = "refactor"
    elif any(k in q for k in ["test", "unit test", "pytest"]):
        state.mode = "test"
    else:
        state.mode = "read"
    return state

def retrieve_node(state: GraphState) -> GraphState:
    dense = VECTOR.search(state.query, k=24)
    sparse = ripgrep(state.query)
    # simple merge + de-dupe by (path,start,end)
    key = lambda s: (s.path, s.start, s.end)
    merged = {(key(s)): s for s in (dense + sparse)}.values()
    # Prefer same-module proximity etc. (placeholder)
    ranked = sorted(merged, key=lambda s: s.score, reverse=True)[:12]
    state.retrieved = [read_span(s) for s in ranked]
    return state

def plan_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        f"FILE {s.path}:{s.start}-{s.end}\n{s.preview or ''}" for s in state.retrieved
    )[:15000]
    prompt = f"""You are a senior engineer. User task: {state.query}
Return a compact plan with steps and target files (JSON with goal, steps[], target_files[])."""
    out = llm_call("planner", prompt, context)
    # Parse JSON robustly (omitted)
    state.plan = Plan(goal=state.query, steps=["read context", "edit X", "add tests Y"], target_files=[s.path for s in state.retrieved[:3]])
    return state

def propose_patch_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        f"FILE {s.path}:{s.start}-{s.end}\n{s.preview or ''}" for s in state.retrieved
    )[:15000]
    prompt = f"""Propose a minimal unified diff patch to achieve: {state.plan.goal}
Rules:
- Output ONLY a valid unified diff (git format), with correct file paths.
- Keep changes surgical; add tests if needed under tests/.
- Include helpful comments in the diff where relevant."""
    diff = llm_call("patcher", prompt, context)
    state.patch = Patch(diff_unified=diff, affected_files=state.plan.target_files)
    return state

def apply_and_test_node(state: GraphState) -> GraphState:
    msgs = apply_unified_diff(state.patch.diff_unified)
    test_cmd = ["pytest", "-q"] if any(p.endswith(".py") for p in state.patch.affected_files) else ["bash", "-lc", "npm test -s"]
    result = run_cmd(test_cmd)
    state.test_result = result
    state.answer = "\n".join(msgs + [f"Ran: {result.command}", f"exit={result.exit_code}"])
    return state

def critic_node(state: GraphState) -> GraphState:
    if not state.test_result:
        state.done = True
        return state
    # If tests fail, loop once with failure logs as context
    if state.test_result.exit_code != 0 and state.iterations < 2:
        failure = f"STDOUT:\n{state.test_result.stdout}\nSTDERR:\n{state.test_result.stderr}"
        # Ask LLM to refine the patch given failures (omitted)
        state.iterations += 1
        state.done = False
    else:
        state.done = True
    return state

# -----------------------------
# Graph wiring
# -----------------------------
builder = StateGraph(GraphState)
builder.add_node("route", route_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("plan", plan_node)
builder.add_node("propose_patch", propose_patch_node)
builder.add_node("apply_and_test", apply_and_test_node)
builder.add_node("critic", critic_node)

builder.set_entry_point("route")
builder.add_edge("route", "retrieve")
builder.add_edge("retrieve", "plan")
builder.add_edge("plan", "propose_patch")
builder.add_edge("propose_patch", "apply_and_test")
builder.add_edge("apply_and_test", "critic")
builder.add_conditional_edges(
    "critic",
    lambda s: "loop" if not s.done else END,
    {"loop": "retrieve"}  # simple loop; you could go to propose_patch directly
)

app = builder.compile(checkpointer=MemorySaver())

# Example run:
# state = GraphState(query="Add a CLI flag --dry-run to skip writes and add tests.")
# for ev in app.stream(state):
#     ...
```

---

## Retrieval details that matter for code

* **Chunking**: one **symbol per chunk** (function/class), plus a small **ancestor window** (enclosing class/module docstring). Avoid fixed 1k-token splits for code.
* **Signals for re-ranking**:

  * same module/package as matched term (+2)
  * import distance / call graph proximity (+1…+3)
  * file name/path match (“auth”, “router”) (+1)
  * recent git changes touching same symbols (+1)
* **Context building**: cap to \~8–12 chunks; include exact **line spans**; never paste entire files.
* **Provenance**: always show `path:start-end` next to explanations and in commit message.

---

## Tools to wire in (suggested)

* **Analysis**: `ripgrep`, `ctags`/Tree-sitter symbol index, `git blame`, `git log -S`
* **Quality**: `ruff`/`mypy` (Py), `eslint`/`tsc` (TS), formatters
* **Execution**: `pytest -q` or `npm test -s`
* **Safety**: apply patches via `git apply` on a branch; never run commands with network by default

---

## Prompts (concise)

**Planner (JSON)**

* “Given the user task and code excerpts, return JSON `{goal, steps[], target_files[]}`. Steps are atomic and ordered. Target only files likely to change.”

**Patcher (Unified diff only)**

* “Produce a minimal **unified diff** implementing the plan. Keep changes surgical. Include tests. Do not output anything except the diff.”

**Explainer (optional final)**

* “Summarize the change, list touched symbols, and cite `path:line-line` for each.”

---

## Evaluating your coding RAG

* **Retrieval hit\@K** by oracle questions (“Where is `FooError` raised?”).
* **Task success** rate on a curated set (bugfixes/refactors).
* **Patch acceptance** (applies cleanly, tests pass).
* **Context efficiency** (tokens per success).
* Add **ablation**: dense vs hybrid vs +structure.

---

## Hard-won pitfalls

* Don’t index huge, generated files; exclude `dist/`, `node_modules/`, `migrations` (unless requested).
* Prevent “context drift”: keep K small, prefer structural proximity over raw cosine score.
* Make the LLM **always** output diffs; never free-form code blocks for edits.
* Sandbox test runs; timeouts and output truncation.

---

If you want, I can adapt this to your stack (Python/TS mono-repo, FAISS vs PGVector, Jenkins, KServe) and drop in a ready-to-run `ingest.py` (Tree-sitter + embeddings) plus a `tools.py` set for ripgrep/pytest/ruff.


Below is a concise, production-oriented starter kit to build a **code-aware RAG** that generates *Blender Python* snippets, validates them with **ruff + mypy + Pyright**, optionally smoke-tests them **inside Blender (headless)**, and asks a **vLLM** “critic” to score correctness before returning results.

---

# 1) Ingestion (Blender docs → symbols + chunks)

**Goal:** index API pages & examples as symbol-centric chunks with `path`, `symbol`, `blender_version`, `anchor`, and short examples.

**Sources to target**

* Blender Python API (current): object/type pages + examples. ([docs.blender.org][1])
* Use the Sphinx inventory to map symbols → URLs (`objects.inv`) via `sphobjinv`. ([sphobjinv.readthedocs.io][2])

**Chunking**

* One **symbol per chunk** (`bpy.ops.mesh.primitive_uv_sphere_add`, `bpy.data.objects`, `bpy.types.Object`) + surrounding example code.
* Keep each chunk ≤ \~1.5k chars, attach `see_also` (neighbour symbols & ops).

**Indexer sketch (Python)**

```python
# ingest_blender_docs.py
# deps: httpx, selectolax(or bs4), sphobjinv, faiss-cpu (or lancedb/pgvector), numpy
from sphobjinv import Inventory
import httpx, re, json, numpy as np

def load_inventory(inv_url: str):
    inv = Inventory(url=inv_url)  # e.g., https://docs.blender.org/api/current/objects.inv
    # inv.objects: list of entries with .name and .uri
    return [(o.name, o.uri) for o in inv.objects]

def fetch_and_chunk(base_url, name_uri_pairs):
    for name, uri in name_uri_pairs:
        url = base_url.rstrip("/") + "/" + uri
        html = httpx.get(url, timeout=30).text
        # parse title, code blocks, first paragraph, anchors (selectolax/bs4)
        # yield dict(symbol=name, url=url, code=best_example, text=summary, meta={...})
        yield {...}

def embed_and_store(chunks):
    # compute embeddings for `text + code`; store in FAISS/Lance/PGVector with metadata
    ...

if __name__ == "__main__":
    pairs = load_inventory("https://docs.blender.org/api/current/objects.inv")
    chunks = list(fetch_and_chunk("https://docs.blender.org/api/current", pairs))
    embed_and_store(chunks)
```

**Notes**

* Keep **Blender version** in metadata; allow multi-version indices.
* Deduplicate by `symbol` and URL; prefer pages with example code.

---

# 2) Retrieval (hybrid)

For a user task (“add a Subdivision modifier to the active object”), combine:

* **Dense**: top-K by embedding similarity.
* **Sparse**: `ops`, symbol names, file globs; optional `ripgrep` over local snippets.
* **Structural re-rank**: prefer same `bpy.types.*` family or same operator namespace (`bpy.ops.mesh.*`).

Return 8–12 chunks with line-level anchors.

---

# 3) Validation toolchain (static + runtime)

## Static

* **ruff** (style + pyflakes).
* **mypy** (type checks). Use **`fake-bpy-module`** that matches your Blender version to provide stubs outside Blender. ([PyPI][3], [GitHub][4])
* **Pyright** (fast, precise; good complement to mypy). Run via `npx pyright` or VSCode Pylance. ([GitHub][5], [Microsoft GitHub][6])

> If you meant “pywright”, the correct tool is **Pyright** (by Microsoft).

## Runtime (optional but recommended)

* Launch **headless Blender** to smoke-test generated snippets:

  ```
  blender -b --python /tmp/snippet.py -- --ci
  ```

  Capture exceptions and stdout to grade runtime health. ([Stack Overflow][7])

---

# 4) vLLM “critic” (LLM validation)

Run a lightweight model on your **vLLM** server that:

* Scores the snippet on a rubric: *uses correct API, avoids context-only calls, idempotent, safe defaults*.
* Reads: the **retrieved docs**, the **snippet**, and **linter + runtime outputs**.
* Emits JSON `{score: 0..1, reasons: [...], suggested_fix: <diff or patch>}`.
* Gate: accept if `score >= 0.75` and all linters pass; otherwise iterate once.

---

# 5) LangGraph wiring (minimal skeleton)

```python
# pyproject deps: langgraph, langchain-core, faiss-cpu, httpx, pydantic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DocChunk(BaseModel):
    symbol: str
    url: str
    text: str
    code: Optional[str] = None
    score: float = 0.0

class LintReport(BaseModel):
    ruff: str; mypy: str; pyright: str
    ok: bool

class RuntimeReport(BaseModel):
    ran: bool; exit_code: int; stdout: str; stderr: str

class CriticReport(BaseModel):
    score: float; reasons: List[str]; suggested_diff: Optional[str] = None

class GS(BaseModel):
    request: str
    retrieved: List[DocChunk] = Field(default_factory=list)
    snippet: Optional[str] = None
    lint: Optional[LintReport] = None
    runtime: Optional[RuntimeReport] = None
    critic: Optional[CriticReport] = None
    done: bool = False
    iter: int = 0

# ---- Tool shims (implement for your infra)
def retrieve(query:str)->List[DocChunk]: ...
def gen_snippet(query:str, docs:List[DocChunk])->str: ...
def run_ruff(code:str)->str: ...
def run_mypy(code:str)->str: ...
def run_pyright(code:str)->str: ...
def run_blender_headless(code:str, timeout:int=20)->RuntimeReport: ...
def call_vllm_critic(payload:Dict)->CriticReport: ...

# ---- Nodes
def n_retrieve(s:GS)->GS:
    s.retrieved = retrieve(s.request); return s

def n_generate(s:GS)->GS:
    s.snippet = gen_snippet(s.request, s.retrieved); return s

def n_lint(s:GS)->GS:
    r1 = run_ruff(s.snippet); r2 = run_mypy(s.snippet); r3 = run_pyright(s.snippet)
    ok = all("error" not in r.lower() for r in (r1,r2,r3))
    s.lint = LintReport(ruff=r1, mypy=r2, pyright=r3, ok=ok); return s

def n_runtime(s:GS)->GS:
    s.runtime = run_blender_headless(s.snippet); return s

def n_critic(s:GS)->GS:
    s.critic = call_vllm_critic({
        "request": s.request,
        "docs": [c.dict() for c in s.retrieved],
        "snippet": s.snippet,
        "lint": s.lint.dict(),
        "runtime": (s.runtime.dict() if s.runtime else None),
    })
    # simple accept/iterate policy
    s.done = s.lint.ok and (s.critic.score >= 0.75) and (not s.runtime or s.runtime.exit_code==0)
    if not s.done and s.iter < 1 and s.critic.suggested_diff:
        # apply suggested diff to s.snippet (or regenerate with failure feedback)
        s.iter += 1; s.done = False
    return s

g = StateGraph(GS)
g.add_node("retrieve", n_retrieve)
g.add_node("generate", n_generate)
g.add_node("lint", n_lint)
g.add_node("runtime", n_runtime)   # make optional via conditional edge
g.add_node("critic", n_critic)
g.set_entry_point("retrieve")
g.add_edge("retrieve","generate")
g.add_edge("generate","lint")
g.add_edge("lint","runtime")
g.add_edge("runtime","critic")
g.add_conditional_edges("critic", lambda s: END if s.done else "retrieve", {"retrieve":"retrieve"})
app = g.compile()
```

**Generation prompt (sketch)**

* System: “You write **Blender Python** that runs headless. Prefer `bpy.data` and explicit contexts; avoid GUI-only calls; cite relevant symbols.”
* User: *task text*
* Context: top chunks (symbol docs + example code).

**Critic prompt (vLLM)**

* “Given docs, code, and linter/runtime outputs, score \[0–1] for: API correctness, context safety, idempotence, and side-effect safety. If <0.75, propose a minimal patch.”

---

# 6) CLI + Config

**`pyproject.toml` (ruff)**

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
select = ["E","F","I","B","UP","ANN"]
ignore = ["ANN101","ANN102"]
```

**`mypy.ini`**

```ini
[mypy]
python_version = 3.12
ignore_missing_imports = False
strict = True
# Make bpy resolvable via stubs:
mypy_path = .venv/lib/python3.12/site-packages/fake_bpy_module;typings
```

**`pyrightconfig.json`**

```json
{
  "pythonVersion": "3.12",
  "venvPath": ".",
  "venv": ".venv",
  "typeCheckingMode": "strict",
  "stubPath": ["typings", ".venv/lib/python3.12/site-packages/fake_bpy_module"]
}
```

**Install stubs** (pick the matching Blender version of `fake-bpy-module*`). ([PyPI][3], [Piwheels][8])

**`Makefile`**

```make
lint:
	ruff check .
	mypy .
	npx -y pyright

run-blender:
	blender -b --python scripts/smoke_test.py -- --ci
```

**Dockerfile (CI-safe)**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y nodejs npm xvfb && rm -rf /var/lib/apt/lists/*
RUN pip install ruff mypy faiss-cpu httpx selectolax sphobjinv fake-bpy-module-latest
RUN npm i -g pyright
# (Optional) add Blender or mount it in CI image that already ships Blender
```

---

# 7) Example: user asks

> “Create the active object if missing, add a Subdivision Surface modifier (levels=2), and shade smooth.”

**Generator should pull**:

* `bpy.data.objects`, `bpy.context.view_layer`, `bpy.types.Object.modifiers`, `bpy.ops.object.shade_smooth`, `bpy.types.SubsurfModifier`.

**Typical safe snippet (shape)**

* Use data API (not context-dependent ops) when possible.
* Guard for no active object; create mesh object.
* Idempotent: check if modifier exists before adding; set properties.

(Your generator node emits code; linter/type checks; runtime smoke test in headless Blender; critic returns score & patch if needed.)

---

# 8) Acceptance policy

* Return **snippet + cited symbols (URLs)** + **lint/runtime summaries** + critic score.
* Reject (and iterate once) if:

  * ruff/mypy/pyright report errors,
  * headless run exits non-zero,
  * vLLM score < 0.75.

---

# 9) Why these components

* **Pyright** is a fast, standards-compliant type checker often favored for IDE responsiveness and precise analysis. ([GitHub][5], [Microsoft GitHub][6], [Discussions on Python.org][9])
* **fake-bpy-module** provides Blender API shims so static checkers work outside Blender. ([PyPI][3])
* **Headless Blender** lets you validate runtime behavior without GUI. ([Stack Overflow][7])
* **Docs via Sphinx inventory** give reliable symbol→URL mapping for retrieval. ([sphobjinv.readthedocs.io][2])

---

If you want, I can convert this into a ready-to-run repo (ingester + FAISS store, LangGraph app, vLLM critic route, CI with `ruff+mypy+pyright+blender -b`).

[1]: https://docs.blender.org/api/current/bpy.types.Object.html?utm_source=chatgpt.com "Object(ID) - Blender Python API"
[2]: https://sphobjinv.readthedocs.io/en/stable/syntax.html?utm_source=chatgpt.com "Sphinx objects.inv v2 Syntax — sphobjinv 2.3.1.3 documentation"
[3]: https://pypi.org/project/fake-bpy-module-latest/?utm_source=chatgpt.com "fake-bpy-module-latest 20250630"
[4]: https://github.com/nutti/fake-bpy-module/releases?utm_source=chatgpt.com "Releases · nutti/fake-bpy-module"
[5]: https://github.com/microsoft/pyright?utm_source=chatgpt.com "microsoft/pyright: Static Type Checker for Python"
[6]: https://microsoft.github.io/pyright/?utm_source=chatgpt.com "Pyright - Microsoft Open Source"
[7]: https://stackoverflow.com/questions/15136852/running-blender-python-script-outside-of-blender?utm_source=chatgpt.com "Running Blender python script outside of blender"
[8]: https://www.piwheels.org/project/fake-bpy-module-4-2/?utm_source=chatgpt.com "fake-bpy-module-4.2"
[9]: https://discuss.python.org/t/mypy-vs-pyright-in-practice/75984?utm_source=chatgpt.com "Mypy vs pyright in practice"


Perfect—here’s a tight upgrade that adds a **headless Blender runtime**, feeds **exceptions back to the graph to auto-heal the code**, and **captures a render/screenshot** as an artifact in your RAG.

# What’s new

* A **Blender harness** that executes the generated snippet in `-b` mode, captures `stdout/stderr`, serializes a **JSON report** (success, traceback, timings), and saves a **still render** to disk.
* LangGraph changes: a **Runtime → Healer** loop using the harness report; the render path is stored as an **artifact** and attached to the final answer (and optionally embedded/hashed for retrieval provenance).
* Guard rails: add camera/light if missing; idempotent scene prep; low-res default render to keep CI fast.

---

# 1) Blender harness (headless, JSON report, render)

Create `scripts/harness_blender.py` (this runs *inside* Blender):

```python
# scripts/harness_blender.py
# Usage (external): blender -b --python scripts/harness_blender.py -- --snippet /tmp/snippet.py --report /tmp/blender_report.json --render /tmp/rag_render.png --max-seconds 20 --ci
import bpy, sys, json, io, time, traceback, contextlib, argparse, os, math

def ensure_scene_minimum():
    # Camera
    if not any(o.type == 'CAMERA' for o in bpy.data.objects):
        cam_data = bpy.data.cameras.new(name="RAG_Camera")
        cam = bpy.data.objects.new("RAG_Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        cam.location = (0, -3, 2)
        cam.rotation_euler = (math.radians(65), 0, 0)
        bpy.context.scene.camera = cam
    # Light
    if not any(o.type == 'LIGHT' for o in bpy.data.objects):
        light_data = bpy.data.lights.new(name="RAG_Light", type='SUN')
        light = bpy.data.objects.new("RAG_Light", light_data)
        bpy.context.scene.collection.objects.link(light)
        light.location = (0, -2, 4)

def safe_render(path: str, max_seconds: int = 20):
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t0

def run_snippet(snippet_path: str, max_seconds: int):
    ns = {"__name__": "__main__"}
    code = open(snippet_path, "r", encoding="utf-8").read()
    # Time-limited exec (soft): we measure and abort after render; Blender has no preemptive kill here.
    exec(compile(code, snippet_path, "exec"), ns, ns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippet", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--max-seconds", type=int, default=20)
    ap.add_argument("--ci", action="store_true")
    args, _ = ap.parse_known_args()

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    os.makedirs(os.path.dirname(args.render), exist_ok=True)

    out, err = io.StringIO(), io.StringIO()
    success = False
    exc = None
    timings = {}

    t0 = time.time()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            ensure_scene_minimum()
            t1 = time.time()
            run_snippet(args.snippet, args.max_seconds)
            timings["run_snippet_s"] = time.time() - t1

            # Render even for “data-only” scripts so we always get a visual artifact
            t2 = time.time()
            timings["render_s"] = safe_render(args.render, args.max_seconds)
            timings["total_s"] = time.time() - t0
            success = True
        except Exception as e:
            exc = {
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=50),
            }
            timings["total_s"] = time.time() - t0

    report = {
        "success": success,
        "stdout": out.getvalue()[-10000:],
        "stderr": err.getvalue()[-10000:],
        "exception": exc,
        "render_path": args.render if os.path.exists(args.render) else None,
        "timings": timings,
        "blender_version": bpy.app.version_string,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```

---

# 2) Launcher from your LangGraph runtime node

```python
import subprocess, json, tempfile, pathlib, textwrap

def run_blender_headless(snippet_code: str, blender_exe: str = "blender") -> dict:
    tmp = pathlib.Path(tempfile.gettempdir())
    snippet_path = tmp / "snippet.py"
    report_path = tmp / "blender_report.json"
    render_path = tmp / "rag_render.png"
    snippet_path.write_text(snippet_code, encoding="utf-8")

    cmd = [
        blender_exe, "-b",
        "--python", "scripts/harness_blender.py",
        "--", "--snippet", str(snippet_path),
        "--report", str(report_path),
        "--render", str(render_path),
        "--max-seconds", "20", "--ci",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    # Fallback in case Blender dies before writing report:
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "success": False,
            "stdout": p.stdout[-10000:], "stderr": p.stderr[-10000:],
            "exception": {"type": "BlenderCrashed", "message": "No report produced", "traceback": ""},
            "render_path": str(render_path) if render_path.exists() else None,
            "timings": {}, "launcher_rc": p.returncode,
        }
    return report
```

---

# 3) LangGraph additions: **Healer** node + Artifacts

### State changes

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class Artifact(BaseModel):
    kind: str  # "render"
    path: str
    meta: Dict[str, str] = {}

class GS(BaseModel):
    request: str
    retrieved: List[DocChunk] = Field(default_factory=list)
    snippet: Optional[str] = None
    lint: Optional[LintReport] = None
    runtime: Optional[RuntimeReport] = None
    critic: Optional[CriticReport] = None
    artifacts: List[Artifact] = Field(default_factory=list)
    done: bool = False
    iter: int = 0
```

### Runtime node (updated)

```python
def n_runtime(s: GS) -> GS:
    rep = run_blender_headless(s.snippet)
    s.runtime = RuntimeReport(
        ran=True, exit_code=0 if rep.get("success") else 1,
        stdout=rep.get("stdout",""), stderr=rep.get("stderr","")
    )
    if rep.get("render_path"):
        s.artifacts.append(Artifact(kind="render", path=rep["render_path"], meta={"blender": rep.get("blender_version","")}))
    # Attach structured exception for the Healer
    s.runtime_meta = rep  # keep the raw dict if you track extra fields
    return s
```

### Healer node

```python
def n_heal(s: GS) -> GS:
    if s.runtime and s.runtime.exit_code != 0:
        # Build a compact “failure bundle” for the model
        failure = {
            "stdout": s.runtime.stdout[-4000:],
            "stderr": s.runtime.stderr[-4000:],
        }
        # Prefer the harness JSON – includes exception type & traceback
        rep = getattr(s, "runtime_meta", {})
        exc = rep.get("exception")
        ctx = {
            "exception_type": exc and exc.get("type"),
            "exception_message": exc and exc.get("message"),
            "traceback": exc and exc.get("traceback"),
        }

        # Ask your model to PATCH the snippet (unified diff or full replacement)
        suggested = call_vllm_critic({
            "mode": "healer",
            "request": s.request,
            "docs": [c.dict() for c in s.retrieved],
            "snippet": s.snippet,
            "lint": s.lint.dict() if s.lint else None,
            "runtime_report": rep,
        })
        # Apply the suggested fix (diff or full overwrite)
        if suggested.suggested_diff:
            s.snippet = apply_text_diff(s.snippet, suggested.suggested_diff)  # implement this
        elif hasattr(suggested, "replacement") and suggested.replacement:
            s.snippet = suggested.replacement

        s.critic = suggested
        s.iter += 1
    return s
```

### Graph wiring (delta)

```python
g.add_node("healer", n_heal)

# retrieve -> generate -> lint -> runtime -> critic -> (if fail) healer -> lint -> runtime -> critic
g.add_edge("runtime", "critic")
g.add_conditional_edges(
    "critic",
    lambda s: END if (s.lint and s.lint.ok and (s.runtime.exit_code == 0) and s.critic.score >= 0.75) or s.iter >= 1 else "healer",
    {"healer": "healer"},
)
g.add_edge("healer", "lint")
g.add_edge("lint", "runtime")
```

---

# 4) Critic / Healer prompts (delta)

**Critic (unchanged rubric + now include `render_path`):**

* Inputs: top docs, snippet, linter outputs, harness report (stdout/stderr/exception), optional thumbnail hash.
* Output JSON: `{score, reasons[], suggested_diff?}`.

**Healer (specialized):**

* Instruction: *“Given the Blender traceback and docs, produce the **smallest** change that makes the snippet run headless and render successfully. Prefer `bpy.data` API over context-sensitive ops; ensure idempotence. Output a unified diff **or** a full replacement.”*

---

# 5) Keep the render “in the RAG”

* After runtime, store the **artifact path** and (optionally) compute:

  * **Perceptual hash (pHash/dHash)** for similarity,
  * **CLIP or TinyCLIP embedding** for future *example-based retrieval* (“like that bevel result”),
  * **Provenance**: request text, symbol hits, Blender version, snippet hash.
* Persist alongside your vector store metadata so future generations can fetch “working exemplars” with visuals.

Example artifact index entry:

```json
{
  "kind": "render",
  "path": "/tmp/rag_render.png",
  "snippet_sha256": "…",
  "blender_version": "4.2.0",
  "symbols": ["bpy.types.Object.modifiers", "bpy.types.SubsurfModifier"],
  "request": "Add Subdivision modifier and shade smooth",
  "passed": true
}
```

---

# 6) CI knobs

* Fast lane: set `512x512`, CPU/Cycles or Workbench/Eevee as available; cap total time to \~20–30s.
* Fail early if **no report**; surface `launcher_rc`.
* Always upload `report.json` + `rag_render.png` as CI artifacts.

---

# 7) Minimal example snippet policy

Your generator should already produce code that:

* Avoids GUI-only contexts,
* Checks/create active object,
* Is **idempotent** (don’t duplicate modifiers/materials),
* Ends without blocking operations.

If it still fails (e.g., context errors, missing data), the **Healer** uses the harness traceback to adjust (e.g., replace `bpy.ops.object.modifier_add` with direct `obj.modifiers.new`, set `scene.camera`, etc.).

---

This slots directly into the blueprint you already have:

* Drop in `scripts/harness_blender.py`.
* Replace the runtime node and add the healer node.
* Start persisting `Artifact(kind="render", path=…)` into your store for *visual RAG*.


Got it — you want a **code-aware RAG over a GitHub repo**. Here’s a compact, production-lean recipe plus a minimal working example.

# What to build (brief)

* **Ingest**: pull repo files, extract symbols (functions/classes), keep file path + line spans.
* **Embed**: use a **code embedding model** (open-weights option below).
* **Index**: a dense vector store (FAISS/Qdrant) + optional BM25 for hybrid.
* **Retrieve**: top-k by dense (optionally hybrid), then **rerank** with a cross-encoder.
* **Generate**: answer with grounded snippets (path\:line-ranges).

**Good open components**

* Code embeddings: **BAAI/bge-code-v1** (Apache-2.0; code-oriented, multilingual; instruction-prompted). ([Hugging Face][1])
* Reranker: **BAAI/bge-reranker-v2-m3** (multilingual, lightweight). ([BGE Model][2], [Hugging Face][3])
* Parsers for symbol-level chunks: **Tree-sitter** (multi-language) and **LibCST** for Python (comment/whitespace-preserving). ([tree-sitter.github.io][4], [GitHub][5])

---

# Minimal working example (Python)

**What it does**

* Walks a local clone of a repo
* Splits code into **function/class chunks** (simple heuristics; works fine for Python/TS/JS/Go/Java/C++)
* Embeds with **bge-code-v1**
* Indexes in **FAISS**
* At query time: retrieves top-k and **reranks** with **bge-reranker-v2-m3**
* Returns grounded hits (path + line numbers) you can feed to your LLM

```bash
pip install "faiss-cpu>=1.8" FlagEmbedding "rank-bm25>=0.2.2" tiktoken
```

```python
import os, re, json, hashlib, pathlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from FlagEmbedding import FlagLLMModel, FlagReranker  # bge-code-v1 + bge-reranker-v2-m3

# ---------------------------
# 1) CONFIG
# ---------------------------
REPO_DIR = "/path/to/local/clone"  # git clone beforehand
INCLUDE_EXT = {".py",".js",".ts",".tsx",".java",".go",".rs",".cpp",".cc",".c",".h",".hpp",".md",".yaml",".yml"}
CHUNK_MAX_LINES = 150
CHUNK_MIN_LINES = 5
TOPK = 20
RERANK_K = 8

# bge-code-v1 needs an instruction for queries (see model card)
QUERY_INSTRUCT = "Retrieve code or docs relevant to the question."
QUERY_PROMPT = f"<instruct>{QUERY_INSTRUCT}\n<query>{{}}"

# ---------------------------
# 2) SIMPLE CHUNKING
#    (fast baseline; for production use tree-sitter/libcst)
# ---------------------------
def iter_files(root: str):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in INCLUDE_EXT and p.stat().st_size < 1_000_000:
            yield p

def split_code_into_chunks(text: str, lang_ext: str) -> List[Tuple[int,int,str]]:
    """Return list of (start_line, end_line, chunk_text)."""
    lines = text.splitlines()
    n = len(lines)
    # heuristic boundaries: function/class signatures and big gaps
    boundaries = [0]
    pattern = {
        ".py": r"^\s*(def |class )",
        ".js": r"^\s*(function |class |const .*=\s*\(|export )",
        ".ts": r"^\s*(function |class |const .*=\s*\(|export )",
        ".tsx": r"^\s*(function |class |const .*=\s*\(|export )",
        ".java": r"^\s*(public |private |protected |class |interface )",
        ".go": r"^\s*(func |type |var |const )",
        ".rs": r"^\s*(fn |struct |enum |impl )",
        ".c": r"^\s*[\w\*\s]+\(.*\)\s*{",
        ".h": r"^\s*[\w\*\s]+\(.*\)\s*;|#pragma|#define",
        ".hpp": r"^\s*(class |struct )|#pragma|#define",
        ".cpp": r"^\s*[\w\*\s]+\(.*\)\s*{|\bclass\b|\bstruct\b",
        ".cc": r"^\s*[\w\*\s]+\(.*\)\s*{|\bclass\b|\bstruct\b",
        ".md": r"^#{1,6}\s",
        ".yaml": r"^.{0,}$",
        ".yml": r"^.{0,}$",
    }.get(lang_ext, r"^$")

    sig = re.compile(pattern)
    for i, line in enumerate(lines):
        if i - boundaries[-1] > CHUNK_MAX_LINES or sig.search(line):
            boundaries.append(i)
    boundaries.append(n)

    # window + prune small chunks
    chunks = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b - a >= CHUNK_MIN_LINES:
            chunk = "\n".join(lines[a:b])
            chunks.append((a+1, b, chunk))  # 1-indexed lines
    return chunks

def doc_for(path: Path, start: int, end: int, text: str) -> Dict:
    return {
        "id": hashlib.md5(f"{path}:{start}-{end}".encode()).hexdigest(),
        "repo_path": str(path.relative_to(REPO_DIR)),
        "start_line": start,
        "end_line": end,
        "content": text
    }

# ---------------------------
# 3) EMBEDDING MODELS
# ---------------------------
embedder = FlagLLMModel(
    "BAAI/bge-code-v1",
    query_instruction_format="<instruct>{}\n<query>{}",
    query_instruction_for_retrieval=QUERY_INSTRUCT,
    trust_remote_code=True,
    use_fp16=True,
)
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

# ---------------------------
# 4) BUILD INDEX
# ---------------------------
docs: List[Dict] = []
for p in iter_files(REPO_DIR):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    for s,e,chunk in split_code_into_chunks(txt, p.suffix.lower()):
        docs.append(doc_for(p,s,e,chunk))

corpus = [d["content"] for d in docs]
emb = embedder.encode_corpus(corpus)  # ndarray [N, D]
emb = np.asarray(emb, dtype="float32")
index = faiss.IndexFlatIP(emb.shape[1])
faiss.normalize_L2(emb)
index.add(emb)

# Persist if you like:
# faiss.write_index(index, "code.index")
# Path("docs.jsonl").write_text("\n".join(json.dumps(d) for d in docs))

# ---------------------------
# 5) QUERY -> RETRIEVE -> RERANK
# ---------------------------
def search(query: str, topk=TOPK, rerank_k=RERANK_K):
    qv = embedder.encode_queries([query])[0].astype("float32")
    qv = qv / np.linalg.norm(qv)
    scores, idxs = index.search(qv[None, :], topk)
    candidates = [(float(scores[0][i]), int(idxs[0][i]), docs[int(idxs[0][i])]) for i in range(len(idxs[0]))]

    # rerank with cross-encoder
    pairs = [(query, c[2]["content"]) for c in candidates]
    rerank_scores = reranker.compute_score(pairs)
    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:rerank_k]

    return [
        {
            "score": float(rerank_score),
            "path": hit[2]["repo_path"],
            "lines": [hit[2]["start_line"], hit[2]["end_line"]],
            "preview": hit[2]["content"][:800]
        }
        for (hit, rerank_score) in reranked
    ]

# Example:
# hits = search("How do we initialize the HTTP client with retries?")
# for h in hits: print(h["path"], h["lines"], h["score"])
```

> Notes
> • **bge-code-v1** uses an instruction format for **queries** (prompt shown above), while **documents** are embedded directly; this is from the official model card. ([Hugging Face][1])
> • The reranker choice **bge-reranker-v2-m3** is open-weights, multilingual, and fast for re-scoring a few dozen candidates. ([BGE Model][2])

---

# Production upgrades (when you need more accuracy)

1. **AST/Tree-sitter chunking (symbol-aware)**
   Use **Tree-sitter** to slice per function/class/module across languages; for Python, use **LibCST** to attach preceding docstrings/comments and retain exact line spans. ([tree-sitter.github.io][4], [GitHub][5])

2. **Hybrid retrieval**
   Add BM25 (e.g., Elastic/OpenSearch) and blend dense + sparse scores (Reciprocal Rank Fusion). Helps with exact symbol/file-name matches.

3. **Structural expansion**
   After initial hits, pull **neighbors**: callers/callees (static analysis), same-file siblings, associated tests, and README sections. This raises answer completeness.

4. **Reranking**
   Keep the cross-encoder stage; **bge-reranker-v2-m3** is a good default. Quantize if needed. ([BGE Model][2])

5. **Grounding & citations**
   When you generate an answer, include **repo-relative path + line ranges** for each snippet, and keep total context under your LLM’s token budget.

6. **Incremental indexing**
   Watch `git rev-parse HEAD` and only re-embed changed files; store `(path, start_line, end_line, sha, mtime)` and soft-delete replaced chunks.

---

# Model options (open-weights friendly)

* **Embeddings (code-focused)**: **BAAI/bge-code-v1** (Apache-2.0). ([Hugging Face][1])
* **Reranker**: **BAAI/bge-reranker-v2-m3** (lightweight). ([BGE Model][2])
  (If you ever consider hosted models: Voyage `voyage-code-2` is strong but not open-weights.) ([Voyage AI][6], [Zilliz][7])

---

If you tell me the repo(s) and preferred stack (FAISS vs Qdrant, Elastic, etc.), I can tailor the chunker (Tree-sitter rules per language), the metadata schema, and a CI job to **incrementally** update the index on every push.

[1]: https://huggingface.co/BAAI/bge-code-v1 "BAAI/bge-code-v1 · Hugging Face"
[2]: https://bge-model.com/bge/bge_reranker_v2.html?utm_source=chatgpt.com "BGE-Reranker-v2 — BGE documentation"
[3]: https://huggingface.co/BAAI/bge-reranker-v2-m3?utm_source=chatgpt.com "BAAI/bge-reranker-v2-m3"
[4]: https://tree-sitter.github.io/tree-sitter/using-parsers/?utm_source=chatgpt.com "Using Parsers - Tree-sitter"
[5]: https://github.com/tree-sitter/py-tree-sitter?utm_source=chatgpt.com "Python bindings to the Tree-sitter parsing library"
[6]: https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/?utm_source=chatgpt.com "voyage-code-2: Elevate Your Code Retrieval"
[7]: https://zilliz.com/ai-models/voyage-code-2?utm_source=chatgpt.com "The guide to voyage-code-2 | Voyage AI"

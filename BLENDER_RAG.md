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

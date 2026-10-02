"""Static release-asset validation for the Aurora 0.25° small Earth-system DIMER pipeline.

Checks the STANDALONE tutorial notebook (DIMER Notebook Specification 2.2 §4, isolated hash-locked environment of
§25.13), the tutorial registry, model card, README, STATUS.md and weight documentation for source conformance and
cross-document identity consistency, and runs the generator parity checks (PAR1-PAR4).

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "aurora_earth_system_pipeline"
REPO_NAME = "aurora-earth-system-pipeline"
NOTEBOOK_NAME = "aurora_earth_system_colab.ipynb"
WORKSHOP_NOTEBOOK_NAME = "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"
EXPECTED_PROFILE = "E2E"
EXPECTED_MODEL_ID = "microsoft/aurora"
PIPELINE_CLASS = "AuroraPipeline"
STAGE_RUNNER = ROOT / "tools" / "tutorial_stages.py"
# The upstream package's own default revision for the small checkpoint, cited in docs/WEIGHTS.md beside the pin.
KNOWN_SHAS: frozenset[str] = frozenset({"0be7e57c685dac86b78c4a19a3ab149d13c6a3dd"})
BYOD_GATES = ("USE_BYOD", "RUN_ACTIVITY")
# EXE1/EXE2 (AUR-m4): the BYOD file comes from a path field on any runtime, the upload dialog only when it is empty.
BYOD_FIELD_LINES = {
    "data": ('USE_BYOD = False  # @param {type:"boolean"}', "BYOD_PATH = ''  # @param {type:\"string\"}"),
}
GUIDED_MARKDOWN_MARKERS = (
    "## How to use this notebook",
    "**Who this notebook is for.**",
    "**Where the code runs.**",
    "**Form controls.**",
    "**Changing a setting.**",
    "**Two kinds of cell.**",
    "**Section tags.**",
    "## The task: Input → Model/System → Output",
    "## Roadmap",
    "**Fast path.**",
    "<summary><strong>Glossary</strong>",
    "> **Infrastructure.**",
    "## 13. Optional activity: change one thing",
    "**Predict → Change one thing → Run → Observe → Explain.**",
    "## Troubleshooting",
    "## Conclude with evidence",
    "**Transfer:**",
)
GUIDED_MIN_COUNTS = {
    "**Predict before running:**": 5,
    "**What to notice:**": 6,
    "**Expected result:**": 7,
    "<summary>Check your reasoning": 6,
    "**Question tested:**": 3,
}
SECTION_TAGS = ("[Concept]", "[Evaluation practice]", "[Engineering]")
INFRASTRUCTURE_TITLES = {
    "check": "# @title Infrastructure: check the runtime and disk; create a fresh run directory",
    "carrier": "# @title Infrastructure: write and verify the carried package, stage runner, lock and manifests",
    "install": "# @title Infrastructure: install the locked runtime into an isolated environment and define the stage runner",
    "weights": "# @title Infrastructure: stage and digest-verify the pinned snapshot",
}
# RUN1: the canonical stages, in this order, each called exactly once.
STAGE_CALLS = (
    "run_stage('weights')",
    "run_stage('runtime')",
    "run_stage('data', *data_options)",
    "run_stage('rollout')",
    "run_stage('frozen')",
    "run_stage('adapt', '--epochs', EPOCHS, '--learning-rate', LEARNING_RATE, '--trainable', TRAINABLE)",
    "run_stage('evaluate')",
    "run_stage('forecast')",
    "run_stage('reload')",
    "run_stage('bundle')",
    "run_stage('activity', '--trainable', ACTIVITY_TRAINABLE)",
)
ALLOWED_KERNEL_IMPORTS = frozenset(
    {"hashlib", "io", "json", "os", "pathlib", "platform", "shutil", "subprocess", "tempfile", "time", "urllib.error", "urllib.request", "uuid", "zipfile", "google.colab"}
)
KERNEL_CODE_MARKERS = (
    "CARRIED_FILES = {",
    "CARRIED_HASHES = {",
    "for name, text in CARRIED_FILES.items():",
    "if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:",
    "NOTEBOOK_SOURCE = json.loads(",
    "if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:",
    "'venv', '--managed-python', '--python', '3.12.12'",
    "'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:'",
    "MPLBACKEND='Agg'",
    "PYTHONIOENCODING='utf-8'",
    "for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'PYTHONHOME'):",
    "def run_stage(stage, *options):",
    "detail = error['type'] + ': ' + error['message']",
    "raise RuntimeError(f'Stage {stage!r} failed (exit {process.returncode}): {detail}')",
    "def obtain_upload(path_text, suffix, field):",
    "from google.colab import files",
    "files.upload()",
    "data_options += ['--byod', obtain_upload(BYOD_PATH, '.nc', 'BYOD_PATH')]",
    "data_options = ['--origin', ORIGIN, '--rollout-steps', ROLLOUT_STEPS, '--max-lead-steps', MAX_LEAD_STEPS]",
    "if RUN_ACTIVITY:",
)
# The stage runner must keep the public-API calls, the checks and the refusals the review fixes rely on.
RUNNER_MARKERS = (
    "AuroraPipeline.from_pretrained(weights_dir=run.snapshot, device=device_name(), use_lora=True)",
    "AuroraPipeline.from_artifact(run.out / ADAPTER_DIR, weights_dir=run.snapshot, device=device_name())",
    "fetched = stage_missing_files(run.snapshot, allow_download=True)",
    "verified = verify_snapshot(run.snapshot)",
    "record = convert_model(run.snapshot)",
    "audit_pickle(run.snapshot / SOURCE_CKPT_NAME, allowed=CKPT_ALLOWED_GLOBALS)",
    'raise RuntimeError(f"the isolated environment does not match the carried lock: {mismatched}")',
    "windows = fetch_sample_dataset(cache_dir=run.data_cache)",
    "test_window = load_byod_dataset(byod_path)[0]",
    "dataset_manifest = validate_dataset(",
    "workflow = check_workflow(",
    'write_window_netcdf(trim_window(test_window, workflow["required_steps"])',
    "return history_steps + max(rollout_steps, max_lead_steps)",
    "if pipe.adapter is not None or lora_b_max_abs(pipe) != 0.0:",
    "persistence = persistence_only(test_window, max_lead_steps=lead)",
    "frozen_test = pipe.evaluate(test_window, max_lead_steps=lead)",
    'result = pipe.adapt(train_windows, val_window, epochs=training["epochs"], lr=training["learning_rate"], trainable=training["trainable"], progress=_progress)',
    "artifact = pipe.save_artifact(run.out / ADAPTER_DIR, metadata=",
    "adapted_test = pipe.evaluate(test_window, max_lead_steps=lead)",
    'if outcome["default_configuration"] and not (',
    "dataset.to_netcdf(nc_path)",
    "np.allclose(field, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL)",
    '"served_from_pickle": False',
    '"remote_code_executed": False',
    '"data_base_url": WB2_BASE_URL',
    "the activity changed a canonical output",
    "error_file.write_text(json.dumps(",
    'print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)',
)
# Machine-readable artifacts the stages must write (OUT1-OUT4, INF3).
EXPECTED_OUTPUTS = (
    "_input_manifest.json",
    "_sample_window.nc",
    "_frozen_test.json",
    "_training_history.json",
    "_evaluation_report.json",
    "_forecast.json",
    "_forecast.nc",
    "_reload_parity.json",
    "_result.json",
    "_adapter",
)
MARKDOWN_MARKERS = (
    "**Capability:** 6-hourly global weather forecasting from two gridded analyses, lead-time evaluation against persistence, and bounded LoRA fine-tuning of the foundation model",
    "Both upstream assets are pickles",
    "unpickles each exactly once",
    "not the model's native",
    "out-of-distribution",
    "persistence",
    "diurnal cycle",
    "latitude-weighted RMSE",
    "split by period",
    "Copernicus",
    "**6 at the defaults**",
    "**Pretraining overlap.**",
    "**cannot be ruled out**",
    "Nothing is installed into the notebook kernel",
    "--require-hashes",
    "an estimate",
)
FORBIDDEN_IN_KERNEL = (
    "huggingface_hub",
    "hf_hub_download(",
    "import aurora",
    "from aurora",
    "import torch",
    "import xarray",
    "pickle",
    "sys.executable",
    "importlib",
    "pip install",
    "'-m', 'pip'",
)
# AUR-m3: timings and runtime claims must name their environment or be labelled estimates.
UNSUPPORTED_LEARNER_CLAIMS = re.compile(r"six epochs take about four minutes|about six minutes of model time|Google Colab or Jupyter, Python 3\.12\)", re.I)
# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 2.2; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

NOTEBOOK_SPEC = "2.2"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:** the Hugging Face Hub",
    "## 1. Check the runtime",
    "## 2. Carry the code and install the locked runtime",
    "## 3. Pin, stage, audit and convert the model",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import in the kernel (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(
            r"\bpickle\.load"
            r"|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"
            r"|weights_only\s*=\s*False"
            r"|getattr\(\s*torch\s*,\s*['\"]load['\"]"
        ),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
    ("unhashed or source install", re.compile(r"--no-binary|--no-build-isolation|--trusted-host|--extra-index-url")),
)
# Checked on the carried files (package, runner, lock): everything above except the kernel-only package-import rule.
CARRIED_FORBIDDEN = tuple(item for item in FORBIDDEN_PATTERNS if not item[0].startswith("repository package import"))


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    """Return the source without comment tokens (string contents are preserved)."""
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _package_identity() -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the package source without importing torch."""
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        "pipeline.py must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check('model_card_spec: "1.1"' in front, "MODEL_CARD.md model_card_spec must be 1.1")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def validate_identity_consistency() -> None:
    """The immutable upstream identity must be the same string in every document."""
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {
    2_173_200: "aurora adapter.safetensors from the Kaggle E2E tutorial run (default recipe)",
    12_459_136: "aurora-0.25-static.safetensors converted locally from microsoft/aurora a96afd7",
    81_930_296: "microsoft/aurora a96afd7 aurora-0.25-small-pretrained-test-output.pickle (not staged)",
    163_850_296: "microsoft/aurora a96afd7 aurora-0.25-small-pretrained-test-input.pickle (not staged)",
    195_963_943: "WeatherBench2 ERA5 1.5deg tutorial subset: 57 pinned zarr objects in total",
    451_230_408: "aurora-0.25-small-pretrained.safetensors converted locally from microsoft/aurora a96afd7",
}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {
    "9bd430b666d9267aca34d5c7924d594c3474296c9e39d85701df389816303bc0": "SHA-256 of converted aurora-0.25-static.safetensors (git-ignored)",
    "aeec283f2dbb5861afffe185d1ee13e6df5b6f8616a475b38310c0791085c27f": "audit_pickle digest of sorted globals in aurora-0.25-static.pickle",
    "e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932": "audit_pickle digest of sorted globals in aurora-0.25-small-pretrained.ckpt data.pkl",
    "fc03b5fc5764e08e1f7a05f06ec4debb87fbb66fa2496ec203142221d0843370": "SHA-256 of converted aurora-0.25-small-pretrained.safetensors (git-ignored)",
}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            digests.add(entry["sha256"])
            sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(bool(digests), "no weights/*/dimer-base-manifest.json found to check weight facts against")
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {int(re.sub(r"[,\u202f\u00a0 ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)}
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(not bad_digests, f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}")
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(not bad_sizes, f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}")
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests)
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---

def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    # The registry holds one table row per notebook. The primary notebook's row carries the repository's
    # status; the supplemental workshop row carries its own, which may trail (Candidate while the primary
    # notebook is Release-grade) but never lead it.
    rows = {
        name: next((line for line in registry.splitlines() if line.startswith(f"| `{name}`")), None)
        for name in (NOTEBOOK_NAME, WORKSHOP_NOTEBOOK_NAME)
    }
    primary = rows[NOTEBOOK_NAME]
    _check(primary is not None, f"tutorials/README.md must have a table row for {NOTEBOOK_NAME}")
    _check(f"| {token}" in primary, f"tutorials/README.md must record the {token} status for {NOTEBOOK_NAME}")
    other = [t for t in STATUS_TOKENS if t != token]
    for stale in other:
        _check(f"| {stale}" not in section.replace("**", ""), "README.md carries a conflicting status token")
        _check(f"| {stale}" not in primary, f"tutorials/README.md gives {NOTEBOOK_NAME} a conflicting status token")
    workshop = rows[WORKSHOP_NOTEBOOK_NAME]
    _check(workshop is not None, f"tutorials/README.md must have a table row for {WORKSHOP_NOTEBOOK_NAME}")
    workshop_tokens = [t for t in STATUS_TOKENS if f"| {t}" in workshop]
    _check(len(workshop_tokens) == 1, f"tutorials/README.md must give {WORKSHOP_NOTEBOOK_NAME} exactly one status token")
    _check(
        not (token == "Candidate" and workshop_tokens == ["Release-grade"]),
        f"{WORKSHOP_NOTEBOOK_NAME} cannot be Release-grade while the repository is Candidate",
    )
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(path: Path, notebook: dict) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == EXPECTED_PROFILE, f"{path.name}: profile {profile!r} != declared {EXPECTED_PROFILE!r}")
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(spec == NOTEBOOK_SPEC, f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'")
    _check(dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"), f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode")
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    _check(dimer.get("requires_dimer_worker") is False, f"{path.name}: metadata.dimer.requires_dimer_worker must be false")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(generated.get("repository") == REPO_NAME, f"{path.name}: generated_from.repository must be {REPO_NAME}")
    _check(generated.get("module") == f"src/{PACKAGE}/pipeline.py", f"{path.name}: generated_from.module must name the package entry module")
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    cells = notebook.get("cells", [])
    _check(bool(cells) and cells[0].get("cell_type") == "markdown", f"{path.name}: first cell must be markdown")
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(index > 0 and cells[index - 1].get("cell_type") == "markdown", f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell")
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    kernel = "\n".join(source for index, source, _ in code_cells if not _is_carrier(cells[index]))
    _check(not PLACEHOLDER.search(kernel + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _is_carrier(cell: dict) -> bool:
    return bool(cell.get("metadata", {}).get("dimer", {}).get("embedded_sources"))


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValidationError(f"{name} is not assigned as a literal at the top level of its cell")


def _validate_carrier(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build) -> tuple[int, dict[str, str]]:
    """PAR1/SRC4: one carrier cell; every carried file is the repository file and its CARRIED_HASHES entry is correct;
    the cell verifies each written file against CARRIED_HASHES and stops on a mismatch."""
    carriers = [(index, tree) for index, _source, tree in code_cells if _is_carrier(notebook["cells"][index])]
    _check(len(carriers) == 1, f"{path.name}: exactly one carrier cell (metadata.dimer.embedded_sources) is expected, found {len(carriers)}")
    index, tree = carriers[0]
    files, hashes = _literal(tree, "CARRIED_FILES"), _literal(tree, "CARRIED_HASHES")
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    ctx = build.load_context(ROOT, template, recorded)
    _check(files == ctx["files"], f"{path.name}: carried files differ from the repository sources (PAR1); regenerate the notebook")
    for name, text in files.items():
        _check(hashes.get(name) == hashlib.sha256(text.encode("utf-8")).hexdigest(), f"{path.name}: CARRIED_HASHES[{name!r}] is wrong (SRC4)")
    _check(set(hashes) == set(files), f"{path.name}: CARRIED_HASHES and CARRIED_FILES name different files")
    _check(notebook["cells"][index]["metadata"]["dimer"].get("files") == hashes, f"{path.name}: carrier metadata must record the carried hashes")
    _check(notebook["metadata"]["dimer"]["generated_from"].get("files") == hashes, f"{path.name}: generated_from.files must record the carried hashes")
    _check(notebook["metadata"]["dimer"]["generated_from"].get("module_sha256") == ctx["module_sha256"], f"{path.name}: generated_from.module_sha256 does not match src/ (PAR4)")
    verifies = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and "CARRIED_FILES.items()" in ast.unparse(node.iter):
            body = ast.unparse(node)
            verifies = "CARRIED_HASHES[name]" in body and "raise RuntimeError(" in body and "hashlib.sha256(path.read_bytes())" in body
    _check(verifies, f"{path.name}: the carrier must verify every written file against CARRIED_HASHES and raise on a mismatch (SRC4)")
    runner = template["stage_runner"]
    _check(files.get(runner) == _read(STAGE_RUNNER), f"{path.name}: the carried stage runner is not tools/tutorial_stages.py (PAR1)")
    lock_source = ROOT / template["carried"][template["lock"]]
    _check(files.get(template["lock"]) == _read(lock_source), f"{path.name}: the carried lock is not {lock_source.relative_to(ROOT).as_posix()} (PAR2)")
    return index, files


def _validate_lock(path: Path, files: dict[str, str], build) -> None:
    """ENV1/ENV2: the carried lock pins every pyproject pin, hashes every entry, and was compiled wheel-only."""
    template = _load_tool("notebook_template").TEMPLATE
    lock = files[template["lock"]]
    try:
        build.check_lock(build._pins(ROOT), lock)
    except SystemExit as exc:
        raise ValidationError(f"{path.name}: {exc}") from exc
    header = "\n".join(lock.splitlines()[:3])
    _check("--generate-hashes" in header and "--only-binary :all:" in header, f"{path.name}: the lock must be compiled with --generate-hashes --only-binary :all:")
    _check("--python-platform x86_64-manylinux" in header, f"{path.name}: the lock must target manylinux x86_64 (Colab/Kaggle)")


def _validate_isolated_install(path: Path, code_cells: list[tuple[int, str, ast.Module]], carrier: int) -> None:
    """RUN10/ENV6 (§25.13): nothing is pip-installed into the kernel; the only installer is the pinned uv, which installs
    the lock with --require-hashes into a separate environment; run_stage re-raises a stage's own error message."""
    template = _load_tool("notebook_template").TEMPLATE
    uv = template["uv"]
    install = [(index, source, tree) for index, source, tree in code_cells if source.startswith(INFRASTRUCTURE_TITLES["install"])]
    _check(len(install) == 1, f"{path.name}: exactly one isolated-install cell is expected")
    install_index, source, tree = install[0]
    _check(_literal(tree, "UV_URL") == uv["url"] and uv["url"].startswith("https://files.pythonhosted.org/"), f"{path.name}: UV_URL must be the pinned PyPI wheel")
    _check(_literal(tree, "UV_BYTES") == uv["bytes"], f"{path.name}: UV_BYTES must be the pinned wheel size")
    _check(SHA64.match(str(_literal(tree, "UV_SHA256"))) is not None and _literal(tree, "UV_SHA256") == uv["sha256"], f"{path.name}: UV_SHA256 must pin the uv wheel (64-hex)")
    run_stage = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_stage"]
    _check(len(run_stage) == 1, f"{path.name}: the install cell must define run_stage")
    _check("raise RuntimeError(" in ast.unparse(run_stage[0]) and "error_file" in ast.unparse(run_stage[0]), f"{path.name}: run_stage must re-raise the stage's error message")
    for index, cell_source, cell_tree in code_cells:
        if index == carrier:
            continue
        stripped = _strip_comments(cell_source)
        leaked = [marker for marker in FORBIDDEN_IN_KERNEL if marker in stripped]
        _check(not leaked, f"{path.name}: cell {index} does kernel-side work that belongs in the isolated environment: {leaked}")
        for node in ast.walk(cell_tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                bad = [n for n in names if n not in ALLOWED_KERNEL_IMPORTS]
                _check(not bad, f"{path.name}: cell {index} imports {bad} into the kernel; only the standard library, IPython.display and google.colab are allowed")
            if isinstance(node, ast.List) and any(isinstance(e, ast.Constant) and e.value == "pip" for e in node.elts):
                call = ast.unparse(node)
                _check(
                    call.startswith("[str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes'"),
                    f"{path.name}: cell {index} runs pip other than `uv pip install --python <isolated env> --require-hashes`: {call[:80]}",
                )
            if isinstance(node, ast.Name) and node.id == "urllib" and index != install_index:
                raise ValidationError(f"{path.name}: cell {index} downloads outside the isolated-install cell")


def _validate_identity(path: Path, code_cells: list[tuple[int, str, ast.Module]], carrier: int, revision: str) -> None:
    for index, source, tree in code_cells:
        if index == carrier:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(not rebound, f"{path.name}: {rebound} must not be rebound outside the carried package (cell {index})")
        _check(revision not in source, f"{path.name}: the model revision may appear only in the carried package and manifests (cell {index})")


def _validate_parity(path: Path, notebook: dict, build) -> None:
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")
    _check(current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate")


def _validate_notebook_content(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], markdown: str, carrier: int, files: dict[str, str]) -> None:
    model_id, _revision = _package_identity()
    kernel_cells = [(index, source) for index, source, _ in code_cells if index != carrier]
    kernel = "\n".join(_strip_comments(source) for _, source in kernel_cells)
    carrier_source = next(source for index, source, _ in code_cells if index == carrier)
    missing = [marker for marker in KERNEL_CODE_MARKERS if marker not in kernel + "\n" + carrier_source]
    _check(not missing, f"{path.name}: missing required kernel-code markers: {missing}")
    for title in INFRASTRUCTURE_TITLES.values():
        _check(sum(source.startswith(title) for _, source, _ in code_cells) == 1, f"{path.name}: expected exactly one cell titled {title!r}")
    positions = []
    for call in STAGE_CALLS:
        hits = [index for index, source in kernel_cells if call in source]
        _check(len(hits) == 1, f"{path.name}: expected exactly one `{call}`, found {len(hits)}")
        positions.append(hits[0])
    _check(positions == sorted(positions), f"{path.name}: the stages must run in order {[c.split(chr(39))[1] for c in STAGE_CALLS]} (RUN1)")
    for stage, fields in BYOD_FIELD_LINES.items():
        byod_cells = [source for _, source in kernel_cells if all(line in source.splitlines() for line in fields)]
        _check(len(byod_cells) == 1, f"{path.name}: one learner cell must hold the BYOD form fields exactly: {fields} (EXE1/EXE2, AUR-m4)")
        _check(f"run_stage('{stage}', *{stage}_options)" in byod_cells[0], f"{path.name}: the BYOD fields {fields} must drive the {stage} stage in the same cell")
    for label, pattern in FORBIDDEN_PATTERNS:
        _check(not pattern.search(kernel), f"{path.name}: forbidden/insecure kernel source: {label}")
    for name, text in files.items():
        present = [label for label, pattern in CARRIED_FORBIDDEN if pattern.search(text)]
        _check(not present, f"{path.name}: forbidden/insecure source in carried {name}: {present}")
    runner = files[_load_tool("notebook_template").TEMPLATE["stage_runner"]]
    missing = [marker for marker in RUNNER_MARKERS if marker not in runner]
    _check(not missing, f"{path.name}: the carried stage runner is missing required markers: {missing}")
    missing = [name for name in EXPECTED_OUTPUTS if name not in runner]
    _check(not missing, f"{path.name}: the stage runner must export {missing}")
    _validate_gates(path, code_cells)
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check("restart" not in markdown.lower().replace("no runtime restart", "").replace("no restart", ""), f"{path.name}: learner prose must not instruct a runtime restart (RUN10)")
    claims = UNSUPPORTED_LEARNER_CLAIMS.findall(markdown)
    _check(not claims, f"{path.name}: learner prose makes claims beyond the evidence (AUR-m3): {claims}")
    _check(f"**Profile:** `{EXPECTED_PROFILE}`" in markdown, f"{path.name}: markdown must state the profile")
    _check(f"https://huggingface.co/{model_id}" in markdown, f"{path.name}: references must link {model_id}")


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    for gate in BYOD_GATES:
        assignments = []
        for index, _source, tree in code_cells:
            for node in ast.walk(tree):
                if gate in _assignment_targets(node):
                    assignments.append((index, node))
        _check(len(assignments) == 1, f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}")


def _validate_guided_layer(path: Path, notebook: dict, markdown: str) -> None:
    """GDL1–GDL15 (§3.5): orientation, predictions, checkpoints, an optional activity, collapsed infrastructure."""
    missing = [marker for marker in GUIDED_MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing, f"{path.name}: guided layer (§3.5) is missing: {missing}")
    for marker, minimum in GUIDED_MIN_COUNTS.items():
        found = markdown.count(marker)
        _check(found >= minimum, f"{path.name}: guided layer needs at least {minimum} × {marker!r}, found {found}")
    _check(markdown.count("<details>") == markdown.count("</details>"), f"{path.name}: unbalanced <details> blocks")
    untagged = [line for line in markdown.splitlines() if re.match(r"^## \d+\. ", line) and not line.rstrip().endswith(SECTION_TAGS)]
    _check(not untagged, f"{path.name}: numbered sections must end with a section tag {SECTION_TAGS} (GDL12): {untagged}")
    _check("workshop" not in markdown.lower(), f"{path.name}: learner prose must say 'notebook', not 'workshop' (GDL15)")
    cells = notebook["cells"]
    activity = reload_cell = None
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        meta = cell.get("metadata", {})
        infrastructure = source.startswith("# @title Infrastructure: ")
        collapsed = meta.get("cellView") == "form" and meta.get("jupyter", {}).get("source_hidden") is True
        if infrastructure or _is_carrier(cell):
            _check(collapsed and infrastructure, f"{path.name}: infrastructure cell {index} must be titled '# @title Infrastructure: …' and collapsed (GDL11)")
        else:
            _check(not collapsed, f"{path.name}: learner cell {index} must not be collapsed")
        if "RUN_ACTIVITY = False" in source:
            activity = index
        if "run_stage('reload')" in source:
            reload_cell = index
    _check(activity is not None, f"{path.name}: the optional activity must default to RUN_ACTIVITY = False (GDL10, UX7)")
    _check(reload_cell is not None and activity > reload_cell, f"{path.name}: the optional activity must come after the canonical path's last stage (GDL10)")
    _check("if RUN_ACTIVITY:" in _cell_source(cells[activity]), f"{path.name}: the activity must be gated by `if RUN_ACTIVITY:`")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    names = {p.name for p in tutorials.glob("*.ipynb")}
    expected = {NOTEBOOK_NAME, WORKSHOP_NOTEBOOK_NAME}
    _check(names == expected, f"tutorial notebooks must be exactly {sorted(expected)}, found {sorted(names)}")
    path = tutorials / NOTEBOOK_NAME
    build = _load_tool("build_notebook")
    notebook = json.loads(_read(path))
    code_cells, markdown = _validate_notebook_structure(path, notebook)
    carrier, files = _validate_carrier(path, notebook, code_cells, build)
    _validate_lock(path, files, build)
    _validate_isolated_install(path, code_cells, carrier)
    _model_id, revision = _package_identity()
    _validate_identity(path, code_cells, carrier, revision)
    _validate_parity(path, notebook, build)
    _validate_notebook_content(path, notebook, code_cells, markdown, carrier, files)
    _validate_guided_layer(path, notebook, markdown)
    registry = _read(tutorials / "README.md")
    _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
    _check(f"`{EXPECTED_PROFILE}`" in registry, f"tutorials/README.md must record `{EXPECTED_PROFILE}`")
    _check(f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry, "tutorials/README.md must name the notebook spec version")
    _check("standalone" in registry.lower(), "tutorials/README.md must record that the notebook is standalone")
    _check("isolated" in registry.lower(), "tutorials/README.md must describe the isolated environment")
    _check(f"`{WORKSHOP_NOTEBOOK_NAME}`" in registry, f"{WORKSHOP_NOTEBOOK_NAME} missing from tutorials/README.md")


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "weight-facts", "release-status", "notebook+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

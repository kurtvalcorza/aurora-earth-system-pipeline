# ruff: noqa: E501
"""The uv isolated environment of the weather workshop notebook (2026-10-03).

The workshop used to pip-install its pins into the notebook kernel. Hosted kernels import NumPy before the first cell,
so that install either kept the host's NumPy (Colab 2.1.3, Kaggle 2.0.2 instead of the pinned 2.5.3) or stopped with a
restart request. The notebook now builds a separate environment instead and routes every later code cell to one
persistent worker there, so nothing is installed into the kernel and Run all needs no restart:

1. a pinned ``uv`` wheel (URL + size + SHA-256; the same wheel the primary Colab tutorial carries) builds a managed
   CPython 3.12.12 virtual environment;
2. ``tutorials/requirements-weather-workshop.lock.txt`` (compiled with ``uv pip compile --generate-hashes`` from
   ``tutorials/requirements-weather-workshop.in``, the notebook's previous pins) is installed with
   ``--require-hashes --only-binary :all:``;
3. the router cell starts the worker and registers an IPython input transformer that sends each later cell to it.

``ISOLATED_INSTALL`` and ``ISOLATED_ROUTER`` are the fleet's verified isolated-runtime carrier, copied verbatim from
``bart-mnli-zero-shot-classification-pipeline`` commit ``ee128d2`` (``tools/build_notebook.py``; hosted Colab T4 runs
passed), which in turn carries ``prithvi-eo-feature-extraction-pipeline``'s eo_workshop runtime. Only the values
substituted into ``ISOLATED_INSTALL`` are specific to this notebook. The environment is built for manylinux x86_64, so
the notebook runs on Linux x86_64 only (Google Colab, Kaggle, Linux Jupyter).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK_INPUT = "tutorials/requirements-weather-workshop.in"
LOCK = "tutorials/requirements-weather-workshop.lock.txt"
MANAGED_PYTHON = "3.12.12"
# The uv wheel pinned by tools/notebook_template.py for the primary Colab tutorial (hosted Colab T4 run 2026-10-03).
UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}

ISOLATED_INSTALL = """# @title Infrastructure: install the locked runtime into an isolated environment
# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)
import hashlib
import io
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

{pins_literal}
MANAGED_PYTHON = {python!r}
UV_URL = {uv_url!r}
UV_BYTES = {uv_bytes}
UV_SHA256 = {uv_sha256!r}
LOCK_NAME = {lock_name!r}
LOCK_SHA256 = {lock_sha256!r}
LOCKED_PACKAGES = {n_locked}
# The hash-locked requirements, compiled from the PINS above with `uv pip compile --generate-hashes` for manylinux x86_64.
LOCK_TEXT = r'''{lock_text}'''

SKIP_INSTALL = os.environ.get("DIMER_NOTEBOOK_CI_PREINSTALLED") == "1"
ISOLATED_ENV = Path(os.environ.get("DIMER_ISOLATED_ENV", "dimer_isolated_env")).resolve()
ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"
ISOLATED_TOOLS = ISOLATED_ENV.with_name(ISOLATED_ENV.name + "_tools")

if SKIP_INSTALL:
    print("DIMER_NOTEBOOK_CI_PREINSTALLED=1: the pins are already installed; the notebook runs in this kernel.")
else:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This notebook needs a Linux x86_64 runtime (Google Colab, Kaggle or Linux Jupyter): its locked environment is built for manylinux x86_64.")
    setup_started = time.perf_counter()
    if hashlib.sha256(LOCK_TEXT.encode("utf-8")).hexdigest() != LOCK_SHA256:
        raise RuntimeError("The carried lock does not match its digest: regenerate the notebook from the repository instead of editing this cell.")
    ISOLATED_TOOLS.mkdir(parents=True, exist_ok=True)
    lock_path = ISOLATED_TOOLS / LOCK_NAME
    lock_path.write_text(LOCK_TEXT, encoding="utf-8", newline="\\n")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(UV_URL, timeout=90) as response:
                wheel = response.read(UV_BYTES + 1)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
        raise RuntimeError("The pinned uv wheel failed its size/SHA-256 check: refusing to run it. Run this cell again; if it repeats, the download is being altered.")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".data/scripts/uv"))
        uv = ISOLATED_TOOLS / "uv"
        uv.write_bytes(archive.read(member))
    uv.chmod(0o700)
    # uv gets no kernel Python path; the managed interpreter is downloaded once and reused on a re-run.
    uv_env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        uv_env.pop(name, None)
    if not ISOLATED_PYTHON.is_file():
        subprocess.run([str(uv), "venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON, str(ISOLATED_ENV)], env=uv_env, check=True)
    isolated_version = subprocess.run([str(ISOLATED_PYTHON), "-c", "import platform; print(platform.python_version())"], env=uv_env, check=True, capture_output=True, text=True).stdout.strip()
    if isolated_version != MANAGED_PYTHON:
        raise RuntimeError(f"{{ISOLATED_ENV}} holds Python {{isolated_version}}, not {{MANAGED_PYTHON}}: delete that folder (or start a fresh runtime) and run this cell again.")
    subprocess.run([str(uv), "pip", "install", "--quiet", "--python", str(ISOLATED_PYTHON), "--require-hashes", "--only-binary", ":all:", "--index-url", "https://pypi.org/simple", "-r", str(lock_path)], env=uv_env, check=True)
    print({{"isolated_environment": str(ISOLATED_ENV), "isolated_python": isolated_version, "kernel_python": platform.python_version(), "locked_packages": LOCKED_PACKAGES, "setup_seconds": round(time.perf_counter() - setup_started)}})"""


ISOLATED_ROUTER = (
    "# @title Route the remaining cells to the isolated environment\n"
    "# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)\n"
    "import signal\n"
    "from multiprocessing.connection import Connection\n\n"
    "from IPython import get_ipython as _kernel_shell\n\n"
    "# The worker runs in the isolated environment. It executes each routed cell in one persistent namespace and sends\n"
    "# back printed text, displayed objects and matplotlib figures, so every cell behaves as it would in the kernel.\n"
    + '_WORKER_SOURCE = r"""\nimport ast, base64, builtins, io, linecache, os, signal, sys, traceback, types\nfrom multiprocessing.connection import Connection\n\n_send = Connection(int(sys.argv[1]), readable=False)\n_recv = Connection(int(sys.argv[2]), writable=False)\n\n\nclass _Stream(io.TextIOBase):\n    def __init__(self, name):\n        self._name = name\n\n    @property\n    def encoding(self):\n        return "utf-8"\n\n    def writable(self):\n        return True\n\n    def isatty(self):\n        return False\n\n    def write(self, text):\n        if text:\n            _send.send(("stream", self._name, str(text)))\n        return len(text)\n\n\nsys.stdout, sys.stderr = _Stream("stdout"), _Stream("stderr")\n\n\ndef _figure_bundle(fig):\n    buffer = io.BytesIO()\n    fig.savefig(buffer, format="png", bbox_inches="tight")\n    return {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": repr(fig)}\n\n\ndef _flush_figures():\n    plt = sys.modules.get("matplotlib.pyplot")\n    if plt is None:\n        return\n    for number in plt.get_fignums():\n        _send.send(("display", _figure_bundle(plt.figure(number))))\n    plt.close("all")\n\n\ndef _mimebundle(obj):\n    if hasattr(obj, "savefig"):\n        return _figure_bundle(obj)\n    data = {"text/plain": repr(obj)}\n    for method, mime in (("_repr_html_", "text/html"), ("_repr_markdown_", "text/markdown"), ("_repr_png_", "image/png")):\n        render = getattr(obj, method, None)\n        if callable(render):\n            try:\n                value = render()\n            except Exception:\n                value = None\n            if isinstance(value, bytes):\n                value = base64.b64encode(value).decode("ascii")\n            if value is not None:\n                data[mime] = value\n    return data\n\n\ndef display(*objects, **kwargs):\n    for obj in objects:\n        _send.send(("display", _mimebundle(obj)))\n\n\ntry:\n    import matplotlib\n\n    matplotlib.use("Agg")\n    import matplotlib.pyplot\n\n    matplotlib.pyplot.show = lambda *args, **kwargs: _flush_figures()\nexcept ImportError:\n    pass\n\nif os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":\n    # google.colab only exists in the kernel; forward the BYOD upload dialog to it.\n    def _upload():\n        _send.send(("upload",))\n        reply = _recv.recv()\n        if reply[1] is None:\n            raise RuntimeError("The notebook kernel could not open the upload dialog.")\n        return reply[1]\n\n    try:\n        import google\n    except ImportError:\n        google = types.ModuleType("google")\n        google.__path__ = []\n        sys.modules["google"] = google\n    _colab = types.ModuleType("google.colab")\n    _files = types.ModuleType("google.colab.files")\n    _files.upload = _upload\n    _colab.files = _files\n    google.colab = _colab\n    sys.modules["google.colab"] = _colab\n    sys.modules["google.colab.files"] = _files\n\n_main = types.ModuleType("__main__")\n_main.__dict__.update(__builtins__=builtins, display=display)\nsys.modules["__main__"] = _main\n_count = 0\nwhile True:\n    # An interrupt only lands inside a running cell; between cells it is ignored.\n    signal.signal(signal.SIGINT, signal.SIG_IGN)\n    try:\n        message = _recv.recv()\n    except EOFError:\n        break\n    if message[0] != "run":\n        continue\n    _count += 1\n    filename = f"<isolated cell {_count}>"\n    source = message[1]\n    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)\n    try:\n        signal.signal(signal.SIGINT, signal.default_int_handler)\n        tree = ast.parse(source, filename)\n        tail = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None\n        exec(compile(tree, filename, "exec"), _main.__dict__)\n        if tail is not None:\n            value = eval(compile(tail, filename, "eval"), _main.__dict__)\n            if value is not None:\n                display(value)\n        _flush_figures()\n        signal.signal(signal.SIGINT, signal.SIG_IGN)\n        _send.send(("done",))\n    except BaseException as exc:\n        signal.signal(signal.SIGINT, signal.SIG_IGN)\n        frames = exc.__traceback__.tb_next if exc.__traceback__ is not None else None\n        _send.send(("error", "".join(traceback.format_exception(type(exc), exc, frames)), f"{type(exc).__name__}: {exc}"))\n"""\n\n\nclass IsolatedCellError(RuntimeError):\n    """A routed cell raised inside the isolated environment; its traceback is printed above."""\n\n\nclass IsolatedRuntime:\n    """One persistent worker process in the isolated environment, fed one cell at a time."""\n\n    def __init__(self, python, display=None):\n        to_kernel_r, to_kernel_w = os.pipe()\n        to_worker_r, to_worker_w = os.pipe()\n        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", DIMER_NOTEBOOK_CI_PREINSTALLED="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")\n        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):\n            env.pop(name, None)\n        env["DIMER_KERNEL_IS_COLAB"] = "1" if "google.colab" in sys.modules else "0"\n        self.proc = subprocess.Popen(\n            [str(python), "-c", _WORKER_SOURCE, str(to_kernel_w), str(to_worker_r)],\n            pass_fds=(to_kernel_w, to_worker_r),\n            env=env,\n            start_new_session=True,  # interrupts reach the worker only through run(), exactly once\n        )\n        os.close(to_kernel_w)\n        os.close(to_worker_r)\n        self._recv = Connection(to_kernel_r, writable=False)\n        self._send = Connection(to_worker_w, readable=False)\n        if display is None:\n            from IPython.display import display\n        self._display = display\n\n    def _exited(self):\n        return RuntimeError(\n            f"The isolated environment\'s Python process exited (code {self.proc.wait()}); a crash of this kind is "\n            "usually running out of memory. Restart the session and choose Run all again."\n        )\n\n    def run(self, source):\n        try:\n            self._send.send(("run", source))\n        except OSError:\n            raise self._exited() from None\n        while True:\n            try:\n                message = self._recv.recv()\n            except EOFError:\n                raise self._exited() from None\n            except KeyboardInterrupt:\n                self.proc.send_signal(signal.SIGINT)\n                continue\n            kind = message[0]\n            if kind == "stream":\n                (sys.stdout if message[1] == "stdout" else sys.stderr).write(message[2])\n            elif kind == "display":\n                self._display(message[1], raw=True)\n            elif kind == "upload":\n                self._send.send(("upload", self._colab_upload()))\n            elif kind == "error":\n                sys.stderr.write(message[1])\n                raise IsolatedCellError(message[2]) from None\n            elif kind == "done":\n                return\n\n    @staticmethod\n    def _colab_upload():\n        try:\n            from google.colab import files\n        except ImportError:\n            return None\n        return files.upload()\n\n    def close(self):\n        self._send.close()\n        self.proc.wait(timeout=30)\n\n\ndef _is_user_cell():\n    # ipykernel transforms a cell before executing it; its caller knows whether this is a silent frontend request.\n    frame = sys._getframe(2)\n    while frame is not None:\n        local = frame.f_locals\n        if "silent" in local and "store_history" in local:\n            return bool(local["store_history"]) and not bool(local["silent"])\n        frame = frame.f_back\n    return True\n\n\ndef _route_to_isolated_runtime(lines):\n    source = "".join(lines)\n    if not source.strip() or "# dimer: kernel cell" in source or not _is_user_cell():\n        return lines\n    return [f"_DIMER_ISOLATED_RUNTIME.run({source!r})\\n"]\n\n\n' +
    "if SKIP_INSTALL:\n"
    "    print(\"Routing disabled: the notebook runs in this kernel.\")\n"
    "else:\n"
    "    _ip = _kernel_shell()\n"
    "    _ip.input_transformers_cleanup[:] = [\n"
    "        t for t in _ip.input_transformers_cleanup if getattr(t, \"__name__\", \"\") != \"_route_to_isolated_runtime\"\n"
    "    ]\n"
    "    if isinstance(globals().get(\"_DIMER_ISOLATED_RUNTIME\"), IsolatedRuntime):\n"
    "        _DIMER_ISOLATED_RUNTIME.close()\n"
    "    _DIMER_ISOLATED_RUNTIME = IsolatedRuntime(ISOLATED_PYTHON)\n"
    "    _ip.input_transformers_cleanup.append(_route_to_isolated_runtime)\n"
    "    print(f\"Every later code cell now runs in {ISOLATED_PYTHON} (pid {_DIMER_ISOLATED_RUNTIME.proc.pid}).\")"
)

# The first routed cell: it imports what the learner cells use and stops if the environment is not the locked one.
IMPORTS_TEMPLATE = '''# @title Import the pinned runtime and choose the device
import importlib.metadata
import platform
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import xarray as xr
import gcsfs
import zarr

# The direct pins as resolved in the hash lock the isolated environment was built from.
{locked_versions}
mismatched = {
    name: (importlib.metadata.version(name), version)
    for name, version in LOCKED_VERSIONS.items()
    if importlib.metadata.version(name).split("+")[0] != version
}
if mismatched:
    raise RuntimeError(
        "This Python environment does not hold the locked versions (installed, locked): "
        f"{mismatched}. Run the two cells of section 1 again, or start a fresh runtime and choose Run all."
    )

print({
    "python": platform.python_version(),
    "executable": sys.executable,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "microsoft-aurora": importlib.metadata.version("microsoft-aurora"),
    "xarray": xr.__version__,
    "numpy": np.__version__,
    "versions_match_lock": True,
})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    print("WARNING: T4/CUDA is the canonical runtime for this notebook; CPU will be substantially slower.")'''


def read_pins(repo: Path = REPO) -> list[str]:
    """The notebook's direct pins: the non-comment lines of the lock input."""
    lines = (repo / LOCK_INPUT).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def read_lock(repo: Path = REPO) -> str:
    # Read as text (universal newlines), so a CRLF checkout carries the same LF lock and digest.
    return (repo / LOCK).read_text(encoding="utf-8")


def lock_packages(lock_text: str) -> dict[str, str]:
    """`{name: version}` of every requirement in a uv/pip-compile hash lock (names lower-cased)."""
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", lock_text, re.M)}


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.match(r"[0-9.]+", version).group(0).strip(".").split("."))


def _satisfies(spec: str, version: str) -> bool:
    """`==`, `>=` and `<` clauses, the only operators the lock input uses."""
    release = _release(version)
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        op, bound = re.fullmatch(r"(==|>=|<)\s*([0-9.]+)", clause).groups()
        bound_release = _release(bound)
        if op == "==" and release != bound_release:
            return False
        if op == ">=" and release < bound_release:
            return False
        if op == "<" and release >= bound_release:
            return False
    return True


def resolved_pins(pins: list[str], lock_text: str) -> dict[str, str]:
    """Each direct pin's locked version. Refuses a lock that misses a pin, breaks a range or has an unhashed entry."""
    locked = lock_packages(lock_text)
    out = {}
    for pin in pins:
        name, spec = re.fullmatch(r"([A-Za-z0-9._-]+)\s*(.*)", pin).groups()
        version = locked.get(name.lower())
        if version is None or not _satisfies(spec, version):
            raise SystemExit(f"{LOCK} does not satisfy {pin} (found {version}); recompile the lock")
        out[name] = version
    entries = re.split(r"\n(?=[A-Za-z0-9])", lock_text)
    unhashed = [e.split("==", 1)[0] for e in entries if "==" in e.split("\n", 1)[0] and "--hash=sha256:" not in e]
    if unhashed:
        raise SystemExit(f"lock entries without --hash: {unhashed}")
    if "'''" in lock_text:
        raise SystemExit("lock text cannot be carried in a raw triple-quoted literal")
    return out


def install_cell(repo: Path = REPO) -> str:
    pins = read_pins(repo)
    lock_text = read_lock(repo)
    resolved_pins(pins, lock_text)
    pins_literal = "PINS = [\n" + "".join(f"    {pin!r},\n" for pin in pins) + "]"
    return ISOLATED_INSTALL.format(
        pins_literal=pins_literal,
        python=MANAGED_PYTHON,
        uv_url=UV["url"],
        uv_bytes=UV["bytes"],
        uv_sha256=UV["sha256"],
        lock_name=Path(LOCK).name,
        lock_sha256=hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
        n_locked=len(lock_packages(lock_text)),
        lock_text=lock_text,
    )


def router_cell() -> str:
    return ISOLATED_ROUTER


def imports_cell(repo: Path = REPO) -> str:
    resolved = resolved_pins(read_pins(repo), read_lock(repo))
    expected = "LOCKED_VERSIONS = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in resolved.items()) + "}"
    return IMPORTS_TEMPLATE.replace("{locked_versions}", expected)

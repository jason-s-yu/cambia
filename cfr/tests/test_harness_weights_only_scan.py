"""
tests/test_harness_weights_only_scan.py

Static scan (cambia-1718, serving-harness v1.1 design 5.7 D61): every
torch.load(...) call site under cfr/src must pass weights_only=True.
torch.load(..., weights_only=True) is the only thing between a node-authored
(or otherwise untrusted) .pt checkpoint and pickle execution on the client and
on every node that later evaluates it, so a call site missing it is a code
execution hole, not a style nit.

The scan parses each source file's AST rather than grepping lines, since a
torch.load call's keyword arguments routinely span multiple lines in this
codebase.
"""

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _iter_py_files(root: Path):
    return sorted(root.rglob("*.py"))


def _torch_load_calls(tree: ast.AST):
    """Yield every `torch.load(...)` ast.Call node in `tree`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            yield node


def _has_weights_only_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "weights_only":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def _scan(root: Path):
    """Return [(relative_path, lineno), ...] for every torch.load call under
    `root` missing weights_only=True. Files that fail to parse are skipped
    (not this scan's concern; other checks catch a syntax error)."""
    violations = []
    for path in _iter_py_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for call in _torch_load_calls(tree):
            if not _has_weights_only_true(call):
                violations.append((str(path.relative_to(root)), call.lineno))
    return violations


def test_every_torch_load_call_site_passes_weights_only_true():
    """AC (6): the scan passes on the current tree -- every torch.load(...)
    call under cfr/src carries weights_only=True."""
    violations = _scan(_SRC_ROOT)
    assert (
        not violations
    ), "torch.load call(s) missing weights_only=True (design 5.7 D61): " + ", ".join(
        f"{p}:{ln}" for p, ln in violations
    )


def test_scan_catches_a_planted_violation(tmp_path):
    """AC (6): the scan fails on a planted violation. Adversarial check on the
    scanner itself, so the acceptance test above cannot pass vacuously (e.g.
    an empty file list, or a matcher that never fires)."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import torch\n"
        "\n"
        "def bad():\n"
        "    return torch.load('checkpoint.pt', map_location='cpu')\n",
        encoding="utf-8",
    )
    violations = _scan(tmp_path)
    assert violations == [("planted.py", 4)]


def test_scan_ignores_a_compliant_call(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import torch\n"
        "\n"
        "def good():\n"
        "    return torch.load(\n"
        "        'checkpoint.pt', map_location='cpu', weights_only=True\n"
        "    )\n",
        encoding="utf-8",
    )
    assert _scan(tmp_path) == []


def test_scan_ignores_unrelated_load_calls(tmp_path):
    """A bare `load(...)` (e.g. json.load, yaml.load) must never be mistaken
    for torch.load: the matcher requires the `torch.` attribute prefix."""
    ok = tmp_path / "ok2.py"
    ok.write_text(
        "import json\n" "\n" "def load_config(fh):\n" "    return json.load(fh)\n",
        encoding="utf-8",
    )
    assert _scan(tmp_path) == []

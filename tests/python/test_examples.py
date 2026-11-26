# SPDX-License-Identifier: MIT
"""Every example and every Python code block of README.md and docs/FROM_FDTDZ.md runs (on a GPU)."""
import os
import re
import subprocess
import sys

import pytest

from conftest import ROOT

pytestmark = pytest.mark.gpu
EXAMPLES = os.path.join(ROOT, "examples")


def run(args, cwd):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.parametrize("name", ["quickstart.py", "coefficients.py", "from_fdtdz.py", "offload.py", "wisdom.py"])
def test_example(name, tmp_path):
    out = run([os.path.join(EXAMPLES, name)], tmp_path)
    if name == "wisdom.py":
        assert "from wisdom: False" in out
        assert "from wisdom: True" in run([os.path.join(EXAMPLES, name)], tmp_path)   # second run: from the file


@pytest.mark.gpu2
def test_example_multi_gpu(tmp_path):
    assert "bit-identical: True" in run([os.path.join(EXAMPLES, "multi_gpu.py")], tmp_path)


@pytest.mark.parametrize("doc", ["README.md", "docs/FROM_FDTDZ.md"])
def test_doc_code_blocks(doc, tmp_path):
    text = open(os.path.join(ROOT, doc)).read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks or doc != "README.md"
    for i, b in enumerate(blocks):
        p = tmp_path / f"block{i}.py"
        p.write_text(b)
        run([str(p)], tmp_path)

#!/usr/bin/env python3
"""Install RotateK's lmms-eval integration into an installed lmms-eval.

Copying the files under ``integrations/lmms_eval/`` is not enough on its own:
lmms-eval resolves ``--model <name>`` through a registry in
``lmms_eval/models/__init__.py``, so the wrappers this repo adds also have to be
registered there. And once copied into site-packages those wrappers still need to
``import rotatek``, which only resolves if the repo root is on ``sys.path``.

This script does all three, and is idempotent. One stock file is replaced
rather than added: ``tasks/_task_utils/file_utils.py`` gains a guard so that
TextVQA's submission-file hook works when ``simple_evaluate`` is called from
Python (no CLI ``args``); stock 0.5.0 crashes there after generation.

Usage:
    python integrations/apply_overrides.py            # install
    python integrations/apply_overrides.py --check    # verify only, change nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

# name -> class, as the eval scripts request them via get_model(...)
MODELS = {
    "llava_next_visionzip": "LlavaNext_VisionZip",
    "llava_next_fastv": "LlavaNext_FastV",
    "qwen2_5_vl_visionzip": "Qwen2_5_VL_VisionZip",
    "qwen2_5_vl_fastv": "Qwen2_5_VL_FastV",
}
MARK_BEGIN = "    # --- RotateK integration (added by integrations/apply_overrides.py) ---"
MARK_END = "    # --- end RotateK integration ---"


def lmms_eval_dir() -> Path:
    spec = importlib.util.find_spec("lmms_eval")
    if spec is None or not spec.origin:
        sys.exit("lmms-eval is not installed in this environment "
                 "(pip install lmms-eval==0.5.0)")
    return Path(spec.origin).parent


def copy_tree(src: Path, dst: Path) -> list[str]:
    copied = []
    for path in sorted(src.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(path.relative_to(src)))
    return copied


def register(init_py: Path, apply: bool) -> str:
    text = init_py.read_text()
    entries = "\n".join(f'    "{k}": "{v}",' for k, v in MODELS.items())
    block = f"{MARK_BEGIN}\n{entries}\n{MARK_END}\n"
    if MARK_BEGIN in text:
        # replace a block written by an earlier version of this script
        start = text.index(MARK_BEGIN)
        end = text.index(MARK_END, start) + len(MARK_END) + 1
        if text[start:end] == block:
            return "already registered"
        if apply:
            init_py.write_text(text[:start] + block + text[end:])
        return "updated registration (%d models)" % len(MODELS)
    # insert just before the closing brace of AVAILABLE_SIMPLE_MODELS
    m = re.search(r"AVAILABLE_SIMPLE_MODELS\s*=\s*\{.*?^\}", text, re.S | re.M)
    if not m:
        return "FAILED: could not find AVAILABLE_SIMPLE_MODELS"
    patched = text[: m.end() - 1] + block + text[m.end() - 1 :]
    if apply:
        shutil.copy2(init_py, init_py.with_suffix(".py.rotatek-backup"))
        init_py.write_text(patched)
    return "registered %d models" % len(MODELS)


def add_repo_to_path(dst: Path, repo: Path, apply: bool) -> str:
    """Drop a .pth so the copied wrappers can `import rotatek`.

    Site-packages is not the repo, so without this the wrappers fail with
    ModuleNotFoundError as soon as lmms-eval imports them. A .pth is what
    `pip install -e .` writes anyway, and it keeps this repo install-free.
    """
    pth = dst.parent / "rotatek.pth"
    if pth.exists() and pth.read_text().strip() == str(repo):
        return "already on sys.path"
    if apply:
        pth.write_text(str(repo) + "\n")
    return f"added {pth.name} -> {repo}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report what is missing without changing anything")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    src = here / "lmms_eval"
    dst = lmms_eval_dir()
    print(f"lmms-eval: {dst}")

    if args.check:
        missing = [str(p.relative_to(src)) for p in src.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and not (dst / p.relative_to(src)).exists()]
        init_text = (dst / "models" / "__init__.py").read_text()
        print(f"  files not yet installed : {len(missing)}")
        for f in missing[:10]:
            print(f"    {f}")
        print("  registry                :",
              "patched" if MARK_BEGIN in init_text else "NOT patched")
        unresolved = [n for n in MODELS if f'"{n}"' not in init_text]
        print("  models not registered   :", unresolved or "none")
        pth = dst.parent / "rotatek.pth"
        print("  repo on sys.path        :",
              "yes" if pth.exists() else "NO (wrappers will fail to import rotatek)")
        return

    copied = copy_tree(src, dst)
    print(f"  copied {len(copied)} files")
    print("  registry:", register(dst / "models" / "__init__.py", apply=True))
    print("  sys.path:", add_repo_to_path(dst, here.parent, apply=True))
    print("\nVerify with:  python integrations/apply_overrides.py --check")


if __name__ == "__main__":
    main()

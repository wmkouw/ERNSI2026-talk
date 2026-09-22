"""Build the static, in-browser version of bridge_demo.py into site/.

The notebook's local modules, data and stored results are zipped into
public/bundle.zip, which the notebook's first cell downloads and unpacks
when it runs under WASM. Then marimo exports the notebook, copying public/
along with it.

    python build_site.py                   # read-only app
    python build_site.py --mode edit       # code visible and editable
    python -m http.server -d site          # preview at localhost:8000
"""
import argparse
import os
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = ["bridge.py", "bridge_viz.py", "models", "data", "results"]


def bundle(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in BUNDLE:
            src = os.path.join(HERE, item)
            if os.path.isfile(src):
                zf.write(src, item)
                continue
            for root, dirs, files in os.walk(src):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    if not f.endswith(".pyc"):
                        p = os.path.join(root, f)
                        zf.write(p, os.path.relpath(p, HERE))
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "edit"], default="run")
    ap.add_argument("--out", default=os.path.join(HERE, "site"))
    args = ap.parse_args()

    z = bundle(os.path.join(HERE, "public", "bundle.zip"))
    print(f"wrote {z} ({os.path.getsize(z) / 1e6:.1f} MB)")
    subprocess.run(["marimo", "export", "html-wasm", os.path.join(HERE, "bridge_demo.py"),
                    "-o", args.out, "--mode", args.mode, "-f"], check=True)
    print(f"site in {args.out}; preview with: python -m http.server -d {args.out}")

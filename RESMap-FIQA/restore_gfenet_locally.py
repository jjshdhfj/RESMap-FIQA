#!/usr/bin/env python3
"""OPTIONAL LOCAL SETUP: copy GFENet files from a separately obtained checkout.
This command is for a user's own local workspace; no upstream code is included
in the distributed archive. Check the upstream's applicable terms beforehand.
"""
import argparse
import shutil
from pathlib import Path
FILES = [
 "models/__init__.py", "models/base_model.py", "models/gfenet_model.py",
 "models/networks.py", "models/backbone/gfenet_backbone.py",
 "models/guided_filter_pytorch/__init__.py",
 "models/guided_filter_pytorch/HFC_filter.py",
 "models/guided_filter_pytorch/box_filter.py",
 "models/guided_filter_pytorch/guided_filter.py",
 "models/guided_filter_pytorch/guided_filter_attention.py",
]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gfenet-source",required=True,type=Path,
                   help="A GFENet checkout you are permitted to use locally")
    args=p.parse_args()
    source=args.gfenet_source.expanduser().resolve()
    target=Path(__file__).resolve().parent/"resmap_generator"
    if not source.is_dir(): p.error("Missing source directory: "+str(source))
    missing=[rel for rel in FILES if not (source/rel).is_file()]
    if missing: p.error("Checkout missing required files: "+str(missing))
    exists=[rel for rel in FILES if (target/rel).exists()]
    if exists: p.error("Refusing to overwrite local existing files: "+str(exists))
    for rel in FILES:
        dest=target/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/rel,dest)
    print("[PASS] 10 GFENet source files installed locally (NOT for automatic re-distribution)")
if __name__=="__main__": main()

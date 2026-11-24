#!/usr/bin/env python3
"""Read-only runtime/package audit (not a clean installation test)."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

CHECKS = [('torch','torch'),('torchvision','torchvision'),('timm','timm'),
          ('numpy','numpy'),('pandas','pandas'),('scipy','scipy'),
          ('cv2','opencv-python'),('PIL','Pillow'),('yaml','PyYAML'),
          ('sklearn','scikit-learn'),('huggingface_hub','huggingface-hub'),
          ('filelock','filelock')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,default=Path('environment_audit.json'))
    args=ap.parse_args()
    result={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,
            'PYTHONPATH_present':bool(os.getenv('PYTHONPATH')),'packages':{},
            'cuda':{},'pip_check':{},'clean_install_verified':False}
    for module_name,pkg in CHECKS:
        row={}
        try: row['version']=metadata.version(pkg)
        except metadata.PackageNotFoundError: row['version']=None
        try:
            module=importlib.import_module(module_name)
            row['import']='PASS'
            row['loaded_from']=str(getattr(module,'__file__',None))
            if module_name=='torch':
                result['cuda']={'available':bool(module.cuda.is_available()),
                                'pytorch_cuda':module.version.cuda,
                                'gpu':module.cuda.get_device_name(0) if module.cuda.is_available() else None}
        except Exception as e:
            row['import']='FAIL';row['exception']='%s: %s'%(type(e).__name__,e)
        result['packages'][pkg]=row
        print('%-18s %-16s %s'%(pkg,str(row['version']),row['import']),flush=True)
    try:
        p=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=120)
        result['pip_check']={'exit_code':p.returncode,'stdout':p.stdout[-15000:],'stderr':p.stderr[-3000:]}
    except Exception as e:
        result['pip_check']={'exception':str(e)}
    path=args.report.expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    essentials=('torch','torchvision','timm','numpy','pandas','scipy','opencv-python','Pillow','PyYAML','scikit-learn','filelock')
    fail=[x for x in essentials if result['packages'][x]['import']!='PASS']
    if fail:print('[FAIL] Missing/broken essential imports:',', '.join(fail),flush=True)
    else:print('[PASS] Essential module imports (existing environment only)',flush=True)
    print('[PIP CHECK]',result['pip_check'].get('exit_code','ERROR'))
    print('[REPORT]',path)
    print('[LIMIT] This is NOT a clean-install or full 165450-step reproducibility verification.')
    if fail:sys.exit(2)
    if result['pip_check'].get('exit_code') not in (0,None):sys.exit(3)


if __name__=='__main__':main()

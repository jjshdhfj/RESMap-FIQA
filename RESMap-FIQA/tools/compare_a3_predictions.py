#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare historical official A3 results with the A3-only release outputs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-dir', required=True, type=Path,
                    help='Original locked_test_a0_a3 result directory')
    ap.add_argument('--new-dir', required=True, type=Path,
                    help='A3-only locked_test_a3_only result directory')
    ap.add_argument('--atol', default=1e-6, type=float,
                    help='Absolute tolerance for floating point outputs (default 1e-6)')
    args = ap.parse_args()
    base = 'A3_B15_TaskQuery_UMap_test_'
    a = pd.read_csv(args.original_dir / (base + 'predictions.csv'))
    b = pd.read_csv(args.new_dir / (base + 'predictions.csv'))
    assert len(a) == len(b) == 1227, (len(a), len(b))
    assert list(a.columns) == list(b.columns), 'CSV column lists differ'
    assert a['file_name'].astype(str).tolist() == b['file_name'].astype(str).tolist(), 'File order differs'
    print('[MATCH] 1227 rows, exact file-name order, same columns')
    for col in a.columns:
        if col == 'file_name':
            continue
        av = pd.to_numeric(a[col], errors='raise').to_numpy(dtype=np.float64)
        bv = pd.to_numeric(b[col], errors='raise').to_numpy(dtype=np.float64)
        d = float(np.max(np.abs(av - bv)))
        print('%-28s max_abs_diff=%.10g' % (col, d))
        if not np.allclose(av, bv, atol=args.atol, rtol=0, equal_nan=True):
            raise RuntimeError('Prediction mismatch at %s' % col)
    ja=json.loads((args.original_dir/(base+'metrics.json')).read_text(encoding='utf-8'))
    jb=json.loads((args.new_dir/(base+'metrics.json')).read_text(encoding='utf-8'))
    assert ja.keys() == jb.keys(), 'Metric key sets differ'
    for key in ja:
        x,y=ja[key],jb[key]
        if isinstance(x,(int,float)) and isinstance(y,(int,float)):
            if math.isnan(x) and math.isnan(y): continue
            if not math.isclose(x,y,rel_tol=0,abs_tol=args.atol):
                raise RuntimeError('Metric %s: original=%r new=%r'%(key,x,y))
        else:
            if x != y:raise RuntimeError('Metric %s differs'%key)
    print('PASS: predictions and metrics agree within atol=%g' % args.atol)


if __name__=='__main__':
    main()

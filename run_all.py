"""
Run all 6 analysis scripts in order.

Usage: python run_all.py

Order is intentional: 01 produces output/df_processed.csv, which is consumed
by 02-06.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'scripts/01_replication.py',
    'scripts/02_extension.py',
    'scripts/03_causal_inference.py',
    'scripts/04_model_comparison.py',
    'scripts/05_heterogeneity.py',
    'scripts/06_fd_redundancy.py',
]


def main():
    # Force UTF-8 stdout for child Python processes (some scripts print
    # box-drawing characters that fail under Windows cp1252).
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    total_start = time.time()
    for s in SCRIPTS:
        path = ROOT / s
        print(f"\n{'='*70}\n>>> Running {s}\n{'='*70}\n", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=env)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n!!! {s} FAILED (exit {result.returncode}) after {elapsed:.1f}s",
                  flush=True)
            sys.exit(result.returncode)
        print(f"\n>>> {s} completed in {elapsed:.1f}s", flush=True)
    print(f"\n{'='*70}\nAll scripts completed in {time.time()-total_start:.1f}s\n{'='*70}",
          flush=True)


if __name__ == '__main__':
    main()

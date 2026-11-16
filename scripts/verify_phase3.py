"""Entry point for the standalone numerical phase3 verifier."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':subprocess.run([sys.executable,str(ROOT/'experiments/phase3/scripts/verify_phase3.py')],cwd=ROOT,check=True)

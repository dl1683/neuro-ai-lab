"""
Runner: D2c + D2d Sequential Execution

Runs the stability analysis (D2c) and depth-matched sweep (D2d)
back-to-back. Launch after D2b completes to avoid GPU contention.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    t0 = time.time()

    print("=" * 70)
    print("PHASE 1: D2c Stability Analysis (D7: sigma_max/rho)")
    print("=" * 70)
    from exp_d2c_stability_analysis import run as run_d2c
    run_d2c()

    t1 = time.time()
    print("\nD2c completed in %.0f seconds" % (t1 - t0))

    print("\n" + "=" * 70)
    print("PHASE 2: D2d Depth-Matched Encoder Sweep (4L/8L x 5 seeds)")
    print("=" * 70)
    from exp_d2d_depth_sweep import run as run_d2d
    run_d2d()

    t2 = time.time()
    print("\nD2d completed in %.0f seconds" % (t2 - t1))
    print("Total D2c+D2d: %.0f seconds (%.1f hours)" % (t2 - t0, (t2 - t0) / 3600))


if __name__ == "__main__":
    main()

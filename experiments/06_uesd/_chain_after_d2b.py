"""Wait for D2b (PID from argv) to finish, then run D2c + D2d."""
import sys
import time
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def pid_alive(pid):
    """Windows-compatible process existence check."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid],
            capture_output=True, text=True,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python _chain_after_d2b.py <D2B_PID>")
        sys.exit(1)

    pid = int(sys.argv[1])
    print("Waiting for PID %d (D2b) to exit..." % pid, flush=True)

    while pid_alive(pid):
        time.sleep(30)

    print("D2b (PID %d) has exited. Starting D2c + D2d chain." % pid, flush=True)
    time.sleep(5)

    print("\n" + "=" * 70, flush=True)
    print("PHASE 1: D2c Stability Analysis", flush=True)
    print("=" * 70, flush=True)
    ret = subprocess.run(
        [sys.executable, str(THIS_DIR / "exp_d2c_stability_analysis.py")],
        cwd=str(THIS_DIR),
    )
    print("D2c exit code: %d" % ret.returncode, flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PHASE 2: D2d Depth Sweep", flush=True)
    print("=" * 70, flush=True)
    ret = subprocess.run(
        [sys.executable, str(THIS_DIR / "exp_d2d_depth_sweep.py")],
        cwd=str(THIS_DIR),
    )
    print("D2d exit code: %d" % ret.returncode, flush=True)

    print("\nAll experiments complete.", flush=True)


if __name__ == "__main__":
    main()

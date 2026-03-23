from __future__ import annotations

import os
import subprocess
import sys
from textwrap import dedent


def run_cmd(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip(), flush=True)
    if proc.stderr:
        print(proc.stderr.rstrip(), flush=True)
    print(f"[exit code: {proc.returncode}]", flush=True)
    return proc.returncode


def main() -> int:
    print("Python executable:", sys.executable, flush=True)
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV", "<unset>"), flush=True)
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        flush=True,
    )

    run_cmd(["nvidia-smi"])
    run_cmd([sys.executable, "-c", "import tensorflow as tf; print(tf.__file__)"])

    probe = dedent(
        """
        from __future__ import annotations

        import faulthandler
        import json
        import time

        import tensorflow as tf

        faulthandler.enable(all_threads=True)

        print(f"TF version: {tf.__version__}", flush=True)
        print(
            "Build info:",
            json.dumps(tf.sysconfig.get_build_info(), indent=2),
            flush=True,
        )

        gpus = tf.config.list_physical_devices("GPU")
        print(f"GPUs: {gpus}", flush=True)

        if not gpus:
            print("ERROR: TensorFlow did not detect a GPU.", flush=True)
            raise SystemExit(1)

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        with tf.device("/GPU:0"):
            a = tf.random.normal((8192, 8192), dtype=tf.float32)
            b = tf.random.normal((8192, 8192), dtype=tf.float32)

            print("Starting repeated GPU matmuls...", flush=True)
            start = time.perf_counter()

            c = None
            for i in range(20):
                c = tf.matmul(a, b)
                _ = c.numpy()
                print(f"Completed iteration {i + 1}/20 on {c.device}", flush=True)

            elapsed = time.perf_counter() - start

        if c is None:
            print("ERROR: No matmul result was produced.", flush=True)
            raise SystemExit(2)

        print(f"Matmul device: {c.device}", flush=True)
        print(f"Matmul shape: {c.shape}", flush=True)
        print(f"Total elapsed seconds: {elapsed:.4f}", flush=True)

        if "GPU:0" not in c.device:
            print(f"ERROR: Expected GPU:0 but got {c.device}", flush=True)
            raise SystemExit(3)

        print("SUCCESS: TensorFlow used the GPU.", flush=True)
        """
    ).strip()

    return run_cmd([sys.executable, "-X", "faulthandler", "-c", probe])


if __name__ == "__main__":
    raise SystemExit(main())

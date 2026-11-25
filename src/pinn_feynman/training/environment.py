"""Where a run was made: library versions, CPU model, threads.

The same configuration and seed can give different results on different machines; `summary.json` keeps this record
of the session that finished the run, so that such a difference can be explained.
"""

import platform
import sys
from pathlib import Path

import numpy as np
import torch


def cpu_model():
    """CPU model name: /proc/cpuinfo on Linux, `platform.processor()` elsewhere; 'unknown' if neither."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def runtime_environment():
    """{python, torch, numpy, cpu, logical_cpus, torch_threads, platform} of this process."""
    import os
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "cpu": cpu_model(),
            "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(), "platform": platform.platform()}

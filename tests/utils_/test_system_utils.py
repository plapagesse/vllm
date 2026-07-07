# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import tempfile
import time
from pathlib import Path

from vllm.utils.system_utils import (
    _maybe_force_spawn,
    arm_fatal_exit_watchdog,
    force_exit,
    unique_filepath,
)


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


# Spawn targets must be module-level to be picklable.


def _watchdog_then_hang():
    arm_fatal_exit_watchdog(reason="test")
    # Simulate cleanup that hangs; the watchdog must force-exit first.
    time.sleep(60)


def _watchdog_cancelled():
    timer = arm_fatal_exit_watchdog(reason="test")
    timer.cancel()
    # Outlive the (cancelled) timeout to prove it does not fire.
    time.sleep(2)


def _force_exit_with_code_3():
    force_exit(3, "test")


def _spawn(target) -> multiprocessing.process.BaseProcess:
    proc = multiprocessing.get_context("spawn").Process(target=target)
    proc.start()
    return proc


def test_fatal_exit_watchdog_force_exits_hung_process(monkeypatch):
    monkeypatch.setenv("VLLM_FATAL_EXIT_TIMEOUT_SECONDS", "1")
    proc = _spawn(_watchdog_then_hang)
    proc.join(30)
    assert proc.exitcode == 1


def test_fatal_exit_watchdog_cancel(monkeypatch):
    monkeypatch.setenv("VLLM_FATAL_EXIT_TIMEOUT_SECONDS", "1")
    proc = _spawn(_watchdog_cancelled)
    proc.join(30)
    assert proc.exitcode == 0


def test_force_exit_uses_exit_code():
    proc = _spawn(_force_exit_with_code_3)
    proc.join(30)
    assert proc.exitcode == 3

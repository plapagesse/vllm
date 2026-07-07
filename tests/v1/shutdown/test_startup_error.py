# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that we handle a startup Error and shutdown."""

import inspect
import time

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.shutdown.utils import (
    SHUTDOWN_TEST_THRESHOLD_BYTES,
    SHUTDOWN_TEST_TIMEOUT_SEC,
)
from vllm import LLM
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore

MODELS = ["hmellor/tiny-random-LlamaForCausalLM"]


def evil_method(self, *args, **kwargs):
    """Evil method that raises an exception."""

    if get_tensor_model_parallel_rank() == 0:
        raise Exception("Simulated Error in startup!")

    return self.model(*args, **kwargs, intermediate_tensors=None)


@pytest.fixture
def rocm_evil_method(rocm_sitecustomize_factory, request):
    failing_method = request.getfixturevalue("failing_method")
    lines = [
        "from vllm.distributed import get_tensor_model_parallel_rank",
        "from vllm.model_executor.models.llama import LlamaForCausalLM",
        inspect.getsource(evil_method),
        f"LlamaForCausalLM.{failing_method} = {evil_method.__name__}",
    ]
    rocm_sitecustomize_factory(lines)


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("failing_method", ["forward", "load_weights"])
def test_async_llm_startup_error(
    monkeypatch,
    rocm_evil_method,
    model: str,
    tensor_parallel_size: int,
    failing_method: str,
) -> None:
    """Test that AsyncLLM propagates an __init__ error & frees memory.
    Test profiling (forward()) and load weights failures.
    AsyncLLM always uses an MP client.
    """
    if current_platform.device_count() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    # Monkeypatch an error in the model.
    monkeypatch.setattr(LlamaForCausalLM, failing_method, evil_method)

    engine_args = AsyncEngineArgs(
        model=model, enforce_eager=True, tensor_parallel_size=tensor_parallel_size
    )

    # Confirm we get an exception.
    with pytest.raises(Exception, match=r"initialization fail(ed|ure)"):
        _ = AsyncLLM.from_engine_args(engine_args)

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=list(range(tensor_parallel_size)),
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
    )


def hanging_shutdown(self):
    """Simulates EngineCore cleanup wedging in distributed teardown after a
    fatal startup error (e.g. NCCL/DeepEP native hang, see #46509)."""
    time.sleep(300)


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="Relies on fork-based monkeypatching of the engine-core process",
)
@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
def test_async_llm_startup_error_exits_despite_hanging_shutdown(
    monkeypatch,
    model: str,
) -> None:
    """A fatal startup error whose cleanup hangs must still kill the
    EngineCore process (via the fatal-exit watchdog) so that the frontend
    observes startup failure instead of blocking forever (#46509)."""
    # Fail during weight loading, and make the engine's cleanup hang.
    monkeypatch.setattr(LlamaForCausalLM, "load_weights", evil_method)
    monkeypatch.setattr(EngineCore, "shutdown", hanging_shutdown)
    monkeypatch.setenv("VLLM_FATAL_EXIT_TIMEOUT_SECONDS", "5")

    engine_args = AsyncEngineArgs(model=model, enforce_eager=True)

    with pytest.raises(Exception, match=r"initialization fail(ed|ure)"):
        _ = AsyncLLM.from_engine_args(engine_args)

    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
    )


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("enable_multiprocessing", [True])
@pytest.mark.parametrize("failing_method", ["forward", "load_weights"])
def test_llm_startup_error(
    monkeypatch,
    rocm_evil_method,
    model: str,
    tensor_parallel_size: int,
    enable_multiprocessing: bool,
    failing_method: str,
) -> None:
    """Test that LLM propagates an __init__ error and frees memory.
    Test profiling (forward()) and load weights failures.
    TODO(andy) - LLM without multiprocessing.
    """
    # Skip non-Llama models since we monkeypatch LlamaForCausalLM specifically.
    # If MODELS list grows, each architecture needs its own test variant.
    if model != "JackFram/llama-68m":
        pytest.skip(reason="Only test JackFram/llama-68m")
    if current_platform.device_count() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        MP_VALUE = "1" if enable_multiprocessing else "0"
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", MP_VALUE)

        # Monkeypatch an error in the model.
        monkeypatch.setattr(LlamaForCausalLM, failing_method, evil_method)

        with pytest.raises(
            Exception,
            match=r"initialization fail(ed|ure)"
            if enable_multiprocessing
            else "Simulated Error in startup!",
        ):
            _ = LLM(
                model=model,
                enforce_eager=True,
                tensor_parallel_size=tensor_parallel_size,
            )

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
        )

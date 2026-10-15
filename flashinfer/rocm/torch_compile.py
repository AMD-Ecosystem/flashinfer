# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ``torch.library`` custom-op registration.

Real registration happens behind FLASHINFER_USE_TORCH_CUSTOM_OPS=1; otherwise
ops carry a guard that raises under torch.compile instead of letting TorchDynamo
trace into an extension. utils.py keeps upstream's unguarded no-op below
torch 2.4, so the guard only exists from 2.4 up.
"""

import contextlib
import functools
import os
import re
import threading
import warnings
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

# torch.library.custom_op adds dispatch overhead, which is why upstream leaves it
# off: https://github.com/vllm-project/vllm/blob/36e76700453924c8d421db99af70a88a1df835cd/vllm/utils.py#L1660-L1674
_USE_TORCH_CUSTOM_OPS = TorchVersion(torch_version) >= TorchVersion(
    "2.4"
) and os.environ.get("FLASHINFER_USE_TORCH_CUSTOM_OPS", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def use_torch_custom_ops_enabled() -> bool:
    """Whether opaque ``torch.library`` custom ops are active."""
    return _USE_TORCH_CUSTOM_OPS


def _guard_compile(f: Callable, op_name: str) -> Callable:
    """Wrap ``f`` to raise if traced by torch.compile while registration is off."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if torch.compiler.is_compiling():
            raise RuntimeError(
                f"torch.compile traced into flashinfer op '{op_name}' but "
                "custom ops are not enabled. Set the environment variable "
                "FLASHINFER_USE_TORCH_CUSTOM_OPS=1 before importing "
                "flashinfer to use torch.compile."
            )
        return f(*args, **kwargs)

    return wrapper


def register_custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: Optional[Union[str, Sequence[str]]] = None,
    schema: Optional[str] = None,
) -> Callable:
    def decorator(f: Callable) -> Callable:
        if not _USE_TORCH_CUSTOM_OPS:
            return _guard_compile(f, name)
        try:
            return torch.library.custom_op(
                name,
                f,
                mutates_args=mutates_args,
                device_types=device_types,
                schema=schema,
            )
        except (ValueError, TypeError):
            # Schema inference rejects some parameter types, e.g.
            # Optional[torch.Generator]. Fall back to the guard so tracing
            # still fails loudly instead of entering the extension.
            warnings.warn(
                f"Could not register '{name}' as a torch.library custom op "
                "(unsupported parameter type in schema inference); falling back "
                "to compile guard. torch.compile will raise a RuntimeError if it "
                "traces into this op.",
                stacklevel=2,
            )
            return _guard_compile(f, name)

    if fn is not None:
        return decorator(fn)
    return decorator


def register_fake_op(
    name: str,
    fn: Optional[Callable] = None,
) -> Callable:
    def decorator(f: Callable) -> Callable:
        if _USE_TORCH_CUSTOM_OPS:
            with contextlib.suppress(Exception):
                torch.library.register_fake(name, f)
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


# One registration per op name, with the writes it declared.
_EXTENSION_OPS: Dict[str, Tuple[Tuple[str, ...], Callable]] = {}
_EXTENSION_OPS_LOCK = threading.Lock()


def extension_custom_op(
    c_op: "torch._ops.OpOverload", mutated: Tuple[str, ...]
) -> Callable:
    """``c_op`` as a custom op whose schema declares its writes, when the flag is on.

    Extension schemas carry no alias annotations, so compiled code would not know
    the ``mutated`` arguments are written. Flag off, ``c_op`` comes back unchanged.
    Registered once per op: torch silently replaces a second registration.
    """
    if not _USE_TORCH_CUSTOM_OPS:
        return c_op
    name = "flashinfer::" + c_op._schema.name.replace("::", "__")
    with _EXTENSION_OPS_LOCK:
        if name in _EXTENSION_OPS:
            registered, op = _EXTENSION_OPS[name]
            if registered != mutated:
                raise ValueError(f"{name} already declares writes to {registered}")
            return op
        else:
            schema = str(c_op._schema)
            schema = schema[schema.index("(") :]
            for alias, arg in enumerate(mutated):
                schema, n = re.subn(
                    rf"Tensor(\??) {arg}\b",
                    rf"Tensor({chr(ord('a') + alias)}!)\1 {arg}",
                    schema,
                )
                if n != 1:
                    raise ValueError(
                        f"{c_op} has no Tensor argument named {arg!r}: {schema}"
                    )
            op = torch.library.custom_op(
                name,
                lambda *args: c_op(*args),
                mutates_args=mutated,
                schema=schema,
            )
            _EXTENSION_OPS[name] = (mutated, op)
            return op

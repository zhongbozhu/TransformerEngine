# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mega C++ grouped MLP forward fuser."""

from __future__ import annotations
from collections.abc import Iterable
import functools
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...quantization import Recipe
from ...tensor import Quantizer
from ...utils import get_device_compute_capability
from ..basic import GroupedLinear, ScaledSReLU, ScaledClampedQGeGLU, ScaledSwiGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import validate_grouped_mlp_dims


def _megacpp_activation_config(activation) -> tuple[str, int, float, float, float]:
    """Return activation parameters consumed by the C++ grouped MLP path."""
    glu_interleave_size = int(getattr(activation, "glu_interleave_size", None) or 0)
    if isinstance(activation, ScaledSwiGLU):
        return "swiglu", glu_interleave_size, 0.0, 0.0, 0.0
    if isinstance(activation, ScaledClampedQGeGLU):
        return (
            "clamped_swiglu",
            glu_interleave_size,
            float(activation._clamped.limit),
            float(activation._clamped.alpha),
            float(activation._clamped.glu_linear_offset),
        )
    if isinstance(activation, ScaledSReLU):
        return "srelu", glu_interleave_size, 0.0, 0.0, 0.0
    raise TypeError(f"Unsupported megacpp grouped MLP activation {activation.__class__.__name__}.")


def _megacpp_weight_arg(
    linear_op: GroupedLinear,
    dtype: torch.dtype,
    *,
    input_requires_grad: bool,
) -> torch.Tensor | list[torch.Tensor]:
    """Return GEMM-ready high-precision weights for the current C++ path.

    Keep the layout policy in GroupedLinear. This handles quantized weights the
    same way as the Python grouped GEMM path: BF16/FP16 compute dequantizes when
    needed, while a future quantized-compute path can preserve quantized weights
    by switching ``with_quantized_compute``.
    """
    with_quantized_compute = False
    if linear_op.single_grouped_weight:
        grouped_weight = linear_op._get_grouped_weight_for_gemm(
            linear_op.weight,
            [linear_op.get_quantizer("forward", 1)],
            columnwise_usage=input_requires_grad,
            with_quantized_compute=with_quantized_compute,
            dtype=dtype,
        )
        if grouped_weight.rowwise_data is None:
            raise RuntimeError("megacpp grouped MLP expected dense grouped weight rowwise_data.")
        return grouped_weight.rowwise_data.view(
            linear_op.num_groups,
            linear_op.out_features,
            linear_op.in_features,
        )
    return linear_op._get_discrete_weights_for_gemm(
        [getattr(linear_op, f"weight{idx}") for idx in range(linear_op.num_groups)],
        [linear_op.get_quantizer("forward", 2 * idx + 1) for idx in range(linear_op.num_groups)],
        columnwise_usage=input_requires_grad,
        with_quantized_compute=with_quantized_compute,
        dtype=dtype,
    )


def _megacpp_bias_arg(linear_op: GroupedLinear, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Return a packed [G, N] high-precision bias tensor or None."""
    grouped_bias = linear_op._get_grouped_bias_for_gemm(dtype)
    if grouped_bias is None:
        return None
    return grouped_bias.rowwise_data.view(linear_op.num_groups, linear_op.out_features)


class ForwardGroupedMLP_MegaCpp(FusedOperation):
    """Experimental BF16/FP16 grouped MLP forward implemented in C++.

    The C++ function returns plain tensors only. Python still owns autograd
    context layout; delayed wgrad is rejected by the matching backward op.
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether the C++ grouped MLP path can be dispatched."""
        if not torch.cuda.is_available():
            return False
        if get_device_compute_capability()[0] < 10:
            return False
        return hasattr(tex, "megacpp_grouped_mlp_forward")

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        activation: Optional[FusibleOperation],
        fc2: GroupedLinear,
    ) -> None:
        if activation is None:
            raise TypeError("Expected a grouped MLP activation op.")
        super().__init__((fc1, activation, fc2))
        validate_grouped_mlp_dims(fc1, activation, fc2)
        _megacpp_activation_config(activation)
        if fc1._scale_bias or fc2._scale_bias:
            raise RuntimeError("megacpp grouped MLP does not support scale_bias yet.")

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        del prev_op_grad_output_quantizer, next_op_input_quantizer, basic_op_kwargs
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs
        num_groups = fc1_op.num_groups

        split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            split_sizes.size() != fc2_split_sizes.size()
            or split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(f"{self.__class__.__name__} got different split sizes for FC1/FC2.")
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, got {int(split_sizes.numel())}.")

        scales = basic_op_extra_inputs[1][0]
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else fc1_weight_param.dtype
        )
        if dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"megacpp grouped MLP supports BF16/FP16 only, got {dtype}.")

        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        fc1_weight_requires_grad = requires_grad and fc1_weight_param.requires_grad
        fc2_weight_requires_grad = requires_grad and fc2_weight_param.requires_grad

        activation_name, glu_interleave_size, act_limit, act_alpha, act_offset = (
            _megacpp_activation_config(activation_op)
        )
        fc1_weights = _megacpp_weight_arg(
            fc1_op,
            dtype,
            input_requires_grad=input_requires_grad,
        )
        fc2_weights = _megacpp_weight_arg(
            fc2_op,
            dtype,
            input_requires_grad=input_requires_grad,
        )
        (
            fc2_out,
            x,
            split_sizes_i64,
            base_split_offsets,
            x_offsets,
            fc1_offsets,
            fc2_offsets,
            fc2_dy_offsets,
            fc1_activation_input,
            fc2_x,
        ) = tex.megacpp_grouped_mlp_forward(
            input_.to(dtype=dtype),
            split_sizes,
            fc1_weights,
            _megacpp_bias_arg(fc1_op, dtype),
            fc2_weights,
            _megacpp_bias_arg(fc2_op, dtype),
            scales,
            activation_name,
            glu_interleave_size,
            act_limit,
            act_alpha,
            act_offset,
        )

        if x.data_ptr() == input_.data_ptr():
            x._do_not_clear = True

        if requires_grad:
            fc1_saved_weights = [fc1_weights] if isinstance(fc1_weights, torch.Tensor) else fc1_weights
            fc2_saved_weights = [fc2_weights] if isinstance(fc2_weights, torch.Tensor) else fc2_weights

            fc1_ctx.save_for_backward(
                split_sizes_i64,
                base_split_offsets,
                x_offsets,
                fc1_offsets,
                x,
                fc1_activation_input,
                *fc1_saved_weights,
            )
            fc1_ctx.use_megacpp_grouped_mlp = True
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = fc1_weight_requires_grad
            fc1_ctx.single_weight_arg = isinstance(fc1_weights, torch.Tensor)

            activation_ctx.save_for_backward(fc1_activation_input, scales)
            activation_ctx.extra_input_requires_grad = True
            activation_ctx.input_requires_grad = True
            activation_ctx.dtype = dtype

            fc2_ctx.save_for_backward(
                split_sizes_i64,
                base_split_offsets,
                fc2_offsets,
                fc2_dy_offsets,
                fc2_x,
                *fc2_saved_weights,
            )
            fc2_ctx.use_megacpp_grouped_mlp = True
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = fc2_weight_requires_grad
            fc2_ctx.single_weight_arg = isinstance(fc2_weights, torch.Tensor)

        return fc2_out, [(), (), ()]


def fuse_forward_megacpp_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply opt-in C++ grouped MLP fusion for BF16/FP16."""
    if not int(os.getenv("NVTE_MEGACPP_GROUPED_LINEAR", "0")):
        return ops
    if recipe is not None or not ForwardGroupedMLP_MegaCpp.is_supported():
        return ops

    out = []
    window, ops = ops[:3], ops[3:]
    activation_types = (ScaledSwiGLU, ScaledClampedQGeGLU, ScaledSReLU)
    while len(window) == 3:
        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], activation_types)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif (
            window[0]._scale_bias
            or window[2]._scale_bias
        ):
            matches_pattern = False
        else:
            try:
                validate_grouped_mlp_dims(window[0], window[1], window[2])
                _megacpp_activation_config(window[1])
            except (TypeError, ValueError, RuntimeError):
                matches_pattern = False

        if matches_pattern:
            window = [
                ForwardGroupedMLP_MegaCpp(
                    fc1=window[0],
                    activation=window[1],
                    fc2=window[2],
                )
            ]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out


# Explicit env opt-in should win over other BF16/FP16 grouped-MLP fusers. When
# the env var is unset, this fuser returns the ops unchanged and does not affect
# lower-priority fusers.
register_forward_fusion(fuse_forward_megacpp_ops, prepend=True)

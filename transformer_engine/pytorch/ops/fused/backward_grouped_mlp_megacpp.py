# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mega C++ grouped MLP backward fuser."""

from __future__ import annotations
import functools
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...quantization import Recipe
from ...tensor.grouped_tensor import GroupedTensor
from ...utils import clear_tensor_data, get_device_compute_capability
from ...triton.grouped_dbias_dscales import compute_grouped_dbias
from ..basic import GroupedLinear, ScaledSReLU, ScaledClampedQGeGLU, ScaledSwiGLU
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    validate_grouped_mlp_dims,
    view_main_grad_as_grouped_buffer,
)
from .backward_grouped_mlp import _compute_grad_params
from .forward_grouped_mlp_megacpp import _megacpp_activation_config


def _megacpp_saved_weight_arg(
    saved_tensors: tuple[torch.Tensor, ...],
    *,
    single_weight_arg: bool,
    num_groups: int,
) -> tuple[torch.Tensor | list[torch.Tensor], tuple[torch.Tensor, ...]]:
    """Unpack saved C++ weight argument in the same shape used by forward."""
    if single_weight_arg:
        return saved_tensors[0], saved_tensors[1:]
    return list(saved_tensors[:num_groups]), saved_tensors[num_groups:]


def _wrap_plain_tensor_for_python_wgrad(
    data: torch.Tensor,
    split_sizes: torch.Tensor,
    tensor_offsets: torch.Tensor,
    *,
    num_groups: int,
    last_dim: int,
    dtype: torch.dtype,
) -> GroupedTensor:
    """Wrap a C++ BF16/FP16 activation buffer for Python-owned wgrad.

    Immediate weight-gradient GEMMs run in C++. Delayed wgrad intentionally
    stays in Python so WeightGradStore can defer the GEMM. That Python path
    consumes GroupedTensor operands, so these wrappers are metadata views over
    plain dense buffers, not returned user-visible tensors.
    """
    data_2d = data.reshape(-1, last_dim)
    return GroupedTensor(
        shape=(data_2d.size(0), last_dim),
        dtype=dtype,
        num_tensors=num_groups,
        quantizer=None,
        data=data_2d.reshape(-1),
        first_dims=split_sizes,
        tensor_offsets=tensor_offsets,
    )


def _compute_bias_grad_params_for_python_wgrad(
    fc_op: GroupedLinear,
    dy_2d: torch.Tensor,
    base_offsets: torch.Tensor,
    *,
    num_groups: int,
    dtype: torch.dtype,
) -> tuple[Optional[list[torch.Tensor]], Optional[torch.Tensor]]:
    """Compute bias grads in the layout expected by Python-owned wgrad plumbing."""
    if not fc_op.has_bias:
        return None, None
    dbias_packed = compute_grouped_dbias(dy_2d, base_offsets, num_groups).to(dtype=dtype)
    if fc_op.single_grouped_bias:
        return None, dbias_packed
    return [dbias_packed[idx] for idx in range(num_groups)], None


def _delay_wgrad(fc_op: GroupedLinear, ctx: OperationContext) -> bool:
    """Whether this FC op must use the Python delayed-wgrad path."""
    return bool(
        ctx.weight_requires_grad
        and fc_op.wgrad_store is not None
        and fc_op.wgrad_store.delay_wgrad_compute()
    )


def _prepare_cpp_wgrad_output(
    fc_op: GroupedLinear,
    ctx: OperationContext,
    *,
    num_groups: int,
    weight_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> tuple[Optional[torch.Tensor | list[torch.Tensor]], bool, list[Optional[torch.Tensor]]]:
    """Allocate or fetch the immediate wgrad output buffer for C++.

    Delay-wgrad deliberately returns ``None`` for the C++ output so Python can
    enqueue the existing delayed GEMM path after dgrad has been computed.
    """
    weights = fc_op._get_weight_tensors()
    weight_grads: list[Optional[torch.Tensor]] = (
        [None] if fc_op.single_grouped_weight else [None] * num_groups
    )
    if not ctx.weight_requires_grad or _delay_wgrad(fc_op, ctx):
        return None, False, weight_grads

    accumulate_into_main_grad = False
    if fc_op.single_grouped_weight:
        if fc_op._accumulate_into_main_grad:
            main_grad = get_main_grad_from_param(weights[0], op_label=label)
            wgrad_output = view_main_grad_as_grouped_buffer(
                main_grad,
                num_groups,
                weight_shape,
                label=f"{label} weight",
            )
            accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
        else:
            wgrad_output = torch.empty(
                (num_groups, *weight_shape),
                dtype=dtype,
                device=device,
            )
        weight_grads = [wgrad_output]
    else:
        if fc_op._accumulate_into_main_grad:
            wgrad_output = [get_main_grad_from_param(w, op_label=label) for w in weights]
            accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
        else:
            packed_wgrad = torch.empty(
                (num_groups, *weight_shape),
                dtype=dtype,
                device=device,
            )
            wgrad_output = [packed_wgrad[idx] for idx in range(num_groups)]
        weight_grads = list(wgrad_output)

    if fc_op._accumulate_into_main_grad:
        weight_grads = get_dummy_wgrads_for_params(weights)
    return wgrad_output, accumulate_into_main_grad, weight_grads


def _assemble_grad_params(
    fc_op: GroupedLinear,
    weight_grads: list[Optional[torch.Tensor]],
    bias_grads: Optional[list[torch.Tensor]],
    bias_grad_packed: Optional[torch.Tensor],
    *,
    num_groups: int,
) -> list[Optional[torch.Tensor]]:
    """Assemble parameter grads in GroupedLinear registration order."""
    if not fc_op.has_bias:
        return weight_grads
    if fc_op.single_grouped_bias:
        return weight_grads + [bias_grad_packed]
    bias_list = bias_grads if bias_grads is not None else [None] * num_groups
    if fc_op.single_grouped_weight:
        return bias_list + weight_grads
    return weight_grads + bias_list


class BackwardGroupedMLP_MegaCpp(FusedOperation):
    """Experimental C++ grouped MLP backward for BF16/FP16.

    Immediate weight gradients are computed in C++; delayed wgrad stays on the
    existing Python path so WeightGradStore can preserve its deferred execution
    semantics.
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        if not torch.cuda.is_available():
            return False
        if get_device_compute_capability()[0] < 10:
            return False
        return hasattr(tex, "megacpp_grouped_mlp_backward")

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

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        **unused,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs
        num_groups = fc1_op.num_groups
        dtype = fc1_ctx.dtype
        device = fc1_op._get_weight_tensors()[0].device

        fc1_saved = fc1_ctx.saved_tensors
        split_sizes, base_offsets, x_offsets, fc1_offsets = fc1_saved[:4]
        x, fc1_activation_input = fc1_saved[4:6]
        fc1_weight_arg, _ = _megacpp_saved_weight_arg(
            fc1_saved[6:],
            single_weight_arg=bool(getattr(fc1_ctx, "single_weight_arg", False)),
            num_groups=num_groups,
        )

        _, scales = activation_ctx.saved_tensors

        fc2_saved = fc2_ctx.saved_tensors
        fc2_offsets = fc2_saved[2]
        fc2_dy_offsets = fc2_saved[3]
        fc2_x = fc2_saved[4]
        fc2_weight_arg, _ = _megacpp_saved_weight_arg(
            fc2_saved[5:],
            single_weight_arg=bool(getattr(fc2_ctx, "single_weight_arg", False)),
            num_groups=num_groups,
        )

        activation_name, glu_interleave_size, act_limit, act_alpha, act_offset = (
            _megacpp_activation_config(activation_op)
        )
        fc1_wgrad_output, fc1_accumulate_wgrad, fc1_weight_grads = _prepare_cpp_wgrad_output(
            fc1_op,
            fc1_ctx,
            num_groups=num_groups,
            weight_shape=(fc1_op.out_features, fc1_op.in_features),
            dtype=dtype,
            device=device,
            label="Grouped MLP megacpp backward (FC1)",
        )
        fc2_wgrad_output, fc2_accumulate_wgrad, fc2_weight_grads = _prepare_cpp_wgrad_output(
            fc2_op,
            fc2_ctx,
            num_groups=num_groups,
            weight_shape=(fc2_op.out_features, fc2_op.in_features),
            dtype=dtype,
            device=device,
            label="Grouped MLP megacpp backward (FC2)",
        )
        grad_input, fc1_dy, fc2_dx, grad_scales = tex.megacpp_grouped_mlp_backward(
            grad_output.to(dtype=dtype),
            split_sizes,
            x_offsets,
            fc1_offsets,
            fc2_offsets,
            fc2_dy_offsets,
            base_offsets,
            x,
            fc1_activation_input,
            fc2_x,
            scales,
            fc1_weight_arg,
            fc2_weight_arg,
            fc1_wgrad_output,
            fc1_accumulate_wgrad,
            fc2_wgrad_output,
            fc2_accumulate_wgrad,
            activation_name,
            glu_interleave_size,
            act_limit,
            act_alpha,
            act_offset,
            bool(fc1_ctx.input_requires_grad),
        )
        if not fc1_ctx.input_requires_grad:
            grad_input = None

        grad_output_2d = grad_output.reshape(-1, fc2_op.out_features).to(dtype=dtype)
        fc2_bias_grads, fc2_bias_grad_packed = _compute_bias_grad_params_for_python_wgrad(
            fc2_op,
            grad_output_2d,
            base_offsets,
            num_groups=num_groups,
            dtype=dtype,
        )
        fc1_bias_grads, fc1_bias_grad_packed = _compute_bias_grad_params_for_python_wgrad(
            fc1_op,
            fc1_dy,
            base_offsets,
            num_groups=num_groups,
            dtype=dtype,
        )

        if _delay_wgrad(fc2_op, fc2_ctx):
            grouped_fc2_x = _wrap_plain_tensor_for_python_wgrad(
                fc2_x,
                split_sizes,
                fc2_offsets,
                num_groups=num_groups,
                last_dim=fc2_op.in_features,
                dtype=dtype,
            )
            grouped_fc2_dy = _wrap_plain_tensor_for_python_wgrad(
                grad_output_2d,
                split_sizes,
                fc2_dy_offsets,
                num_groups=num_groups,
                last_dim=fc2_op.out_features,
                dtype=dtype,
            )
            fc2_grad_params = _compute_grad_params(
                fc_op=fc2_op,
                ctx=fc2_ctx,
                num_groups=num_groups,
                weight_shape=(fc2_op.out_features, fc2_op.in_features),
                grouped_x=grouped_fc2_x,
                grouped_dy=grouped_fc2_dy,
                dtype=dtype,
                device=device,
                bias_grads=fc2_bias_grads,
                bias_grad_packed=fc2_bias_grad_packed,
                label="FC2",
                cudnn_wgrad_kernel_fn=None,
                offsets=None,
            )
        else:
            fc2_grad_params = _assemble_grad_params(
                fc2_op,
                fc2_weight_grads,
                fc2_bias_grads,
                fc2_bias_grad_packed,
                num_groups=num_groups,
            )

        if not _delay_wgrad(fc2_op, fc2_ctx):
            clear_tensor_data(fc2_x)

        if _delay_wgrad(fc1_op, fc1_ctx):
            grouped_fc1_x = _wrap_plain_tensor_for_python_wgrad(
                x,
                split_sizes,
                x_offsets,
                num_groups=num_groups,
                last_dim=fc1_op.in_features,
                dtype=dtype,
            )
            grouped_fc1_dy = _wrap_plain_tensor_for_python_wgrad(
                fc1_dy,
                split_sizes,
                fc1_offsets,
                num_groups=num_groups,
                last_dim=fc1_op.out_features,
                dtype=dtype,
            )
            fc1_grad_params = _compute_grad_params(
                fc_op=fc1_op,
                ctx=fc1_ctx,
                num_groups=num_groups,
                weight_shape=(fc1_op.out_features, fc1_op.in_features),
                grouped_x=grouped_fc1_x,
                grouped_dy=grouped_fc1_dy,
                dtype=dtype,
                device=device,
                bias_grads=fc1_bias_grads,
                bias_grad_packed=fc1_bias_grad_packed,
                label="FC1",
                cudnn_wgrad_kernel_fn=None,
                offsets=None,
            )
        else:
            fc1_grad_params = _assemble_grad_params(
                fc1_op,
                fc1_weight_grads,
                fc1_bias_grads,
                fc1_bias_grad_packed,
                num_groups=num_groups,
            )

        if not _delay_wgrad(fc1_op, fc1_ctx):
            clear_tensor_data(x)

        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [(None,), (grad_scales.to(dtype=dtype),), (None,)],
        )


def fuse_backward_megacpp_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply opt-in C++ grouped MLP backward fusion for BF16/FP16."""
    if recipe is not None or not BackwardGroupedMLP_MegaCpp.is_supported():
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
            getattr(window[0], "grouped_mlp_backend", "python") != "megacpp"
            or getattr(window[2], "grouped_mlp_backend", "python") != "megacpp"
            or window[0]._scale_bias
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
                BackwardGroupedMLP_MegaCpp(
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


register_backward_fusion(fuse_backward_megacpp_ops, prepend=True)

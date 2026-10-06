# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN grouped GEMM calls using TE's physical tensor buffers.

Older frontends require PyTorch tensors with the kernel's logical layout. Newer
frontends accept the same buffers and describe that layout inside CuTe instead.
Capability discovery belongs to layer construction, never to a GEMM invocation.
"""

from __future__ import annotations

from functools import lru_cache

import torch


def _supports_tensor_layouts(kernel) -> bool:
    """Whether a frontend callable implements the raw-buffer layout contract."""
    return kernel is not None and bool(getattr(kernel, "supports_tensor_layouts", False))


@lru_cache(maxsize=128)
def _matrix_layout(rows, cols, dtype):
    return ((rows, cols, 1), (cols, 1, rows * cols), dtype)


@lru_cache(maxsize=128)
def _weight_layout(groups, rows, cols, transposed):
    if transposed:
        return ((cols, rows, groups), (1, cols, rows * cols), torch.float8_e4m3fn)
    return ((rows, cols, groups), (cols, 1, rows * cols), torch.float8_e4m3fn)


@lru_cache(maxsize=128)
def _scale_layout(groups, rows, cols):
    """Logical MMA view of a physically swizzled MXFP8 scale buffer."""
    row_tiles, col_tiles = (rows + 127) // 128, (cols + 127) // 128
    return (
        (32, 4, row_tiles, 4, col_tiles, groups),
        (16, 4, col_tiles * 512, 1, 512, row_tiles * col_tiles * 512),
        torch.float8_e8m0fnu,
    )


def _legacy_operand(tensor, layout):
    shape, strides, dtype = layout
    if tensor.dtype != dtype:
        tensor = tensor.view(dtype=dtype)
    return tensor.as_strided(shape, strides)


def _physical_outputs(result):
    """Undo legacy FE views once, at the compatibility boundary."""
    for name in ("sfd_row_tensor", "sfd_col_tensor", "sfd_col_d_srelu_tensor"):
        value = result.get(name)
        if value is not None:
            result[name] = value.permute(5, 2, 4, 0, 1, 3).reshape(-1)
    for name in ("d_tensor", "d_row_tensor", "d_col_tensor", "d_srelu_tensor", "dprob_tensor"):
        value = result.get(name)
        if value is not None:
            result[name] = value.reshape(-1)
    value = result.get("c_tensor")
    if value is not None:
        result["c_tensor"] = value.squeeze(-1)
    return result


class CudnnGroupedGemm:
    """Layer-local dispatch for MXFP8 grouped MLP GEMMs.

    Only immutable layout metadata is cached; tensors, gathered weights and
    output buffers always belong to the current invocation. Non-MXFP8 recipes
    retain their existing frontend interfaces.
    """

    def __init__(self, activation, dactivation, quant, wgrad):
        self.activation_kernel = activation
        self.dactivation_kernel = dactivation
        self.quant_kernel = quant
        self.wgrad_kernel = wgrad
        self.use_native_layouts = all(
            _supports_tensor_layouts(kernel) for kernel in (activation, dactivation, quant)
        )
        self.use_native_wgrad_layouts = _supports_tensor_layouts(wgrad)

    def __call__(
        self,
        kernel,
        *,
        a_shape,
        b_shape=None,
        transpose_b=False,
        **kwargs,
    ):
        """Submit a GEMM from raw MXFP8 buffers and logical matrix dimensions."""
        rows, cols = a_shape
        layouts = {
            "a_tensor": _matrix_layout(rows, cols, torch.float8_e4m3fn),
            "sfa_tensor": _scale_layout(1, rows, cols),
        }
        if kwargs.get("b_tensor") is not None:
            groups, weight_rows, weight_cols = b_shape
            layouts["b_tensor"] = _weight_layout(groups, weight_rows, weight_cols, transpose_b)
            n, k = (weight_cols, weight_rows) if transpose_b else (weight_rows, weight_cols)
            layouts["sfb_tensor"] = _scale_layout(groups, n, k)
        bias = kwargs.get("bias_tensor")
        if bias is not None:
            groups, weight_rows, weight_cols = b_shape
            n = weight_cols if transpose_b else weight_rows
            layouts["bias_tensor"] = ((n, groups), (1, n), bias.dtype)
        for name in ("prob_tensor", "dprob_tensor"):
            tensor = kwargs.get(name)
            if tensor is not None:
                layouts[name] = ((rows, 1, 1), (1, rows, rows), tensor.dtype)
        for name in ("c_tensor", "d_tensor"):
            tensor = kwargs.get(name)
            if tensor is not None:
                layouts[name] = _matrix_layout(rows, tensor.shape[-1], tensor.dtype)
        if self.use_native_layouts:
            return kernel(**kwargs, tensor_layouts=layouts)
        for name, layout in layouts.items():
            kwargs[name] = _legacy_operand(kwargs[name], layout)
        return _physical_outputs(kernel(**kwargs))

    @staticmethod
    def bias(linear_op, *, mxfp8):
        """Collect bias storage; only the legacy non-MXFP8 path needs a view."""
        if not linear_op.has_bias:
            return None
        grouped = getattr(linear_op, "bias", None)
        if grouped is not None:
            buffer = grouped.rowwise_data
        else:
            buffer = torch.stack(
                [getattr(linear_op, f"bias{index}") for index in range(linear_op.num_groups)]
            )
        if mxfp8:
            return buffer
        return buffer.view(linear_op.num_groups, -1).transpose(0, 1)

    def wgrad(
        self,
        *,
        tokens,
        weight_shape,
        groups,
        **kwargs,
    ):
        """Submit WGRAD without materializing transposed or typed Torch views."""
        rows, cols = weight_shape
        sf_rows = ((rows + 127) // 128) * 128
        sf_cols = ((cols + 127) // 128) * 128
        sfa_pitch = kwargs["sfa_tensor"].numel() // sf_rows
        sfb_pitch = kwargs["sfb_tensor"].numel() // sf_cols
        layouts = {
            "a_tensor": ((rows, tokens), (1, rows), torch.float8_e4m3fn),
            "b_tensor": ((tokens, cols), (cols, 1), torch.float8_e4m3fn),
            "sfa_tensor": ((sf_rows, sfa_pitch), (sfa_pitch, 1), torch.float8_e8m0fnu),
            "sfb_tensor": ((sf_cols, sfb_pitch), (sfb_pitch, 1), torch.float8_e8m0fnu),
        }
        output = kwargs.get("wgrad_tensor")
        if output is not None:
            layouts["wgrad_tensor"] = (
                (groups, rows, cols),
                (rows * cols, cols, 1),
                output.dtype,
            )
        if self.use_native_wgrad_layouts:
            return self.wgrad_kernel(**kwargs, tensor_layouts=layouts)
        for name, layout in layouts.items():
            kwargs[name] = _legacy_operand(kwargs[name], layout)
        return self.wgrad_kernel(**kwargs)

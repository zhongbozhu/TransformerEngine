# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops


_HIDDEN_SIZE = 512
_FFN_HIDDEN_SIZE = 256


def _megacpp_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if not te.is_bf16_available():
        return False, "BF16 is required"
    if torch.cuda.get_device_capability() < (10, 0):
        return False, "megacpp grouped MLP uses SM100 grouped GEMM"
    if not te_ops.fused.ForwardGroupedMLP_MegaCpp.is_supported():
        return False, "ForwardGroupedMLP_MegaCpp is not supported"
    if not te_ops.fused.BackwardGroupedMLP_MegaCpp.is_supported():
        return False, "BackwardGroupedMLP_MegaCpp is not supported"
    return True, ""


_AVAILABLE, _SKIP_REASON = _megacpp_available()
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=_SKIP_REASON)


def _make_grouped_mlp(
    *,
    num_groups: int,
    hidden_size: int,
    ffn_hidden_size: int,
    bias: bool,
    delay_wgrad_compute: bool,
    accumulate_into_main_grad: bool,
    glu_interleave_size: int | None,
    single_grouped_param: bool,
) -> te_ops.Sequential:
    fc1 = te_ops.GroupedLinear(
        num_groups,
        hidden_size,
        2 * ffn_hidden_size,
        bias=bias,
        device="cuda",
        dtype=torch.bfloat16,
        delay_wgrad_compute=delay_wgrad_compute,
        accumulate_into_main_grad=accumulate_into_main_grad,
        single_grouped_weight=single_grouped_param,
        single_grouped_bias=single_grouped_param and bias,
    )
    act = te_ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
    fc2 = te_ops.GroupedLinear(
        num_groups,
        ffn_hidden_size,
        hidden_size,
        bias=bias,
        device="cuda",
        dtype=torch.bfloat16,
        delay_wgrad_compute=delay_wgrad_compute,
        accumulate_into_main_grad=accumulate_into_main_grad,
        single_grouped_weight=single_grouped_param,
        single_grouped_bias=single_grouped_param and bias,
    )
    return te_ops.Sequential(fc1, act, fc2)


def _copy_grouped_mlp_params(dst: te_ops.Sequential, src: te_ops.Sequential) -> None:
    with torch.no_grad():
        for dst_linear, src_linear in ((dst[0], src[0]), (dst[2], src[2])):
            if dst_linear.single_grouped_weight:
                dst_linear.weight.rowwise_data.copy_(src_linear.weight.rowwise_data)
                if dst_linear.has_bias:
                    dst_linear.bias.rowwise_data.copy_(src_linear.bias.rowwise_data)
                continue
            for group_idx in range(dst_linear.num_groups):
                getattr(dst_linear, f"weight{group_idx}").copy_(
                    getattr(src_linear, f"weight{group_idx}")
                )
                if dst_linear.has_bias:
                    getattr(dst_linear, f"bias{group_idx}").copy_(
                        getattr(src_linear, f"bias{group_idx}")
                    )


def _init_main_grads(module: te_ops.Sequential) -> None:
    for linear in (module[0], module[2]):
        if linear.single_grouped_weight:
            linear.weight.main_grad = torch.zeros(
                linear.num_groups,
                linear.out_features,
                linear.in_features,
                device="cuda",
                dtype=torch.bfloat16,
            )
            continue
        for group_idx in range(linear.num_groups):
            weight = getattr(linear, f"weight{group_idx}")
            weight.main_grad = torch.zeros_like(weight)


def _run_grouped_mlp(
    module: te_ops.Sequential,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    scales: torch.Tensor,
    dy: torch.Tensor,
    *,
    delay_wgrad_compute: bool,
) -> torch.Tensor:
    y = module(x, split_sizes, scales, split_sizes)
    y.backward(dy)
    if delay_wgrad_compute:
        module[0].backward_dw()
        module[2].backward_dw()
    return y


def _assert_grouped_mlp_close(
    test: te_ops.Sequential,
    ref: te_ops.Sequential,
    *,
    accumulate_into_main_grad: bool,
) -> None:
    for test_linear, ref_linear in ((test[0], ref[0]), (test[2], ref[2])):
        if test_linear.single_grouped_weight:
            if accumulate_into_main_grad:
                torch.testing.assert_close(
                    test_linear.weight.main_grad,
                    ref_linear.weight.main_grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            else:
                torch.testing.assert_close(
                    test_linear.weight.grad,
                    ref_linear.weight.grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            if test_linear.has_bias:
                torch.testing.assert_close(
                    test_linear.bias.grad,
                    ref_linear.bias.grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            continue
        for group_idx in range(test_linear.num_groups):
            if accumulate_into_main_grad:
                torch.testing.assert_close(
                    getattr(test_linear, f"weight{group_idx}").main_grad,
                    getattr(ref_linear, f"weight{group_idx}").main_grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            else:
                torch.testing.assert_close(
                    getattr(test_linear, f"weight{group_idx}").grad,
                    getattr(ref_linear, f"weight{group_idx}").grad,
                    rtol=2e-2,
                    atol=2e-2,
                )
            if test_linear.has_bias:
                torch.testing.assert_close(
                    getattr(test_linear, f"bias{group_idx}").grad,
                    getattr(ref_linear, f"bias{group_idx}").grad,
                    rtol=2e-2,
                    atol=2e-2,
                )


def _assert_grouped_mlp_nonzero_expert_grads_close(
    test: te_ops.Sequential,
    ref: te_ops.Sequential,
    split_sizes: list[int],
) -> None:
    """Compare only non-empty experts; zero-token expert grads may be unwritten."""
    for test_linear, ref_linear in ((test[0], ref[0]), (test[2], ref[2])):
        for group_idx, split_size in enumerate(split_sizes):
            if split_size == 0:
                continue
            torch.testing.assert_close(
                getattr(test_linear, f"weight{group_idx}").grad,
                getattr(ref_linear, f"weight{group_idx}").grad,
                rtol=2e-2,
                atol=2e-2,
            )
            if test_linear.has_bias:
                torch.testing.assert_close(
                    getattr(test_linear, f"bias{group_idx}").grad,
                    getattr(ref_linear, f"bias{group_idx}").grad,
                    rtol=2e-2,
                    atol=2e-2,
                )


def _assert_valid_prefix_close(
    test: torch.Tensor,
    ref: torch.Tensor,
    valid_tokens: int,
) -> None:
    """Paged-stashed buffers only guarantee correctness in the valid token prefix."""
    if valid_tokens == 0:
        return
    torch.testing.assert_close(test[:valid_tokens], ref[:valid_tokens], rtol=2e-2, atol=2e-2)


def _make_split_tensor(
    split_sizes: list[int],
    *,
    dtype: torch.dtype = torch.int64,
    device: str = "cuda",
) -> torch.Tensor:
    return torch.tensor(split_sizes, dtype=dtype, device=device)


def _run_megacpp_against_python(
    *,
    split_sizes_list: list[int],
    physical_tokens: int,
    split_dtype: torch.dtype,
    split_device: str,
    bias: bool = True,
    glu_interleave_size: int | None = None,
    single_grouped_param: bool = False,
    accumulate_into_main_grad: bool = False,
    compare_zero_expert_grads: bool = True,
    monkeypatch,
) -> None:
    num_groups = len(split_sizes_list)
    valid_tokens = sum(split_sizes_list)
    assert physical_tokens >= valid_tokens
    split_sizes = _make_split_tensor(split_sizes_list, dtype=split_dtype, device=split_device)
    ref = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        bias=bias,
        delay_wgrad_compute=False,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        single_grouped_param=single_grouped_param,
    )
    test = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        bias=bias,
        delay_wgrad_compute=False,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        single_grouped_param=single_grouped_param,
    )
    _copy_grouped_mlp_params(test, ref)
    if accumulate_into_main_grad:
        _init_main_grads(ref)
        _init_main_grads(test)

    # Paged stashing passes a static physical buffer to the op while m_splits
    # describe only the valid prefix. Rows after sum(m_splits) are garbage and
    # must not affect outputs/gradients for the valid prefix.
    x_ref = torch.randn(
        physical_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    x_test = x_ref.detach().clone().requires_grad_()
    scales_ref = torch.rand(physical_tokens, device="cuda", dtype=torch.bfloat16).requires_grad_()
    scales_test = scales_ref.detach().clone().requires_grad_()
    dy = torch.randn(physical_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "0")
    y_ref = _run_grouped_mlp(
        ref,
        x_ref,
        split_sizes,
        scales_ref,
        dy,
        delay_wgrad_compute=False,
    )
    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "1")
    y_test = _run_grouped_mlp(
        test,
        x_test,
        split_sizes,
        scales_test,
        dy,
        delay_wgrad_compute=False,
    )

    fuser = test._module_groups[0]
    assert isinstance(fuser._forward_ops[0][0], te_ops.fused.ForwardGroupedMLP_MegaCpp)
    assert isinstance(fuser._backward_ops[0][0], te_ops.fused.BackwardGroupedMLP_MegaCpp)

    _assert_valid_prefix_close(y_test, y_ref, valid_tokens)
    _assert_valid_prefix_close(x_test.grad, x_ref.grad, valid_tokens)
    _assert_valid_prefix_close(scales_test.grad, scales_ref.grad, valid_tokens)
    if valid_tokens == physical_tokens and compare_zero_expert_grads:
        _assert_grouped_mlp_close(test, ref, accumulate_into_main_grad=accumulate_into_main_grad)
    elif valid_tokens > 0 and not single_grouped_param and not accumulate_into_main_grad:
        _assert_grouped_mlp_nonzero_expert_grads_close(test, ref, split_sizes_list)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            False, True, torch.int64, "cuda", None, False, False, id="bias_no_delay_i64_cuda"
        ),
        pytest.param(
            False,
            True,
            torch.int64,
            "cuda",
            None,
            False,
            True,
            id="bias_no_delay_accumulate_main_grad",
        ),
        pytest.param(
            False,
            True,
            torch.int64,
            "cuda",
            None,
            True,
            False,
            id="bias_no_delay_single_grouped_param",
        ),
        pytest.param(
            False,
            True,
            torch.int32,
            "cuda",
            32,
            False,
            False,
            id="bias_no_delay_i32_cuda_interleave",
        ),
        pytest.param(
            False,
            False,
            torch.int64,
            "cpu",
            32,
            False,
            False,
            id="no_bias_no_delay_i64_cpu_interleave",
        ),
    ],
)
def test_megacpp_grouped_mlp_bf16_matches_python(case, monkeypatch):
    (
        delay_wgrad_compute,
        bias,
        split_dtype,
        split_device,
        glu_interleave_size,
        single_grouped_param,
        accumulate_into_main_grad,
    ) = case
    if single_grouped_param:
        monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    torch.manual_seed(1234)
    num_groups = 3
    split_sizes_cuda = torch.tensor([256, 256, 512], dtype=torch.int64, device="cuda")
    split_sizes = split_sizes_cuda.to(device=split_device, dtype=split_dtype)
    total_tokens = int(split_sizes_cuda.sum().item())

    ref = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        bias=bias,
        delay_wgrad_compute=delay_wgrad_compute,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        single_grouped_param=single_grouped_param,
    )
    test = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        bias=bias,
        delay_wgrad_compute=delay_wgrad_compute,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        single_grouped_param=single_grouped_param,
    )
    _copy_grouped_mlp_params(test, ref)
    if accumulate_into_main_grad:
        _init_main_grads(ref)
        _init_main_grads(test)

    x_ref = torch.randn(
        total_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    x_test = x_ref.detach().clone().requires_grad_()
    scales_ref = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16).requires_grad_()
    scales_test = scales_ref.detach().clone().requires_grad_()
    dy = torch.randn(total_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "0")
    y_ref = _run_grouped_mlp(
        ref,
        x_ref,
        split_sizes,
        scales_ref,
        dy,
        delay_wgrad_compute=delay_wgrad_compute,
    )
    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "1")
    y_test = _run_grouped_mlp(
        test,
        x_test,
        split_sizes,
        scales_test,
        dy,
        delay_wgrad_compute=delay_wgrad_compute,
    )

    fuser = test._module_groups[0]
    assert isinstance(fuser._forward_ops[0][0], te_ops.fused.ForwardGroupedMLP_MegaCpp)
    assert isinstance(fuser._backward_ops[0][0], te_ops.fused.BackwardGroupedMLP_MegaCpp)

    torch.testing.assert_close(y_test, y_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_test.grad, x_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(scales_test.grad, scales_ref.grad, rtol=2e-2, atol=2e-2)
    _assert_grouped_mlp_close(test, ref, accumulate_into_main_grad=accumulate_into_main_grad)


@pytest.mark.parametrize(
    "split_sizes_list,physical_tokens",
    [
        pytest.param([256, 256, 256, 256], 1024, id="even"),
        pytest.param([0, 256, 256, 512], 1024, id="zero_front"),
        pytest.param([256, 0, 256, 512], 1024, id="zero_middle"),
        pytest.param([256, 256, 512, 0], 1024, id="zero_end"),
        pytest.param([256, 256], 1024, id="paged_stashing_even_with_garbage"),
        pytest.param([0, 256, 256], 1024, id="paged_stashing_zero_front_with_garbage"),
        pytest.param([256, 0, 256], 1024, id="paged_stashing_zero_middle_with_garbage"),
        pytest.param([256, 256, 0], 1024, id="paged_stashing_zero_end_with_garbage"),
        pytest.param([0, 0, 0, 0], 1024, id="paged_stashing_zero_tokens_all_nonempty_input"),
    ],
)
def test_megacpp_grouped_mlp_split_edge_cases(
    split_sizes_list,
    physical_tokens,
    monkeypatch,
):
    torch.manual_seed(1234)
    _run_megacpp_against_python(
        split_sizes_list=split_sizes_list,
        physical_tokens=physical_tokens,
        split_dtype=torch.int64,
        split_device="cuda",
        compare_zero_expert_grads=False,
        monkeypatch=monkeypatch,
    )


def test_megacpp_grouped_mlp_delay_wgrad_raises(monkeypatch):
    torch.manual_seed(1234)
    num_groups = 3
    split_sizes = torch.tensor([256, 256, 512], dtype=torch.int64, device="cuda")
    total_tokens = int(split_sizes.sum().item())
    module = _make_grouped_mlp(
        num_groups=num_groups,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        bias=True,
        delay_wgrad_compute=True,
        accumulate_into_main_grad=False,
        glu_interleave_size=None,
        single_grouped_param=False,
    )
    x = torch.randn(total_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16).requires_grad_()
    scales = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn(total_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("NVTE_MEGACPP_GROUPED_LINEAR", "1")
    with pytest.raises(ValueError, match="delay_wgrad_compute"):
        _run_grouped_mlp(
            module,
            x,
            split_sizes,
            scales,
            dy,
            delay_wgrad_compute=False,
        )

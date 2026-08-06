# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped-operand validation for custom quantization recipes."""

from ..tensor import Float8Quantizer, HybridQuantizer, IdentityQuantizer


_DYNAMIC_QUANTIZER_FIELDS = frozenset(
    {
        "rowwise_usage",
        "columnwise_usage",
        "internal",
        "optimize_for_gemm",
    }
)


def _uses_identity_quantizer(quantizer):
    """Whether a quantizer, including a hybrid sub-quantizer, is Identity-backed."""
    if quantizer is None:
        return False
    if isinstance(quantizer, IdentityQuantizer):
        return True
    if isinstance(quantizer, HybridQuantizer):
        return _uses_identity_quantizer(quantizer.rowwise_quantizer) or _uses_identity_quantizer(
            quantizer.columnwise_quantizer
        )
    return False


def _identity_quantizer_signature(quantizer):
    """Identity usage per GEMM direction: (rowwise, columnwise)."""
    if isinstance(quantizer, HybridQuantizer):
        return (
            _uses_identity_quantizer(quantizer.rowwise_quantizer),
            _uses_identity_quantizer(quantizer.columnwise_quantizer),
        )
    identity = isinstance(quantizer, IdentityQuantizer)
    return (identity, identity)


def _backend_quantizer_signature(quantizer):
    """Return backend configuration that grouped kernels require to be uniform."""
    if quantizer is None:
        return None

    # Identity is not registered as a torch.compile value quantizer, but its dtype changes
    # the grouped GEMM input type and therefore must be uniform across grouped operands.
    if isinstance(quantizer, IdentityQuantizer):
        return (type(quantizer), (("dtype", quantizer.dtype),))

    fields = quantizer._value_fields()
    if fields is None:
        # Delayed-scaling quantizers carry different scale/amax tensors per expert. Only their
        # emitted FP8 dtype is a group-wide backend choice.
        fields = ("dtype",) if isinstance(quantizer, Float8Quantizer) else ()

    config = []
    for name in fields:
        if name in _DYNAMIC_QUANTIZER_FIELDS:
            continue
        value = getattr(quantizer, name)
        if name == "dtype":
            value = int(value)
        config.append((name, value))
    return (type(quantizer), tuple(config))


def _validate_backend_match(reference, quantizer, operand_name, direction, expert_index):
    """Validate one expert against the group's reference backend."""
    if type(quantizer) is not type(reference):
        raise ValueError(
            f"GroupedLinear {operand_name} quantizers use incompatible {direction} backend"
            f" families across experts: expert 0 uses {type(reference).__name__}, but expert"
            f" {expert_index} uses {type(quantizer).__name__}. Grouped operands require one"
            " quantizer family per direction."
        )
    reference_signature = _backend_quantizer_signature(reference)
    quantizer_signature = _backend_quantizer_signature(quantizer)
    if quantizer_signature != reference_signature:
        raise ValueError(
            f"GroupedLinear {operand_name} quantizers use incompatible {direction} backend"
            f" configurations across experts: expert 0 uses {reference_signature}, but expert"
            f" {expert_index} uses {quantizer_signature}. Grouped operands require the same"
            " backend-relevant configuration per direction."
        )


def validate_grouped_quantizer_list(quantizers, *, operand_name="operand") -> None:
    """Validate that a custom recipe produced one compatible grouped operand."""
    if not quantizers:
        return

    reference = quantizers[0]
    reference_is_hybrid = isinstance(reference, HybridQuantizer)
    reference_identity = _identity_quantizer_signature(reference)

    for expert_index, quantizer in enumerate(quantizers[1:], start=1):
        if (quantizer is None) != (reference is None):
            raise ValueError(
                f"GroupedLinear {operand_name} quantizers mix None and concrete quantizers"
                f" across experts: expert 0 is {type(reference).__name__}, but expert"
                f" {expert_index} is {type(quantizer).__name__}."
            )
        if reference is None:
            continue

        quantizer_is_hybrid = isinstance(quantizer, HybridQuantizer)
        if quantizer_is_hybrid != reference_is_hybrid:
            raise ValueError(
                f"GroupedLinear {operand_name} quantizers mix HybridQuantizer and non-hybrid"
                f" quantizers across experts: expert 0 is {type(reference).__name__}, but expert"
                f" {expert_index} is {type(quantizer).__name__}."
            )

        identity = _identity_quantizer_signature(quantizer)
        if identity != reference_identity:
            raise ValueError(
                f"GroupedLinear {operand_name} quantizers mix Identity-backed and quantized"
                f" directions across experts: expert 0 uses {reference_identity}, but expert"
                f" {expert_index} uses {identity}."
            )

        if reference_is_hybrid:
            _validate_backend_match(
                reference.rowwise_quantizer,
                quantizer.rowwise_quantizer,
                operand_name,
                "rowwise",
                expert_index,
            )
            _validate_backend_match(
                reference.columnwise_quantizer,
                quantizer.columnwise_quantizer,
                operand_name,
                "columnwise",
                expert_index,
            )
            if quantizer.columnwise_source != reference.columnwise_source:
                raise ValueError(
                    f"GroupedLinear {operand_name} HybridQuantizer list has mixed columnwise"
                    " source policies across experts: expert 0 uses"
                    f" {reference.columnwise_source!r}, but expert {expert_index} uses"
                    f" {quantizer.columnwise_source!r}."
                )
        else:
            _validate_backend_match(
                reference,
                quantizer,
                operand_name,
                "plain",
                expert_index,
            )


__all__ = ["validate_grouped_quantizer_list"]

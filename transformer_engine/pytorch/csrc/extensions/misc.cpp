/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

namespace {

std::vector<at::Tensor> prepare_grouped_splits_impl(const at::Tensor &split_sizes,
                                                    int64_t num_groups,
                                                    const std::vector<int64_t> &logical_last_dims) {
  NVTE_CHECK(split_sizes.scalar_type() == at::kInt || split_sizes.scalar_type() == at::kLong,
             "split_sizes must have dtype int32 or int64.");
  NVTE_CHECK(split_sizes.dim() == 1, "split_sizes must be a 1D tensor.");
  NVTE_CHECK(num_groups > 0, "num_groups must be greater than 0.");
  NVTE_CHECK(split_sizes.numel() == num_groups, "split_sizes must have length ", num_groups, ".");
  NVTE_CHECK(!logical_last_dims.empty(), "logical_last_dims must be non-empty.");
  NVTE_CHECK(logical_last_dims.size() <= NVTE_MAX_GROUPED_SPLIT_TENSOR_OFFSETS,
             "logical_last_dims supports at most ", NVTE_MAX_GROUPED_SPLIT_TENSOR_OFFSETS,
             " entries.");
  for (const auto logical_last_dim : logical_last_dims) {
    NVTE_CHECK(logical_last_dim >= 0, "logical_last_dim values must be non-negative.");
  }
  const c10::Device device = c10::Device(c10::kCUDA, c10::cuda::current_device());

  at::Tensor split_sizes_for_kernel;
  if (split_sizes.is_cuda()) {
    NVTE_CHECK(split_sizes.device() == device, "CUDA split_sizes must be on current CUDA device ",
               device.index(), ", but got CUDA device ", split_sizes.device().index(), ".");
    split_sizes_for_kernel = split_sizes;
  } else {
    // Preserve the legacy eager path: host m_splits are copied to the target
    // CUDA device here, then all derived metadata is produced by one CUDA kernel.
    split_sizes_for_kernel =
        split_sizes.to(at::TensorOptions().dtype(split_sizes.scalar_type()).device(device),
                       /*non_blocking=*/true);
  }

  const int64_t offsets_length = num_groups + 1;

  // Return order is part of the Python contract:
  //   0. split_sizes_i64: int64[num_groups], canonical TE GroupedTensor first dims.
  //   1. base_offsets: int64[num_groups + 1], [0, cumsum(split_sizes)].
  //   2. split_points: int32[num_groups], cumsum(split_sizes) without the leading 0
  //      for cuDNN grouped GEMM padded offsets.  This is intentionally int32
  //      even though TE grouped tensor metadata uses int64 below.
  //   3..: tensor_offsets[i]: int64[num_groups + 1],
  //      base_offsets * logical_last_dims[i].
  //
  // Force 16-byte alignment on every output so ``split_points`` (consumed by
  // cuDNN CuTe-DSL grouped GEMM as ``padded_offsets``, which requires 16-byte
  // alignment) lands on a 16-byte boundary inside the bulk buffer.
  std::vector<std::vector<size_t>> shapes = {{static_cast<size_t>(num_groups)},
                                             {static_cast<size_t>(offsets_length)},
                                             {static_cast<size_t>(num_groups)}};
  std::vector<at::ScalarType> dtypes = {at::kLong, at::kLong, at::kInt};
  std::vector<size_t> alignments = {16, 16, 16};
  shapes.reserve(3 + logical_last_dims.size());
  dtypes.reserve(3 + logical_last_dims.size());
  alignments.reserve(3 + logical_last_dims.size());
  for (size_t i = 0; i < logical_last_dims.size(); ++i) {
    shapes.emplace_back(std::vector<size_t>{static_cast<size_t>(offsets_length)});
    dtypes.emplace_back(at::kLong);
    alignments.emplace_back(16);
  }
  auto outputs = bulk_allocate(shapes, dtypes, device, alignments);
  auto split_sizes_i64 = outputs[0];
  auto base_offsets = outputs[1];
  auto split_points = outputs[2];

  auto split_sizes_nvte = makeTransformerEngineTensor(split_sizes_for_kernel);
  auto split_sizes_i64_nvte = makeTransformerEngineTensor(split_sizes_i64);
  auto base_offsets_nvte = makeTransformerEngineTensor(base_offsets);
  auto split_points_nvte = makeTransformerEngineTensor(split_points);
  std::vector<TensorWrapper> tensor_offsets_nvte;
  tensor_offsets_nvte.reserve(logical_last_dims.size());

  NVTEGroupedSplitMetadata metadata = {};
  metadata.first_dims = split_sizes_nvte.data();
  metadata.first_dims_i64 = split_sizes_i64_nvte.data();
  metadata.base_offsets = base_offsets_nvte.data();
  metadata.split_points = split_points_nvte.data();
  metadata.num_tensor_offsets = logical_last_dims.size();
  for (size_t i = 0; i < logical_last_dims.size(); ++i) {
    tensor_offsets_nvte.emplace_back(makeTransformerEngineTensor(outputs[3 + i]));
    metadata.tensor_offsets[i] = tensor_offsets_nvte.back().data();
    metadata.logical_last_dims[i] = logical_last_dims[i];
  }

  NVTE_SCOPED_GIL_RELEASE({
    nvte_prepare_grouped_splits(&metadata, at::cuda::getCurrentCUDAStream());
  });

  return outputs;
}

}  // namespace

at::Tensor splits_to_offsets(const at::Tensor &first_dims, int64_t logical_last_dim) {
  NVTE_CHECK(first_dims.is_cuda(), "first_dims must be on CUDA.");
  NVTE_CHECK(first_dims.scalar_type() == at::kLong, "first_dims must have dtype int64.");
  NVTE_CHECK(first_dims.dim() == 1, "first_dims must be a 1D tensor.");
  NVTE_CHECK(logical_last_dim > 0, "logical_last_dim must be greater than 0.");

  auto first_dims_contiguous = first_dims.contiguous();
  const auto num_tensors = static_cast<size_t>(first_dims_contiguous.numel());
  auto output = at::empty({static_cast<int64_t>(num_tensors) + 1},
                          first_dims_contiguous.options().dtype(at::kLong));

  nvte_splits_to_offsets(static_cast<const int64_t *>(first_dims_contiguous.data_ptr()),
                         static_cast<int64_t *>(output.data_ptr()), num_tensors, logical_last_dim,
                         at::cuda::getCurrentCUDAStream());

  return output;
}

std::vector<at::Tensor> prepare_grouped_splits(const at::Tensor &split_sizes, int64_t num_groups,
                                               int64_t logical_last_dim) {
  return prepare_grouped_splits_impl(split_sizes, num_groups, {logical_last_dim});
}

std::vector<at::Tensor> prepare_grouped_splits(const at::Tensor &split_sizes, int64_t num_groups,
                                               const std::vector<int64_t> &logical_last_dims) {
  return prepare_grouped_splits_impl(split_sizes, num_groups, logical_last_dims);
}

}  // namespace transformer_engine::pytorch

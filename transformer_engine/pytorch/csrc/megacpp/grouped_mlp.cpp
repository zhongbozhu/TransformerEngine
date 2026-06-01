/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../extensions.h"
#include "../pybind.h"
#include "transformer_engine/activation.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/transformer_engine.h"

namespace py = pybind11;

namespace transformer_engine::pytorch {
namespace {

constexpr int64_t kGroupedGemmCublasWorkspaceSize = 32 * 1024 * 1024 + 1024;

bool is_none(py::handle obj) { return obj.is_none(); }

std::vector<size_t> tensor_shape_1d(const at::Tensor &tensor) {
  return {static_cast<size_t>(tensor.numel())};
}

at::Tensor as_compute_tensor(const at::Tensor &tensor, at::ScalarType dtype) {
  at::Tensor out = tensor;
  if (out.scalar_type() != dtype) {
    out = out.to(out.options().dtype(dtype));
  }
  if (!out.is_contiguous()) {
    out = out.contiguous();
  }
  return out;
}

at::Tensor as_cuda_i64_splits(const at::Tensor &split_sizes, const c10::Device &device) {
  NVTE_CHECK(split_sizes.dim() == 1, "split_sizes must be a 1D tensor.");
  if (split_sizes.device() == device && split_sizes.scalar_type() == at::kLong) {
    return split_sizes;
  }
  return split_sizes.to(split_sizes.options().device(device).dtype(at::kLong));
}

GroupedTensorWrapper make_grouped_tensor(at::Tensor data, const at::Tensor &split_sizes_i64,
                                         const at::Tensor &tensor_offsets, int64_t logical_last_dim) {
  const auto num_groups = static_cast<size_t>(split_sizes_i64.numel());
  const auto total_tokens = static_cast<size_t>(data.numel() / logical_last_dim);
  auto grouped = GroupedTensorWrapper(
      num_groups, std::vector<size_t>{total_tokens, static_cast<size_t>(logical_last_dim)});
  grouped.set_rowwise_data(data.data_ptr(), GetTransformerEngineDType(data.scalar_type()),
                           tensor_shape_1d(data));
  grouped.set_first_dims(split_sizes_i64.data_ptr(), DType::kInt64,
                         std::vector<size_t>{num_groups});
  grouped.set_tensor_offsets(tensor_offsets.data_ptr(), DType::kInt64,
                             std::vector<size_t>{num_groups + 1});
  return grouped;
}

std::vector<at::Tensor> tensor_list_from_arg(py::handle arg, size_t num_groups,
                                             at::ScalarType dtype,
                                             const std::string &name) {
  std::vector<at::Tensor> out;
  out.reserve(num_groups);
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto seq = py::reinterpret_borrow<py::sequence>(arg);
    NVTE_CHECK(static_cast<size_t>(seq.size()) == num_groups, name, " must have ", num_groups,
               " tensors.");
    for (size_t i = 0; i < num_groups; ++i) {
      out.emplace_back(as_compute_tensor(seq[i].cast<at::Tensor>(), dtype));
    }
    return out;
  }

  auto packed = as_compute_tensor(arg.cast<at::Tensor>(), dtype);
  NVTE_CHECK(packed.dim() == 3, name, " must be a tensor with shape [num_groups, rows, cols].");
  NVTE_CHECK(static_cast<size_t>(packed.size(0)) == num_groups, name, " first dimension must be ",
             num_groups, ".");
  for (size_t i = 0; i < num_groups; ++i) {
    out.emplace_back(packed.select(0, static_cast<int64_t>(i)).contiguous());
  }
  return out;
}

std::vector<at::Tensor> bias_list_from_arg(py::handle arg, size_t num_groups, at::ScalarType dtype,
                                           const std::string &name) {
  std::vector<at::Tensor> out;
  if (is_none(arg)) {
    return out;
  }
  out.reserve(num_groups);
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto seq = py::reinterpret_borrow<py::sequence>(arg);
    if (seq.size() == 0) {
      return out;
    }
    NVTE_CHECK(static_cast<size_t>(seq.size()) == num_groups, name, " must have ", num_groups,
               " tensors.");
    for (size_t i = 0; i < num_groups; ++i) {
      out.emplace_back(as_compute_tensor(seq[i].cast<at::Tensor>(), dtype));
    }
    return out;
  }

  auto packed = as_compute_tensor(arg.cast<at::Tensor>(), dtype);
  NVTE_CHECK(packed.dim() == 2, name, " must be a tensor with shape [num_groups, features].");
  NVTE_CHECK(static_cast<size_t>(packed.size(0)) == num_groups, name, " first dimension must be ",
             num_groups, ".");
  for (size_t i = 0; i < num_groups; ++i) {
    out.emplace_back(packed.select(0, static_cast<int64_t>(i)).contiguous());
  }
  return out;
}

std::vector<NVTETensor> nvte_tensor_list_from_tensors(const std::vector<at::Tensor> &tensors,
                                                      std::vector<TensorWrapper> *wrappers) {
  wrappers->clear();
  wrappers->reserve(tensors.size());
  std::vector<NVTETensor> out;
  out.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    wrappers->emplace_back(makeTransformerEngineTensor(tensor));
    out.emplace_back(wrappers->back().data());
  }
  return out;
}

struct GroupedGemmWorkspace {
  at::Tensor alpha;
  at::Tensor beta;
  at::Tensor setup;
  at::Tensor cublas;
  TensorWrapper te_alpha;
  TensorWrapper te_beta;
  TensorWrapper te_setup;
  TensorWrapper te_cublas;
  std::optional<GroupedMatmulConfigWrapper> config;

  GroupedGemmWorkspace(const c10::Device &device, size_t num_groups, bool use_split_accumulator,
                       bool accumulate = false)
      : alpha(at::ones({static_cast<int64_t>(num_groups)}, at::device(device).dtype(at::kFloat))),
        beta(accumulate ? at::ones({static_cast<int64_t>(num_groups)},
                                   at::device(device).dtype(at::kFloat))
                        : at::zeros({static_cast<int64_t>(num_groups)},
                                    at::device(device).dtype(at::kFloat))),
        setup(at::empty(
            {static_cast<int64_t>(nvte_get_grouped_gemm_setup_workspace_size(num_groups))},
            at::device(device).dtype(at::kByte))),
        cublas(at::empty({kGroupedGemmCublasWorkspaceSize}, at::device(device).dtype(at::kByte))),
        te_alpha(makeTransformerEngineTensor(alpha)),
        te_beta(makeTransformerEngineTensor(beta)),
        te_setup(makeTransformerEngineTensor(setup.data_ptr(),
                                             std::vector<size_t>{static_cast<size_t>(setup.numel())},
                                             DType::kByte)),
        te_cublas(makeTransformerEngineTensor(
            cublas.data_ptr(), std::vector<size_t>{static_cast<size_t>(cublas.numel())},
            DType::kByte)) {
    if (use_split_accumulator) {
      config.emplace();
      config->set_use_split_accumulator(true);
    }
  }
};

void grouped_gemm_discrete_weight(const std::vector<at::Tensor> &weights, bool trans_weight,
                                  GroupedTensorWrapper *input, bool trans_input,
                                  GroupedTensorWrapper *output, bool use_split_accumulator) {
  std::vector<TensorWrapper> weight_wrappers;
  auto weight_nvte = nvte_tensor_list_from_tensors(weights, &weight_wrappers);
  GroupedGemmWorkspace workspace(weights[0].device(), weights.size(), use_split_accumulator);
  NVTEGroupedMatmulConfig config =
      workspace.config.has_value() ? static_cast<NVTEGroupedMatmulConfig>(*workspace.config)
                                   : nullptr;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_gemm_with_discrete_inputA(
        weight_nvte.data(), weights.size(), trans_weight, input->data(), trans_input,
        output->data(), output->data(), workspace.te_alpha.data(), workspace.te_beta.data(),
        workspace.te_setup.data(), workspace.te_cublas.data(), config,
        at::cuda::getCurrentCUDAStream());
  });
}

std::vector<at::Tensor> output_tensor_list_from_arg(py::handle arg, size_t num_groups,
                                                    at::ScalarType dtype,
                                                    const std::string &name) {
  std::vector<at::Tensor> out;
  if (is_none(arg)) {
    return out;
  }
  out.reserve(num_groups);
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto seq = py::reinterpret_borrow<py::sequence>(arg);
    NVTE_CHECK(static_cast<size_t>(seq.size()) == num_groups, name, " must have ", num_groups,
               " tensors.");
    for (size_t i = 0; i < num_groups; ++i) {
      auto tensor = seq[i].cast<at::Tensor>();
      NVTE_CHECK(tensor.is_cuda(), name, " tensors must be CUDA tensors.");
      NVTE_CHECK(tensor.scalar_type() == dtype, name, " tensors must have the requested dtype.");
      NVTE_CHECK(tensor.dim() == 2, name, " tensors must be rank-2 wgrad buffers.");
      NVTE_CHECK(tensor.is_contiguous(), name, " tensors must be contiguous.");
      out.emplace_back(tensor);
    }
    return out;
  }

  auto packed = arg.cast<at::Tensor>();
  NVTE_CHECK(packed.is_cuda(), name, " must be a CUDA tensor.");
  NVTE_CHECK(packed.scalar_type() == dtype, name, " must have the requested dtype.");
  NVTE_CHECK(packed.dim() == 3, name, " must have shape [num_groups, rows, cols].");
  NVTE_CHECK(static_cast<size_t>(packed.size(0)) == num_groups, name, " first dimension must be ",
             num_groups, ".");
  NVTE_CHECK(packed.is_contiguous(), name, " must be contiguous.");
  for (size_t i = 0; i < num_groups; ++i) {
    out.emplace_back(packed.select(0, static_cast<int64_t>(i)));
  }
  return out;
}

struct WgradOutput {
  std::vector<at::Tensor> tensors;
  at::Tensor packed;
};

WgradOutput wgrad_output_from_arg(py::handle arg, bool compute_wgrad, size_t num_groups,
                                  at::ScalarType dtype, const c10::Device &device, int64_t rows,
                                  int64_t cols, const std::string &name) {
  WgradOutput out;
  if (!compute_wgrad) {
    return out;
  }
  if (is_none(arg)) {
    out.packed = at::empty({static_cast<int64_t>(num_groups), rows, cols},
                           at::device(device).dtype(dtype));
    out.tensors.reserve(num_groups);
    for (size_t i = 0; i < num_groups; ++i) {
      out.tensors.emplace_back(out.packed.select(0, static_cast<int64_t>(i)));
    }
    return out;
  }
  out.tensors = output_tensor_list_from_arg(arg, num_groups, dtype, name);
  return out;
}

at::Tensor grouped_gemm_wgrad(GroupedTensorWrapper *x, GroupedTensorWrapper *dy, py::handle output,
                              bool compute_wgrad, bool accumulate, bool use_split_accumulator,
                              size_t num_groups, at::ScalarType dtype, const c10::Device &device,
                              int64_t rows, int64_t cols, const std::string &name) {
  auto prepared =
      wgrad_output_from_arg(output, compute_wgrad, num_groups, dtype, device, rows, cols, name);
  if (prepared.tensors.empty()) {
    return at::Tensor();
  }

  std::vector<TensorWrapper> output_wrappers;
  auto output_nvte = nvte_tensor_list_from_tensors(prepared.tensors, &output_wrappers);
  GroupedGemmWorkspace workspace(prepared.tensors[0].device(), num_groups, use_split_accumulator,
                                 accumulate);
  NVTEGroupedMatmulConfig config =
      workspace.config.has_value() ? static_cast<NVTEGroupedMatmulConfig>(*workspace.config)
                                   : nullptr;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_gemm_with_discrete_out(
        x->data(), false, dy->data(), true, output_nvte.data(), num_groups, output_nvte.data(),
        num_groups, workspace.te_alpha.data(), workspace.te_beta.data(), workspace.te_setup.data(),
        workspace.te_cublas.data(), config, at::cuda::getCurrentCUDAStream());
  });
  return prepared.packed;
}

GroupedTensorWrapper make_grouped_bias(const std::vector<at::Tensor> &biases, at::ScalarType dtype,
                                       int64_t out_features, at::Tensor *bias_data) {
  NVTE_CHECK(!biases.empty(), "Bias tensor list must be non-empty.");
  if (biases.size() == 1 && biases[0].dim() == 2) {
    *bias_data = as_compute_tensor(biases[0], dtype);
  } else {
    std::vector<at::Tensor> rows;
    rows.reserve(biases.size());
    for (const auto &bias : biases) {
      rows.emplace_back(as_compute_tensor(bias.reshape({out_features}), dtype));
    }
    *bias_data = at::stack(rows, 0).contiguous();
  }
  auto grouped = GroupedTensorWrapper(
      biases.size(), std::vector<size_t>{biases.size(), static_cast<size_t>(out_features)});
  grouped.set_rowwise_data(bias_data->data_ptr(), GetTransformerEngineDType(dtype),
                           tensor_shape_1d(*bias_data));
  return grouped;
}

void add_grouped_bias(GroupedTensorWrapper *output, const std::vector<at::Tensor> &biases,
                      at::ScalarType dtype, int64_t out_features,
                      std::optional<at::Tensor> bias_scale = std::nullopt) {
  if (biases.empty()) {
    return;
  }
  at::Tensor bias_data;
  auto grouped_bias = make_grouped_bias(biases, dtype, out_features, &bias_data);
  if (bias_scale.has_value()) {
    auto scale = as_compute_tensor(bias_scale->reshape({-1}), at::kFloat);
    auto te_scale = makeTransformerEngineTensor(scale);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_grouped_scaled_bias_add(output->data(), grouped_bias.data(), te_scale.data(),
                                   at::cuda::getCurrentCUDAStream());
    });
  } else {
    NVTE_SCOPED_GIL_RELEASE({
      nvte_grouped_bias_add(output->data(), grouped_bias.data(), at::cuda::getCurrentCUDAStream());
    });
  }
}

bool is_gated_activation(const std::string &activation) {
  return activation == "swiglu" || activation == "clamped_swiglu" || activation == "geglu" ||
         activation == "reglu" || activation == "qgeglu" || activation == "sreglu";
}

at::Tensor maybe_deinterleave_glu(const at::Tensor &input, int64_t glu_interleave_size) {
  if (glu_interleave_size <= 0) {
    return input;
  }
  auto shape = input.sizes().vec();
  const int64_t last_dim = shape.back();
  NVTE_CHECK(last_dim % (2 * glu_interleave_size) == 0,
             "GLU interleaving requires the last dimension to be divisible by 2*interleave.");
  return input.reshape({-1, last_dim / (2 * glu_interleave_size), 2, glu_interleave_size})
      .transpose(1, 2)
      .contiguous()
      .view(shape);
}

at::Tensor maybe_reinterleave_glu_grad(const at::Tensor &input, int64_t glu_interleave_size) {
  if (glu_interleave_size <= 0) {
    return input;
  }
  auto shape = input.sizes().vec();
  const int64_t last_dim = shape.back();
  return input.reshape({-1, 2, last_dim / (2 * glu_interleave_size), glu_interleave_size})
      .transpose(1, 2)
      .contiguous()
      .view(shape);
}

at::Tensor activation_forward(const at::Tensor &input, const std::string &activation,
                              double activation_limit, double activation_alpha,
                              double activation_glu_linear_offset) {
  const int64_t out_features =
      is_gated_activation(activation) ? input.size(-1) / 2 : input.size(-1);
  auto output = at::empty({input.size(0), out_features}, input.options());
  auto te_input = makeTransformerEngineTensor(input);
  auto te_output = makeTransformerEngineTensor(output);
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    if (activation == "swiglu") {
      nvte_swiglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "glu") {
      nvte_glu(te_input.data(), te_output.data(), stream);
    } else if (activation == "geglu") {
      nvte_geglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "qgeglu") {
      nvte_qgeglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "reglu") {
      nvte_reglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "sreglu") {
      nvte_sreglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "clamped_swiglu") {
      nvte_clamped_swiglu_v2(te_input.data(), te_output.data(), static_cast<float>(activation_limit),
                             static_cast<float>(activation_alpha),
                             static_cast<float>(activation_glu_linear_offset), stream);
    } else if (activation == "srelu") {
      nvte_srelu(te_input.data(), te_output.data(), stream);
    } else if (activation == "gelu") {
      nvte_gelu(te_input.data(), te_output.data(), stream);
    } else if (activation == "qgelu") {
      nvte_qgelu(te_input.data(), te_output.data(), stream);
    } else if (activation == "relu") {
      nvte_relu(te_input.data(), te_output.data(), stream);
    } else if (activation == "silu") {
      nvte_silu(te_input.data(), te_output.data(), stream);
    } else {
      NVTE_ERROR("Unsupported megacpp grouped MLP activation: ", activation);
    }
  });
  return output;
}

at::Tensor activation_backward(const at::Tensor &grad, const at::Tensor &input,
                               const std::string &activation, double activation_limit,
                               double activation_alpha, double activation_glu_linear_offset) {
  auto output = at::empty_like(input);
  auto te_grad = makeTransformerEngineTensor(grad);
  auto te_input = makeTransformerEngineTensor(input);
  auto te_output = makeTransformerEngineTensor(output);
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    if (activation == "swiglu") {
      nvte_dswiglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "glu") {
      nvte_dglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "geglu") {
      nvte_dgeglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "qgeglu") {
      nvte_dqgeglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "reglu") {
      nvte_dreglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "sreglu") {
      nvte_dsreglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "clamped_swiglu") {
      nvte_clamped_dswiglu_v2(te_grad.data(), te_input.data(), te_output.data(),
                              static_cast<float>(activation_limit),
                              static_cast<float>(activation_alpha),
                              static_cast<float>(activation_glu_linear_offset), stream);
    } else if (activation == "srelu") {
      nvte_dsrelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "gelu") {
      nvte_dgelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "qgelu") {
      nvte_dqgelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "relu") {
      nvte_drelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "silu") {
      nvte_dsilu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else {
      NVTE_ERROR("Unsupported megacpp grouped MLP activation backward: ", activation);
    }
  });
  return output;
}

}  // namespace

std::vector<at::Tensor> megacpp_grouped_mlp_forward(
    const at::Tensor &input, const at::Tensor &split_sizes, py::handle fc1_weight,
    py::handle fc1_bias, py::handle fc2_weight, py::handle fc2_bias, const at::Tensor &scales,
    const std::string &activation, int64_t glu_interleave_size, double activation_limit,
    double activation_alpha, double activation_glu_linear_offset) {
  NVTE_CHECK(input.is_cuda(), "megacpp_grouped_mlp_forward requires CUDA input.");
  at::cuda::CUDAGuard device_guard(input.device());

  const auto num_groups = static_cast<size_t>(split_sizes.numel());
  NVTE_CHECK(num_groups > 0, "megacpp grouped MLP requires at least one group.");

  const auto dtype = input.scalar_type();
  NVTE_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
             "megacpp grouped MLP currently supports BF16/FP16 only.");

  auto fc1_weights = tensor_list_from_arg(fc1_weight, num_groups, dtype, "fc1_weight");
  auto fc2_weights = tensor_list_from_arg(fc2_weight, num_groups, dtype, "fc2_weight");
  auto fc1_biases = bias_list_from_arg(fc1_bias, num_groups, dtype, "fc1_bias");
  auto fc2_biases = bias_list_from_arg(fc2_bias, num_groups, dtype, "fc2_bias");

  const int64_t in_features = fc1_weights[0].size(1);
  const int64_t fc1_out_features = fc1_weights[0].size(0);
  const int64_t fc2_out_features = fc2_weights[0].size(0);
  const int64_t fc2_in_features = fc2_weights[0].size(1);
  const int64_t activation_out_features =
      is_gated_activation(activation) ? fc1_out_features / 2 : fc1_out_features;
  NVTE_CHECK(activation_out_features == fc2_in_features,
             "FC1 activation output dimension must match FC2 input dimension.");

  auto x = as_compute_tensor(input.reshape({-1, in_features}), dtype);
  auto split_metadata =
      prepare_grouped_splits(split_sizes, static_cast<int64_t>(num_groups),
                             std::vector<int64_t>{1, in_features, fc1_out_features,
                                                  fc2_in_features, fc2_out_features});
  // prepare_grouped_splits returns:
  //   [0] split sizes as int64
  //   [1] split_points as int32, consumed by cuDNN grouped GEMM paths
  //   [2..] tensor offsets in the same order as the stride list above.
  // megacpp uses cuBLAS grouped GEMM here, so split_points is intentionally skipped.
  auto split_sizes_i64 = split_metadata[0];
  auto base_offsets = split_metadata[2];
  auto x_offsets = split_metadata[3];
  auto fc1_offsets = split_metadata[4];
  auto fc2_offsets = split_metadata[5];
  auto output_offsets = split_metadata[6];
  const int64_t total_tokens = x.size(0);

  auto fc1_preact = at::empty({total_tokens, fc1_out_features}, x.options());
  auto grouped_x = make_grouped_tensor(x.reshape({-1}), split_sizes_i64, x_offsets, in_features);
  auto grouped_fc1_preact =
      make_grouped_tensor(fc1_preact.reshape({-1}), split_sizes_i64, fc1_offsets, fc1_out_features);
  grouped_gemm_discrete_weight(fc1_weights, true, &grouped_x, false, &grouped_fc1_preact, true);
  add_grouped_bias(&grouped_fc1_preact, fc1_biases, dtype, fc1_out_features);

  auto activation_input = maybe_deinterleave_glu(fc1_preact, glu_interleave_size);
  auto activation_unscaled = activation_forward(activation_input, activation, activation_limit,
                                                activation_alpha, activation_glu_linear_offset);
  auto fc2_x = activation_unscaled * scales.reshape({-1, 1}).to(x.options().dtype(dtype));

  std::vector<int64_t> out_shape = input.sizes().vec();
  out_shape.back() = fc2_out_features;
  auto output = at::empty(out_shape, x.options());
  auto output_2d = output.reshape({-1, fc2_out_features});
  auto grouped_fc2_x =
      make_grouped_tensor(fc2_x.reshape({-1}), split_sizes_i64, fc2_offsets, fc2_in_features);
  auto grouped_output =
      make_grouped_tensor(output_2d.reshape({-1}), split_sizes_i64, output_offsets, fc2_out_features);
  grouped_gemm_discrete_weight(fc2_weights, true, &grouped_fc2_x, false, &grouped_output, true);
  add_grouped_bias(&grouped_output, fc2_biases, dtype, fc2_out_features);

  return {output, x, split_sizes_i64, base_offsets, x_offsets, fc1_offsets, fc2_offsets,
          output_offsets, fc1_preact, fc2_x};
}

std::vector<at::Tensor> megacpp_grouped_mlp_backward(
    const at::Tensor &grad_output, const at::Tensor &split_sizes, const at::Tensor &x_offsets,
    const at::Tensor &fc1_offsets, const at::Tensor &fc2_offsets,
    const at::Tensor &fc2_dy_offsets, const at::Tensor &base_offsets,
    const at::Tensor &x, const at::Tensor &fc1_activation_input, const at::Tensor &fc2_x,
    const at::Tensor &scales, py::handle fc1_weight, py::handle fc2_weight,
    py::handle fc1_wgrad_output, bool fc1_compute_wgrad, bool fc1_accumulate_wgrad,
    py::handle fc2_wgrad_output, bool fc2_compute_wgrad, bool fc2_accumulate_wgrad,
    const std::string &activation, int64_t glu_interleave_size, double activation_limit,
    double activation_alpha, double activation_glu_linear_offset, bool input_requires_grad) {
  (void)base_offsets;
  NVTE_CHECK(grad_output.is_cuda(), "megacpp_grouped_mlp_backward requires CUDA grad_output.");
  at::cuda::CUDAGuard device_guard(grad_output.device());

  const auto num_groups = static_cast<size_t>(split_sizes.numel());
  const auto dtype = grad_output.scalar_type();
  auto fc1_weights = tensor_list_from_arg(fc1_weight, num_groups, dtype, "fc1_weight");
  auto fc2_weights = tensor_list_from_arg(fc2_weight, num_groups, dtype, "fc2_weight");

  const int64_t in_features = fc1_weights[0].size(1);
  const int64_t fc1_out_features = fc1_weights[0].size(0);
  const int64_t fc2_out_features = fc2_weights[0].size(0);
  const int64_t fc2_in_features = fc2_weights[0].size(1);

  auto split_sizes_i64 = as_cuda_i64_splits(split_sizes, grad_output.device());
  auto dy = as_compute_tensor(grad_output.reshape({-1, fc2_out_features}), dtype);
  const int64_t total_tokens = dy.size(0);

  auto grouped_dy =
      make_grouped_tensor(dy.reshape({-1}), split_sizes_i64, fc2_dy_offsets, fc2_out_features);
  at::Tensor fc2_wgrad_packed;
  if (fc2_compute_wgrad) {
    auto fc2_x_for_wgrad = as_compute_tensor(fc2_x.reshape({-1, fc2_in_features}), dtype);
    auto grouped_fc2_x_for_wgrad =
        make_grouped_tensor(fc2_x_for_wgrad.reshape({-1}), split_sizes_i64, fc2_offsets,
                            fc2_in_features);
    fc2_wgrad_packed =
        grouped_gemm_wgrad(&grouped_fc2_x_for_wgrad, &grouped_dy, fc2_wgrad_output,
                           fc2_compute_wgrad, fc2_accumulate_wgrad, true, num_groups, dtype,
                           fc2_weights[0].device(), fc2_out_features, fc2_in_features,
                           "fc2_wgrad_output");
  }

  auto fc2_dx = at::empty({total_tokens, fc2_in_features}, dy.options());
  auto grouped_fc2_dx =
      make_grouped_tensor(fc2_dx.reshape({-1}), split_sizes_i64, fc2_offsets, fc2_in_features);
  grouped_gemm_discrete_weight(fc2_weights, false, &grouped_dy, false, &grouped_fc2_dx, true);

  auto activation_input = maybe_deinterleave_glu(fc1_activation_input, glu_interleave_size);
  auto activation_unscaled = activation_forward(activation_input, activation, activation_limit,
                                                activation_alpha, activation_glu_linear_offset);
  auto grad_scales = (activation_unscaled * fc2_dx).sum(-1);
  auto grad_activation_unscaled = fc2_dx * scales.reshape({-1, 1}).to(dy.options().dtype(dtype));
  auto fc1_dy_deinterleaved =
      activation_backward(grad_activation_unscaled, activation_input, activation, activation_limit,
                          activation_alpha, activation_glu_linear_offset);
  auto fc1_dy = maybe_reinterleave_glu_grad(fc1_dy_deinterleaved, glu_interleave_size);
  at::Tensor fc1_wgrad_packed;
  if (fc1_compute_wgrad) {
    auto x_for_wgrad = as_compute_tensor(x.reshape({-1, in_features}), dtype);
    auto grouped_x_for_wgrad =
        make_grouped_tensor(x_for_wgrad.reshape({-1}), split_sizes_i64, x_offsets, in_features);
    auto grouped_fc1_dy_for_wgrad =
        make_grouped_tensor(fc1_dy.reshape({-1}), split_sizes_i64, fc1_offsets, fc1_out_features);
    fc1_wgrad_packed =
        grouped_gemm_wgrad(&grouped_x_for_wgrad, &grouped_fc1_dy_for_wgrad, fc1_wgrad_output,
                           fc1_compute_wgrad, fc1_accumulate_wgrad, true, num_groups, dtype,
                           fc1_weights[0].device(), fc1_out_features, in_features,
                           "fc1_wgrad_output");
  }

  at::Tensor grad_input;
  if (input_requires_grad) {
    std::vector<int64_t> grad_input_shape = grad_output.sizes().vec();
    grad_input_shape.back() = in_features;
    grad_input = at::empty(grad_input_shape, dy.options());
    auto grad_input_2d = grad_input.reshape({-1, in_features});
    auto grouped_fc1_dy =
        make_grouped_tensor(fc1_dy.reshape({-1}), split_sizes_i64, fc1_offsets, fc1_out_features);
    auto grouped_grad_input = make_grouped_tensor(grad_input_2d.reshape({-1}), split_sizes_i64,
                                                 x_offsets, in_features);
    grouped_gemm_discrete_weight(fc1_weights, false, &grouped_fc1_dy, false, &grouped_grad_input,
                                 true);
  } else {
    grad_input = at::empty({0}, dy.options());
  }

  auto empty_return = at::empty({0}, dy.options());
  if (!fc1_wgrad_packed.defined()) {
    fc1_wgrad_packed = empty_return;
  }
  if (!fc2_wgrad_packed.defined()) {
    fc2_wgrad_packed = empty_return;
  }
  return {grad_input, fc1_dy, fc2_dx, grad_scales, fc1_wgrad_packed, fc2_wgrad_packed};
}

}  // namespace transformer_engine::pytorch

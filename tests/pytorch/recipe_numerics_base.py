# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Iterable, Optional
import os
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

import transformer_engine_torch as tex

from transformer_engine.common.recipe import PerTensorCurrentScaling
from transformer_engine.pytorch.fp8 import fp8_autocast, get_fp8_torch_dtype


class GetRecipes:

    @staticmethod
    def none():
        return None

    @staticmethod
    def fp8_per_tensor_current_scaling_default():
        # return default configs
        return PerTensorCurrentScaling()


# base class for validating current_scaling x linear layer
class TestFP8RecipeLinearBase:
    @staticmethod
    def _prepare_data(
        batch_size, hidden_size, out_size, use_bias=True, seed=0, dtype=torch.float32
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
        w = torch.randn((out_size, hidden_size), dtype=dtype, device="cuda")
        bias = torch.randn((out_size), dtype=dtype, device="cuda") if use_bias else None
        gradient = torch.randn((batch_size, out_size), dtype=dtype, device="cuda")

        return x, w, bias, gradient

    @staticmethod
    def _shard_tensor(x, world_size, axis):
        split_size = x.size()[axis] // world_size
        split_tensor = torch.split(x, split_size, axis)
        out = []
        for tensor in split_tensor:
            out.append(tensor.detach().clone().requires_grad_(x.requires_grad))
        return out

    @staticmethod
    def _gather_tensor(local, world_size, tp_group, concat_dim):
        out_list = [torch.zeros_like(local) for _ in range(world_size)]
        torch.distributed.all_gather(out_list, local, tp_group)
        return torch.cat(out_list, dim=concat_dim)

    @staticmethod
    def _all_reduce_tensor(local, world_size, tp_group):
        if world_size == 1:
            return local
        handle = torch.distributed.all_reduce(local, group=tp_group, async_op=False)
        return local

    @staticmethod
    def _get_sum_abs_error(a, b):
        return torch.sum(torch.abs(a - b))

    @staticmethod
    def _load_golden_tensor_values(a, b):
        return torch.sum(torch.abs(a - b))

    @staticmethod
    def _check_golden_tensor_dumps(dump_dir, get_recipe, dims, input_dtype):
        recipe = get_recipe()
        batch_size, hidden_size, out_size = dims
        fp8_type_x = get_fp8_torch_dtype(recipe, fprop_tensor=True)
        fp8_type_w = get_fp8_torch_dtype(recipe, fprop_tensor=True)
        fp8_type_g = get_fp8_torch_dtype(recipe, fprop_tensor=False)

        # Expected tensor names based on the naming template
        scaling_type = (  # Assuming the scaling type is PER_TENSOR for this example
            "ScalingType.PER_TENSOR"
        )
        current_seed = torch.initial_seed()  # Get the current seed

        expected_tensor_names = {
            "y": f"y_{scaling_type}_{batch_size}_{hidden_size}_{out_size}_{current_seed}_{input_dtype}_{fp8_type_x}_{fp8_type_w}_{fp8_type_g}.pt",
            "dgrad": f"dgrad_{scaling_type}_{batch_size}_{hidden_size}_{out_size}_{current_seed}_{input_dtype}_{fp8_type_x}_{fp8_type_w}_{fp8_type_g}.pt",
            "wgrad": f"wgrad_{scaling_type}_{batch_size}_{hidden_size}_{out_size}_{current_seed}_{input_dtype}_{fp8_type_x}_{fp8_type_w}_{fp8_type_g}.pt",
            "bgrad": f"bgrad_{scaling_type}_{batch_size}_{hidden_size}_{out_size}_{current_seed}_{input_dtype}_{fp8_type_x}_{fp8_type_w}_{fp8_type_g}.pt",
        }

        # Check if all expected tensors are in the tensor dumps directory
        tensor_map = {}
        for tensor_key, tensor_name in expected_tensor_names.items():
            tensor_path = dump_dir / tensor_name
            if not os.path.exists(tensor_path):
                print(f"Missing tensor: {tensor_name}")
                return None

            # Load the tensor
            tensor_map[tensor_key] = torch.load(tensor_path)
        return tensor_map

    @classmethod
    def run_linear_preprocess_parallel(
        cls,
        x,
        w,
        bias,
        gradient,
        parallel_mode=None,
        sequence_parallel=False,
        tp_size=1,
        rank=0,
    ):
        if tp_size > 1:
            if parallel_mode == "column":
                # split w in N dim, which should be axis 0
                w = cls._shard_tensor(w, tp_size, 0)[rank]
                bias = cls._shard_tensor(bias, tp_size, 0)[rank] if bias is not None else None
                # split gradient in N dim, which should be axis 1
                gradient = cls._shard_tensor(gradient, tp_size, 1)[rank]
                if sequence_parallel:
                    # split x in M dim, which should be axis 0
                    x = cls._shard_tensor(x, tp_size, 0)[rank]
            # row parallel, split x in k dim, which should be axis 1, split w in k dim, should be axis 1
            if parallel_mode == "row":
                # split x in K dim, which should be axis 1
                x = cls._shard_tensor(x, tp_size, 1)[rank]
                # split w in K dim, which should be axis 1
                w = cls._shard_tensor(w, tp_size, 1)[rank]
                if sequence_parallel:
                    # split gradient in M dim, which should be axis 0
                    gradient = cls._shard_tensor(gradient, tp_size, 0)[rank]
        return x, w, bias, gradient

    @classmethod
    def run_linear_postprocess_parallel(
        cls,
        y_q,
        dgrad,
        wgrad,
        bgrad,
        parallel_mode,
        sequence_parallel,
        tp_size,
        tp_group,
    ):
        if tp_size > 1:
            if parallel_mode == "column":
                # gather y_q in N dim, which should be axis 1
                y_q = cls._gather_tensor(y_q, tp_size, tp_group, 1)
                # gather wgrad in N dim, which should be axis 0
                wgrad = cls._gather_tensor(wgrad, tp_size, tp_group, 0)
                # gather bgrad in N dim, which should be axis 0
                bgrad = (
                    cls._gather_tensor(bgrad, tp_size, tp_group, 0) if bgrad is not None else None
                )
                if sequence_parallel:
                    # gather dgrad in M dim, which should be axis 0
                    dgrad = cls._gather_tensor(dgrad, tp_size, tp_group, 0)
            if parallel_mode == "row":
                # gather dgrad in K dim, which should be axis 1
                dgrad = cls._gather_tensor(dgrad, tp_size, tp_group, 1)
                # gather wgrad in K dim, which should be axis 1
                wgrad = cls._gather_tensor(wgrad, tp_size, tp_group, 1)
                if sequence_parallel:
                    # gather y_q in M dim, which should be axis 0
                    y_q = cls._gather_tensor(y_q, tp_size, tp_group, 0)
                    # we need to sum bias gradient when using TP + SP
                    bgrad = (
                        cls._all_reduce_tensor(bgrad, tp_size, tp_group)
                        if bgrad is not None
                        else None
                    )

        return y_q, dgrad, wgrad, bgrad

    @classmethod
    def run_linear_one_step(
        cls, layer, x, gradient, is_first_microbatch=None, fuse_wgrad_accumulation=False
    ):
        # reset gradients
        layer.zero_grad()
        x.grad = None

        # Forward pass
        if isinstance(layer, te.Linear):
            # Kitchen Linear
            y_q = layer.forward(x, is_first_microbatch=is_first_microbatch)
        else:
            # the default torch.nn.Linear
            y_q = layer(x)

        # Backward pass
        y_q.backward(gradient)

        # Collect gradients
        dgrad = x.grad
        bgrad = (
            layer._parameters["bias"].grad
            if layer._parameters.get("bias", None) is not None
            else None
        )
        assert "weight" in layer._parameters
        if fuse_wgrad_accumulation:
            wgrad = layer._parameters["weight"].main_grad
            assert layer._parameters["weight"].grad is None
        else:
            wgrad = layer._parameters["weight"].grad

        return y_q, dgrad, wgrad, bgrad

    @classmethod
    def run_linear_multiple_steps(
        cls,
        layer,
        x,
        gradient,
        run_num_steps,
        enable_weight_cache,
        fuse_wgrad_accumulation=False,
    ):
        """
        Run multiple steps of linear layer and collect results.
        """

        y_q_list, dgrad_list, wgrad_list = [], [], []
        bgrad_list = [] if layer._parameters.get("bias", None) is not None else None

        for i in range(run_num_steps):
            x_i = (x + i).clone().detach().requires_grad_(True)
            # run_linear_one_step
            y_q, dgrad, wgrad, bgrad = cls.run_linear_one_step(
                layer,
                x_i,
                gradient,
                is_first_microbatch=(i == 0) if enable_weight_cache else None,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            )

            # Collect results
            y_q_list.append(y_q.detach().clone())
            dgrad_list.append(dgrad.detach().clone())
            wgrad_list.append(wgrad.detach().clone())
            if bgrad_list is not None and bgrad is not None:
                bgrad_list.append(bgrad.detach().clone())

    @classmethod
    def run_linear(
        cls,
        x,
        w,
        bias,
        gradient,
        parallel_mode=None,
        sequence_parallel=False,
        tp_group=None,
        tp_size=1,
        rank=0,
        run_num_steps=1,
        enable_weight_cache=False,
        fuse_wgrad_accumulation=False,
    ):
        """
        If Model parallel, split inputs for a given rank and return the gathered output and gradients, so that they can be compared with
        the reference single GPU run.
        """
        # clone inputs and move to current device
        # w has shape [N, K], x has shape [M, K], gradient has shape [M, N]
        x = x.clone().detach().requires_grad_(True).to("cuda")
        w = w.clone().detach().to("cuda")
        gradient = gradient.clone().detach().to("cuda")
        bias = bias.clone().detach().to("cuda") if bias is not None else None
        in_features = x.shape[1]
        out_features = w.shape[0]

        # If Model parallel: split inputs for a given rank
        x, w, bias, gradient = cls.run_linear_preprocess_parallel(
            x, w, bias, gradient, parallel_mode, sequence_parallel, tp_size, rank
        )

        # set data types
        params_dtype = x.dtype

        # Create linear layer and copy weights
        layer = te.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_size=tp_size,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )

        layer = layer.to("cuda")

        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None:
                layer.bias.copy_(bias)

        if fuse_wgrad_accumulation:
            assert (
                run_num_steps > 1
            ), "Fused weight gradient accumulation requires run_num_steps > 1"
            layer.weight.main_grad = torch.zeros_like(layer.weight)

        # Run one step or multiple steps
        if run_num_steps == 1:
            y_q, dgrad, wgrad, bgrad = cls.run_linear_one_step(layer, x, gradient)
        else:
            y_q, dgrad, wgrad, bgrad = cls.run_linear_multiple_steps(
                layer,
                x,
                gradient,
                run_num_steps,
                enable_weight_cache,
                fuse_wgrad_accumulation,
            )

        # If Model parallel: gather output and gradients from all ranks
        y_q, dgrad, wgrad, bgrad = cls.run_linear_postprocess_parallel(
            y_q,
            dgrad,
            wgrad,
            bgrad,
            parallel_mode,
            sequence_parallel,
            tp_size,
            tp_group,
        )

        return y_q, dgrad, wgrad, bgrad

    def compare_recipe(
        self,
        recipe1,
        recipe2,
        batch_size,
        hidden_size,
        out_size,
        use_bias,
        seed,
        dtype,
        y_error=0.0,
        dgrad_error=0.0,
        wgrad_error=0.0,
        bgrad_error=0.0,
        recipe1_golden_tensors=None,
        recipe2_golden_tensors=None,
    ):
        x, w, bias, gradient = self._prepare_data(
            batch_size, hidden_size, out_size, use_bias, seed=seed, dtype=dtype
        )

        # recipe1
        using_fp8_recipe = recipe1 != GetRecipes.none
        if using_fp8_recipe:
            with fp8_autocast(enabled=True, fp8_recipe=recipe1()):
                y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(x, w, bias, gradient)
        else:
            y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(x, w, bias, gradient)

        # recipe2
        using_fp8_recipe = recipe2 != GetRecipes.none
        if using_fp8_recipe:
            with fp8_autocast(enabled=True, fp8_recipe=recipe2()):
                y_q, dgrad, wgrad, bgrad = self.run_linear(x, w, bias, gradient)
        else:
            y_q, dgrad, wgrad, bgrad = self.run_linear(x, w, bias, gradient)

        # Compare results
        torch.testing.assert_close(y_q.float(), y_q_ref.float(), rtol=0.0, atol=y_error)
        torch.testing.assert_close(dgrad, dgrad_ref, rtol=0.0, atol=dgrad_error)
        torch.testing.assert_close(wgrad, wgrad_ref, rtol=0.0, atol=wgrad_error)
        if use_bias:
            torch.testing.assert_close(bgrad, bgrad_ref, rtol=0.0, atol=bgrad_error)

        if recipe2_golden_tensors is not None:
            print(y_q)
            torch.testing.assert_close(
                y_q.float(), recipe2_golden_tensors["y"].float(), atol=0.25, rtol=0.0
            )
            torch.testing.assert_close(dgrad, recipe2_golden_tensors["dgrad"], atol=0.0, rtol=0.0)
            torch.testing.assert_close(wgrad, recipe2_golden_tensors["wgrad"], atol=0.0, rtol=0.0)
            if use_bias:
                torch.testing.assert_close(
                    bgrad, recipe2_golden_tensors["bgrad"], atol=0.0, rtol=0.0
                )

        # measure numerical difference
        print("y abs sum error: ", self._get_sum_abs_error(y_q, y_q_ref))
        print("dgrad abs sum error: ", self._get_sum_abs_error(dgrad, dgrad_ref))
        print("wgrad abs sum error: ", self._get_sum_abs_error(wgrad, wgrad_ref))
        if use_bias:
            print("bgrad abs sum error: ", self._get_sum_abs_error(bgrad, bgrad_ref))

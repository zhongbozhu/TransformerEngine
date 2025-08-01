..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Jax
=======

Pre-defined Variable of Logical Axes
------------------------------------
Variables are available in `transformer_engine.jax.sharding`.

* BATCH_AXES: The logical axis of batch dimension. It is usually sharded along DP + FSDP on Mesh.
* SEQLEN_AXES: The logical axis of sequence length dimension. It is usually not sharded.
* SEQLEN_TP_AXES: The logical axis of sequence length dimension. It is usually sharded along TP on Mesh.
* HEAD_AXES: The logical axis of head dimension of MHA. It is usually sharded along TP on Mesh.
* HIDDEN_AXES: The logical axis of hidden dimension. It is usually not sharded.
* HIDDEN_TP_AXES: The logical axis of hidden dimension. It is usually sharded along TP on Mesh.
* JOINED_AXES: The logical axis of non-defined dimension. It is usually not sharded.


Checkpointing
------------------------------------
When using checkpointing with Transformer Engine JAX, please be aware of the checkpointing policy being applied to your model. Any JAX checkpointing policy using `dot`, such as `jax.checkpoint_policies.dots_with_no_batch_dims`, may not work with GEMMs provided by Transformer Engine as they do not always use the `jax.lax.dot_general` primitive. Instead, you can use `transformer_engine.jax.checkpoint_policies.dots_and_te_gemms_with_no_batch_dims` or similar policies that are designed to work with Transformer Engine's GEMMs and `jax.lax.dot_general` GEMMs. You may also use any JAX policies that do not filter by primitive, such as `jax.checkpoint_policies.save_only_these_names` or `jax.checkpoint_policies.everything_saveable`.

Modules
------------------------------------
.. autoapiclass:: transformer_engine.jax.flax.TransformerLayerType
.. autoapiclass:: transformer_engine.jax.MeshResource()


.. autoapifunction:: transformer_engine.jax.fp8_autocast
.. autoapifunction:: transformer_engine.jax.update_collections


.. autoapiclass:: transformer_engine.jax.flax.LayerNorm(epsilon=1e-6, layernorm_type='layernorm', **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.DenseGeneral(features, layernorm_type='layernorm', use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.LayerNormDenseGeneral(features, layernorm_type='layernorm', epsilon=1e-6, use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.LayerNormMLP(intermediate_dim=2048, layernorm_type='layernorm', epsilon=1e-6, use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.RelativePositionBiases(num_buckets, max_distance, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.DotProductAttention(head_dim, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.MultiHeadAttention(head_dim, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.TransformerLayer(hidden_size=512, mlp_hidden_size=2048, num_attention_heads=8, **kwargs)
  :members: __call__

.. autoapifunction:: transformer_engine.jax.flax.extend_logical_axis_rules

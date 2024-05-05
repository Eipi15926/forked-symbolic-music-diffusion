"""Copyed from flax-0.3.0 flax.nn.attention"""
class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=zeros,
            bias=True,
            attention_fn=dot_product_attention):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token w/ False.
      key_padding_mask: boolean specifying key-value tokens that are pad token
        w/ False.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape
      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    if inputs_kv is None:
      inputs_kv = inputs_q

    is_self_attention = inputs_kv is inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    if cache:
      assert isinstance(cache, Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(onp.array((key.ndim,) + key.shape[-2:], dtype=onp.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        if not isinstance(cache_entry, _CacheEntry):
          raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        indices = [0] * len(cshape)
        i = cache_entry.i
        attn_size = onp.prod(onp.take(cshape, attention_axis))
        for attn_dim in attention_axis:
          attn_size //= cshape[attn_dim]
          indices[attn_dim] = i // attn_size
          i = i % attn_size

        key = lax.dynamic_update_slice(cache_entry.key, key, indices)
        value = lax.dynamic_update_slice(cache_entry.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one,
                                          key=key,
                                          value=value)
        cache.store(cache_entry)

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(onp.take(key.shape, attention_axis))
        attn_size = onp.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if (padding_mask is not None or key_padding_mask is not None) and not cache:
      if key_padding_mask is None:
        if is_self_attention:
          key_padding_mask = padding_mask
        else:
          key_padding_shape = [inputs_kv.shape[dim] for dim in attention_axis]
          key_padding_mask = jnp.full(key_padding_shape, True)
      if padding_mask is None:
        if is_self_attention:
          padding_mask = key_padding_mask
        else:
          padding_shape = [inputs_q.shape[dim] for dim in attention_axis]
          padding_mask = jnp.full(padding_shape, True)

      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        assert is_self_attention
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x = attention_fn(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    # back to the original inputs dimensions
    out = DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out


# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
SelfAttention = MultiHeadDotProductAttention.partial(inputs_kv=None)

"""copyed from flax-0.3.0 flax.nn.normalization"""
class LayerNorm(base.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  Operates on the last axis of the input data.
  """

  def apply(self,
            x,
            epsilon=1e-6,
            dtype=jnp.float32,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones):
    """Applies layer normalization on the input.

    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Args:
      x: the inputs
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  If True, bias (beta) is added.
      scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + epsilon)
    if scale:
      mul = mul * jnp.asarray(self.param('scale', (features,), scale_init),
                              dtype)
    y = (x - mean) * mul
    if bias:
      y = y + jnp.asarray(self.param('bias', (features,), bias_init), dtype)
    return jnp.asarray(y, dtype)
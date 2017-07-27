// Adapted from https://github.com/Maratyszcza/NNPACK/blob/master/include/nnpack.h

export const NNP_STATUS = {
  /** The call succeeded, and all output arguments now contain valid data. */
  SUCCESS: 0,
  /** NNPACK function was called with batch_size == 0. */
  INVALID_BATCH_SIZE: 2,
  /** NNPACK function was called with channels == 0. */
  INVALID_CHANNELS: 3,
  /** NNPACK function was called with input_channels == 0. */
  INVALID_INPUT_CHANNELS: 4,
  /** NNPACK function was called with output_channels == 0. */
  INVALID_OUTPUT_CHANNELS: 5,
  /** NNPACK function was called with input_size.height == 0 or input_size.width == 0 */
  INVALID_INPUT_SIZE: 10,
  /** NNPACK function was called with input_stride.height == 0 or input_stride.width == 0 */
  INVALID_INPUT_STRIDE: 11,
  /** NNPACK function was called with input_padding not less than respective kernel (or pooling) size, i.e.:
	 *
	 *  - input_padding.left   >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.right  >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.top    >= kernel_size.height (>= pooling_size.height)
	 *  - input_padding.bottom >= kernel_size.height (>= pooling_size.height)
	 */
  INVALID_INPUT_PADDING: 12,
  /** NNPACK function was called with kernel_size.height == 0 or kernel_size.width == 0 */
  INVALID_KERNEL_SIZE: 13,
  /** NNPACK function was called with pooling_size.height == 0 or pooling_size.width == 0 */
  INVALID_POOLING_SIZE: 14,
  /** NNPACK function was called with pooling_stride.height == 0 or pooling_stride.width == 0 */
  INVALID_POOLING_STRIDE: 15,
  /** NNPACK function was called with convolution algorithm not in nnp_convolution_algorithm enumeration */
  INVALID_ALGORITHM: 16,
  /** NNPACK function was called with convolution transform strategy not in nnp_convolution_transform_strategy enum */
  INVALID_TRANSFORM_STRATEGY: 17,
  /** NNPACK function was called with output_subsampling.height == 0 or output_subsampling.width == 0 */
  INVALID_OUTPUT_SUBSAMPLING: 13,
  /** NNPACK function was called with activation not in nnp_activation enum */
  INVALID_ACTIVATION: 14,
  /** NNPACK function was called with invalid activation parameters */
  INVALID_ACTIVATION_PARAMETERS: 15,

  /** NNPACK does not support the particular input size for the function */
  UNSUPPORTED_INPUT_SIZE: 20,
  /** NNPACK does not support the particular input stride for the function */
  UNSUPPORTED_INPUT_STRIDE: 21,
  /** NNPACK does not support the particular input padding for the function */
  UNSUPPORTED_INPUT_PADDING: 22,
  /** NNPACK does not support the particular kernel size for the function */
  UNSUPPORTED_KERNEL_SIZE: 23,
  /** NNPACK does not support the particular pooling size for the function */
  UNSUPPORTED_POOLING_SIZE: 24,
  /** NNPACK does not support the particular pooling stride for the function */
  UNSUPPORTED_POOLING_STRIDE: 25,
  /** NNPACK does not support the particular convolution algorithm for the function */
  UNSUPPORTED_ALGORITHM: 26,
  /** NNPACK does not support the particular convolution transform strategy for the algorithm */
  UNSUPPORTED_TRANSFORM_STRATEGY: 27,
  /** NNPACK does not support the particular activation function for the function */
  UNSUPPORTED_ACTIVATION: 28,
  /** NNPACK does not support the particular activation function parameters for the function */
  UNSUPPORTED_ACTIVATION_PARAMETERS: 29,

  /** NNPACK function was called before the library was initialized */
  UNINITIALIZED: 50,
  /** NNPACK does not implement this function for the host CPU */
  UNSUPPORTED_HARDWARE: 51,
  /** NNPACK failed to allocate memory for temporary buffers */
  OUT_OF_MEMORY: 52,
  /** Scratch space buffer is too small */
  INSUFFICIENT_BUFFER: 53,
  /** Scratch space buffer is not properly aligned */
  MISALIGNED_BUFFER: 54
}

/**
 * @brief Activation applied applied after a convolutional or fully-connected layer.
 */
export const NNP_ACTIVATION = {
  /** Identity activation f(x) := x, i.e. no transformation */
  IDENTITY: 0,
  /** ReLU activation f(x) := max(0, x) */
  RELU: 1
}

/**
 * @brief Algorithm for computing convolutional layers.
 */
export const NNP_CONVOLUTION_ALGORITHM = {
  /** Let NNPACK choose the algorithm depending on layer parameters */
  AUTO: 0,
  /** Tiled convolution based on 2D Fourier transform with 8x8 blocks. Supports kernels up to 8x8. */
  FT8X8: 1,
  /** Tiled convolution based on 2D Fourier transform with 16x16 blocks. Supports kernels up to 16x16. */
  FT16X16: 2,
  /** Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks. Supports only 3x3 kernels. */
  WT8X8: 3,
  /** Direct convolution via implicit GEMM. */
  IMPLICIT_GEMM: 4,
  /** Direct convolution implementation. */
  DIRECT: 5
}

export const NNP_CONVOLUTION_TRANSFORM_STRATEGY = {
  COMPUTE: 1,
  PRECOMPUTE: 2,
  REUSE: 3
}

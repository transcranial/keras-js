import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * _Pooling3D layer class
 */
export default class _Pooling3D extends Layer {
  /**
   * Creates a _Pooling3D layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_Pooling3D'

    const { pool_size = [2, 2, 2], strides = null, padding = 'valid', data_format = 'channels_last' } = attrs

    if (Array.isArray(pool_size)) {
      this.poolSize = pool_size
    } else {
      this.poolSize = [pool_size, pool_size, pool_size]
    }

    if (Array.isArray(strides)) {
      this.strides = strides
    } else if (strides !== null) {
      this.strides = [strides, strides, strides]
    } else {
      this.strides = this.poolSize
    }

    this.padding = padding
    this.dataFormat = data_format

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max'
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * @param {Tensor} x
   */
  _calcOutputShape(x) {
    const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape
    const [poolDim1, poolDim2, poolDim3] = this.poolSize

    const outputDim1 = this.padding === 'same'
      ? Math.floor((inputDim1 + this.strides[0] - 1) / this.strides[0])
      : Math.floor((inputDim1 - poolDim1 + this.strides[0]) / this.strides[0])
    const outputDim2 = this.padding === 'same'
      ? Math.floor((inputDim2 + this.strides[1] - 1) / this.strides[1])
      : Math.floor((inputDim2 - poolDim2 + this.strides[1]) / this.strides[1])
    const outputDim3 = this.padding === 'same'
      ? Math.floor((inputDim3 + this.strides[2] - 1) / this.strides[2])
      : Math.floor((inputDim3 - poolDim3 + this.strides[2]) / this.strides[2])

    const paddingDim1 = this.padding === 'same'
      ? Math.max(0, Math.floor((outputDim1 - 1) * this.strides[0] + poolDim1 - inputDim1))
      : 0
    const paddingDim2 = this.padding === 'same'
      ? Math.max(0, Math.floor((outputDim2 - 1) * this.strides[1] + poolDim2 - inputDim2))
      : 0
    const paddingDim3 = this.padding === 'same'
      ? Math.max(0, Math.floor((outputDim3 - 1) * this.strides[2] + poolDim3 - inputDim3))
      : 0
    const paddingDim1Before = Math.floor(paddingDim1 / 2)
    const paddingDim1After = paddingDim1 - paddingDim1Before
    const paddingDim2Before = Math.floor(paddingDim2 / 2)
    const paddingDim2After = paddingDim2 - paddingDim2Before
    const paddingDim3Before = Math.floor(paddingDim3 / 2)
    const paddingDim3After = paddingDim3 - paddingDim3Before

    this.outputShape = [outputDim1, outputDim2, outputDim3, inputChannels]
    this.inputPadding = [
      paddingDim1Before,
      paddingDim1After,
      paddingDim2Before,
      paddingDim2After,
      paddingDim3Before,
      paddingDim3After
    ]
  }

  /**
   * Pad input tensor if necessary, for padding='same'.
   * See above for notes on calculating padding.
   * For max, we pad with -infinity.
   * For average we pad with zero.
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _padInput(x) {
    if (this.padding === 'same') {
      const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape
      const [
        paddingDim1Before,
        paddingDim1After,
        paddingDim2Before,
        paddingDim2After,
        paddingDim3Before,
        paddingDim3After
      ] = this.inputPadding
      const newDim1 = inputDim1 + paddingDim1Before + paddingDim1After
      const newDim2 = inputDim2 + paddingDim2Before + paddingDim2After
      const newDim3 = inputDim3 + paddingDim3Before + paddingDim3After

      let _x = new Tensor([], [newDim1, newDim2, newDim3, inputChannels])
      if (this.poolingFunc === 'max') {
        ops.assigns(_x.tensor, Number.NEGATIVE_INFINITY)
      }

      ops.assign(
        _x.tensor
          .hi(
            inputDim1 + paddingDim1Before,
            inputDim2 + paddingDim2Before,
            inputDim3 + paddingDim3Before,
            inputChannels
          )
          .lo(paddingDim1Before, paddingDim2Before, paddingDim3Before, 0),
        x.tensor
      )
      x.tensor = _x.tensor
    }
    return x
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (this.poolingFunc !== 'max' && this.poolingFunc !== 'average') {
      throw new Error(`[pooling._Pooling3D] pooling function must be max or average.`)
    }

    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0)
    }

    this._calcOutputShape(x)
    this._padInput(x)

    const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape
    const [poolDim1, poolDim2, poolDim3] = this.poolSize
    let y = new Tensor([], this.outputShape)
    let patch = new Tensor([], [poolDim1, poolDim2, poolDim3, inputChannels])

    // keep track of padding since these values are not included in pooling
    // for max, we can ignore since padding values are set to -infinity
    const [
      paddingDim1Before,
      paddingDim1After,
      paddingDim2Before,
      paddingDim2After,
      paddingDim3Before,
      paddingDim3After
    ] = this.inputPadding

    for (let i = 0, _i = 0; i <= inputDim1 - poolDim1; (i += this.strides[0]), _i++) {
      let dim1InPadding = 0
      if (i < paddingDim1Before) {
        dim1InPadding = paddingDim1Before - i
      } else if (i + poolDim1 > inputDim1 - paddingDim1After) {
        dim1InPadding = i + poolDim1 - (inputDim1 - paddingDim1After)
      }

      for (let j = 0, _j = 0; j <= inputDim2 - poolDim2; (j += this.strides[1]), _j++) {
        let dim2InPadding = 0
        if (j < paddingDim2Before) {
          dim2InPadding = paddingDim2Before - j
        } else if (j + poolDim2 > inputDim2 - paddingDim2After) {
          dim2InPadding = j + poolDim2 - (inputDim2 - paddingDim2After)
        }

        for (let k = 0, _k = 0; k <= inputDim3 - poolDim3; (k += this.strides[2]), _k++) {
          let dim3InPadding = 0
          if (k < paddingDim3Before) {
            dim3InPadding = paddingDim3Before - k
          } else if (k + poolDim3 > inputDim3 - paddingDim3After) {
            dim3InPadding = k + poolDim3 - (inputDim3 - paddingDim3After)
          }

          ops.assign(patch.tensor, x.tensor.hi(i + poolDim1, j + poolDim2, k + poolDim3, inputChannels).lo(i, j, k, 0))
          for (let c = 0; c < inputChannels; c++) {
            if (this.poolingFunc === 'max') {
              y.tensor.set(_i, _j, _k, c, ops.sup(patch.tensor.pick(null, null, null, c)))
            } else if (this.poolingFunc === 'average') {
              let nbCellsEffective =
                (poolDim1 - dim1InPadding) * (poolDim2 - dim2InPadding) * (poolDim3 - dim3InPadding)
              y.tensor.set(_i, _j, _k, c, ops.sum(patch.tensor.pick(null, null, null, c)) / nbCellsEffective)
            }
          }
        }
      }
    }

    x.tensor = y.tensor

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(3, 0, 1, 2)
    }

    return x
  }
}

import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'

/**
 * Conv3D layer class
 */
export default class Conv3D extends Layer {
  /**
   * Creates a Conv3D layer
   * @param {Number} attrs.filters - Number of convolution filters to use.
   * @param {Array<Number>|Number} attrs.kernel_size - Size of the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Conv3D'

    const {
      filters = 1,
      kernel_size = [1, 1, 1],
      strides = [1, 1, 1],
      padding = 'valid',
      data_format = 'channels_last',
      dilation_rate = [1, 1, 1],
      activation = 'linear',
      use_bias = true
    } = attrs

    if (Array.isArray(kernel_size)) {
      this.kernelShape = [filters, ...kernel_size]
    } else {
      this.kernelShape = [filters, kernel_size, kernel_size, kernel_size]
    }

    if (Array.isArray(strides)) {
      this.strides = strides
    } else {
      this.strides = [strides, strides, strides]
    }

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding
    } else {
      throw new Error(`${this.name} [Conv3D layer] Invalid padding.`)
    }

    if (data_format === 'channels_last' || data_format === 'channels_first') {
      this.dataFormat = data_format
    } else {
      throw new Error(`${this.name} [Conv3D layer] Only channels_last and channels_first data formats are allowed.`)
    }

    if (Array.isArray(dilation_rate)) {
      this.dilationRate = dilation_rate
    } else {
      this.dilationRate = [dilation_rate, dilation_rate, dilation_rate]
    }
    if (
      (this.dilationRate[0] !== 1 || this.dilationRate[1] !== 1 || this.dilationRate[2] !== 1) &&
      (this.strides[0] !== 1 || this.strides[1] !== 1 || this.strides[2] !== 1)
    ) {
      // Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1
      // https://keras.io/layers/convolutional/#conv3d
      throw new Error(`${this.name} [Conv3D layer] Incompatible combination of dilation_rate with strides.`)
    }

    this.activation = activation
    this.activationFunc = activations[activation]

    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel']

    // Enable layer gpu +/- pipeline mode if supported
    if (this.gpu && weblas) {
      this._useWeblas = true
      this._pipelineEnabled = false
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * W weight tensor is converted to `channels_last` mode if in `channels_first` mode.
   * In `channels_last` mode, W weight tensor has shape [kernelDim1, kernelDim2, kernelDim3, inputChannels, nbFilter]
   * In `channels_first` mode, W weight tensor has shape [nbFilter, inputChannels, kernelDim1, kernelDim2, kernelDim3]
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    if (this.dataFormat === 'channels_first') {
      // W
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 4, 1, 0)
    }
    super.setWeights(weightsArr)

    this._wRowsMat = this._w2row()
    if (this._useWeblas) {
      this._wRowsMat.createWeblasTensor()
      if (!this._wRowsMat._gpuMaxSizeExceeded) {
        this._wRowsMat.weblasTensor = this._wRowsMat.weblasTensor.transpose()
      }
      if (this.use_bias) {
        this.weights['bias'].createWeblasTensor()
      } else {
        this._zerosVec = new Tensor([], [this.weights['kernel'].tensor.shape[4]])
        this._zerosVec.createWeblasTensor()
      }
    }
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * @param {Tensor} x
   */
  _calcOutputShape(x) {
    const inputDim1 = x.tensor.shape[0]
    const inputDim2 = x.tensor.shape[1]
    const inputDim3 = x.tensor.shape[2]
    const [nbFilter, kernelDim1, kernelDim2, kernelDim3] = this.kernelShape

    // effective shape after filter dilation
    const kernelDim1Dilated = kernelDim1 + (kernelDim1 - 1) * (this.dilationRate[0] - 1)
    const kernelDim2Dilated = kernelDim2 + (kernelDim2 - 1) * (this.dilationRate[1] - 1)
    const kernelDim3Dilated = kernelDim3 + (kernelDim3 - 1) * (this.dilationRate[2] - 1)

    const outputDim1 = this.padding === 'same'
      ? Math.floor((inputDim1 + this.strides[0] - 1) / this.strides[0])
      : Math.floor((inputDim1 - kernelDim1Dilated + this.strides[0]) / this.strides[0])
    const outputDim2 = this.padding === 'same'
      ? Math.floor((inputDim2 + this.strides[1] - 1) / this.strides[1])
      : Math.floor((inputDim2 - kernelDim2Dilated + this.strides[1]) / this.strides[1])
    const outputDim3 = this.padding === 'same'
      ? Math.floor((inputDim3 + this.strides[2] - 1) / this.strides[2])
      : Math.floor((inputDim3 - kernelDim3Dilated + this.strides[2]) / this.strides[2])
    const outputChannels = nbFilter

    const paddingDim1 = this.padding === 'same'
      ? Math.max(0, Math.floor((outputDim1 - 1) * this.strides[0] + kernelDim1Dilated - inputDim1))
      : 0
    const paddingDim2 = this.padding === 'same'
      ? Math.max(0, Math.floor((outputDim2 - 1) * this.strides[1] + kernelDim2Dilated - inputDim2))
      : 0
    const paddingDim3 = this.padding === 'same'
      ? Math.max(0, Math.floor((outputDim3 - 1) * this.strides[2] + kernelDim3Dilated - inputDim3))
      : 0
    const paddingDim1Before = Math.floor(paddingDim1 / 2)
    const paddingDim1After = paddingDim1 - paddingDim1Before
    const paddingDim2Before = Math.floor(paddingDim2 / 2)
    const paddingDim2After = paddingDim2 - paddingDim2Before
    const paddingDim3Before = Math.floor(paddingDim3 / 2)
    const paddingDim3After = paddingDim3 - paddingDim3Before

    this.outputShape = [outputDim1, outputDim2, outputDim3, outputChannels]
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
   * Pad input tensor if necessary, for padding='same'
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
   * Convert input volume to column matrix
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _vol2col(x) {
    const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape
    const kernelDim1 = this.kernelShape[1]
    const kernelDim2 = this.kernelShape[2]
    const kernelDim3 = this.kernelShape[3]
    const outputDim1 = this.outputShape[0]
    const outputDim2 = this.outputShape[1]
    const outputDim3 = this.outputShape[2]
    const nbPatches = outputDim1 * outputDim2 * outputDim3
    const patchLen = kernelDim1 * kernelDim2 * kernelDim3 * inputChannels

    // effective shape after filter dilation
    const kernelDim1Dilated = kernelDim1 + (kernelDim1 - 1) * (this.dilationRate[0] - 1)
    const kernelDim2Dilated = kernelDim2 + (kernelDim2 - 1) * (this.dilationRate[1] - 1)
    const kernelDim3Dilated = kernelDim3 + (kernelDim3 - 1) * (this.dilationRate[2] - 1)

    if (!this._volColsMat) {
      this._volColsMat = new Tensor([], [nbPatches, patchLen])
    }

    if (
      kernelDim1Dilated === 1 &&
      kernelDim2Dilated === 1 &&
      kernelDim3Dilated === 1 &&
      this.strides[0] === 1 &&
      this.strides[1] === 1 &&
      this.strides[2] === 1
    ) {
      this._volColsMat.replaceTensorData(x.tensor.data)
      if (this._useWeblas) {
        this._volColsMat.createWeblasTensor()
      }
      return this._volColsMat
    }

    let patch = new Tensor([], [kernelDim1, kernelDim2, kernelDim3, inputChannels])
    let offset = 0
    for (let i = 0, limit = inputDim1 - kernelDim1Dilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputDim2 - kernelDim2Dilated; j <= limit; j += this.strides[1]) {
        for (let k = 0, limit = inputDim3 - kernelDim3Dilated; k <= limit; k += this.strides[2]) {
          ops.assign(
            patch.tensor,
            x.tensor
              .hi(i + kernelDim1Dilated, j + kernelDim2Dilated, k + kernelDim3Dilated, inputChannels)
              .lo(i, j, k, 0)
              .step(this.dilationRate[0], this.dilationRate[1], this.dilationRate[2], 1)
          )
          this._volColsMat.tensor.data.set(patch.tensor.data, offset)
          offset += patchLen
        }
      }
    }
    if (this._useWeblas) {
      this._volColsMat.createWeblasTensor()
    }
    return this._volColsMat
  }

  /**
   * Convert filter weights to row matrix
   * @returns {Tensor|weblas.pipeline.Tensor} wRowsMat
   */
  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[3]
    const [nbFilter, kernelDim1, kernelDim2, kernelDim3] = this.kernelShape
    const patchLen = kernelDim1 * kernelDim2 * kernelDim3 * inputChannels

    const wRowsMat = new Tensor([], [patchLen, nbFilter])

    let patch = new Tensor([], [kernelDim1, kernelDim2, kernelDim3, inputChannels])
    let patchRaveled = new Tensor([], [patchLen])
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(patch.tensor, this.weights['kernel'].tensor.pick(null, null, null, null, n))
      patchRaveled.replaceTensorData(patch.tensor.data)
      ops.assign(wRowsMat.tensor.pick(null, n), patchRaveled.tensor)
    }

    return wRowsMat
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0)
    }

    this._calcOutputShape(x)
    this._padInput(x)

    this._vol2col(x)

    const nbFilter = this.kernelShape[0]
    const outputDim1 = this.outputShape[0]
    const outputDim2 = this.outputShape[1]
    const outputDim3 = this.outputShape[2]
    const nbPatches = outputDim1 * outputDim2 * outputDim3
    const matMul = new Tensor([], [nbPatches, nbFilter])

    if (this._useWeblas && !(this._volColsMat._gpuMaxSizeExceeded || this._wRowsMat._gpuMaxSizeExceeded)) {
      const bias = this.use_bias ? this.weights['bias'].weblasTensor : this._zerosVec.weblasTensor
      matMul.tensor.data = weblas.pipeline
        .sgemm(1, this._volColsMat.weblasTensor, this._wRowsMat.weblasTensor, 1, bias)
        .transfer()
    } else {
      if (this.use_bias) {
        for (let n = 0; n < nbFilter; n++) {
          ops.assigns(matMul.tensor.pick(null, n), this.weights['bias'].tensor.get(n))
        }
      }
      gemm(matMul.tensor, this._volColsMat.tensor, this._wRowsMat.tensor, 1, 1)
    }

    let output = new Tensor([], this.outputShape)
    let outputChannelRaveled = new Tensor([], [outputDim1 * outputDim2 * outputDim3])
    let outputChannel = new Tensor([], [outputDim1, outputDim2, outputDim3])
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(outputChannelRaveled.tensor, matMul.tensor.pick(null, n))
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data)
      ops.assign(output.tensor.pick(null, null, null, n), outputChannel.tensor)
    }
    x.tensor = output.tensor

    this.activationFunc(x)

    // convert back to th ordering if necessary
    if (this.dataFormat === 'th') {
      x.tensor = x.tensor.transpose(3, 0, 1, 2)
    }

    return x
  }
}

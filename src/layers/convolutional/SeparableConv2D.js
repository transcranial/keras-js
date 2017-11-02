import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../activations'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'
import gemm from 'ndarray-gemm'
import Conv2D from './Conv2D'

/**
 * _DepthwiseConv2D layer class
 */
class _DepthwiseConv2D extends Conv2D {
  constructor(attrs = {}) {
    super(attrs)
  }

  _calcOutputShape(inputShape) {
    super._calcOutputShape(inputShape)
    const nbFilter = this.kernelShape[0]
    const inputChannels = inputShape[2]
    this.outputShape[2] = nbFilter * inputChannels
  }

  _im2col(x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    const nbRow = this.kernelShape[1]
    const nbCol = this.kernelShape[2]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const patchLen = nbRow * nbCol

    if (!this._imColsMat) {
      this._imColsMat = new Tensor([], [nbPatches * inputChannels, patchLen])
    }

    let patch = new Tensor([], [nbRow, nbCol, 1])
    let offset = 0
    for (let c = 0; c < inputChannels; c++) {
      for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.strides[0]) {
        for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.strides[1]) {
          ops.assign(patch.tensor, x.tensor.hi(i + nbRow, j + nbCol, c + 1).lo(i, j, c))
          this._imColsMat.tensor.data.set(patch.tensor.data, offset)
          offset += patchLen
        }
      }
    }

    if (this.gpu) {
      this._imColsMat.createGLTexture()
    }
    return this._imColsMat
  }

  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[2]
    const [nbFilter, nbRow, nbCol] = this.kernelShape
    const patchLen = nbRow * nbCol

    this._wRowsMat = new Tensor([], [patchLen, nbFilter * inputChannels])

    let patch = new Tensor([], [nbRow, nbCol])
    let patchRaveled = new Tensor([], [patchLen])
    let p = 0
    for (let c = 0; c < inputChannels; c++) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assign(patch.tensor, this.weights['kernel'].tensor.pick(null, null, c, n))
        patchRaveled.replaceTensorData(patch.tensor.data)
        ops.assign(this._wRowsMat.tensor.pick(null, p), patchRaveled.tensor)
        p += 1
      }
    }

    return this._wRowsMat
  }

  _call_cpu(x) {
    this.inputShape = x.tensor.shape
    this._calcOutputShape(this.inputShape)
    this._padInput(x)
    this._im2col(x)

    const nbFilter = this.kernelShape[0]
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]
    const nbPatches = outputRows * outputCols
    const inputChannels = this.inputShape[2]
    const matMul = new Tensor([], [nbPatches * inputChannels, nbFilter * inputChannels])

    gemm(matMul.tensor, this._imColsMat.tensor, this._wRowsMat.tensor, 1, 1)

    this.output = new Tensor([], this.outputShape)

    const outputDataLength = outputRows * outputCols * nbFilter * inputChannels
    let dataFiltered = new Float32Array(outputDataLength)
    for (let c = 0; c < inputChannels; c++) {
      for (let n = c * outputDataLength + c * nbFilter; n < (c + 1) * outputDataLength; n += nbFilter * inputChannels) {
        for (let m = 0; m < nbFilter; m++) {
          dataFiltered[n + m - c * outputDataLength] = matMul.tensor.data[n + m]
        }
      }
    }
    this.output.replaceTensorData(dataFiltered)
  }

  _createOutputReshapeMap() {
    if (this.reshapeRowIndexMap && this.reshapeColIndexMap) {
      return
    }

    const nbFilter = this.kernelShape[0]
    const reshape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]]
    this.reshapeRowIndexMap = new Tensor([], reshape, { type: Int32Array })
    this.reshapeColIndexMap = new Tensor([], reshape, { type: Int32Array })
    for (let j = 0; j < reshape[1]; j++) {
      for (let i = 0; i < reshape[0]; i++) {
        ops.assigns(this.reshapeRowIndexMap.tensor.pick(i, j), i + Math.floor(j / nbFilter) * reshape[0])
      }
    }
    for (let j = 0; j < reshape[1]; j++) {
      ops.assigns(this.reshapeColIndexMap.tensor.pick(null, j), j)
    }

    if (this.gpu) {
      this.reshapeRowIndexMap.createGLTexture('2d', 'int')
      this.reshapeColIndexMap.createGLTexture('2d', 'int')
    }
  }

  _call_gpu(x) {
    super._call_gpu(x)

    this._createOutputReshapeMap()
    if (!this.outputReshaped) {
      const reshape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]]
      this.outputReshaped = new Tensor([], reshape)
      this.outputReshaped.createGLTexture()
      this.outputReshaped.glTextureIsTiled = true
      this.outputReshaped.untiledShape = this.outputShape
    }

    webgl2.selectProgram(this.mapInputProgram)
    webgl2.bindOutputTexture(this.outputReshaped.glTexture, this.outputReshaped.glTextureShape)
    let textures = [this.output.glTexture, this.reshapeRowIndexMap.glTexture, this.reshapeColIndexMap.glTexture]
    let textureTypes = ['2d', '2d', '2d']
    let textureNames = ['x', 'rowIndexMap', 'colIndexMap']
    webgl2.bindInputTextures(this.mapInputProgram, textures, textureTypes, textureNames)
    webgl2.runProgram()
  }
}

/**
 * SeparableConv2D layer class
 */
export default class SeparableConv2D extends Layer {
  /**
   * Creates a SeparableConv2D layer
   * @param {Number} attrs.filters - Number of convolution filters to use.
   * @param {Array<Number>|Number} attrs.kernel_size - Size of the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'SeparableConv2D'

    const {
      filters = 1,
      kernel_size = [1, 1],
      strides = [1, 1],
      padding = 'valid',
      data_format = 'channels_last',
      depth_multiplier = 1,
      activation = 'linear',
      use_bias = true
    } = attrs

    if (Array.isArray(kernel_size)) {
      this.kernelShape = [filters, ...kernel_size]
    } else {
      this.kernelShape = [filters, kernel_size, kernel_size]
    }

    if (Array.isArray(strides)) {
      this.strides = strides
    } else {
      this.strides = [strides, strides]
    }

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding
    } else {
      throw new Error(`${this.name} [Conv2D layer] Invalid padding.`)
    }

    if (data_format === 'channels_last' || data_format === 'channels_first') {
      this.dataFormat = data_format
    } else {
      throw new Error(`${this.name} [Conv2D layer] Only channels_last and channels_first data formats are allowed.`)
    }

    this.activation = activation
    this.activationFunc = activations[activation]

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding
    } else {
      throw new Error(`${this.name} [SeparableConv2D layer] Invalid padding.`)
    }

    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias
      ? ['depthwise_kernel', 'pointwise_kernel', 'bias']
      : ['depthwise_kernel', 'pointwise_kernel']

    // SeparableConv2D has two components: depthwise, and pointwise.
    // Activation function and bias is applied at the end.
    // Subsampling (striding) only performed on depthwise part, not the pointwise part.
    this.depthwiseConvAttrs = {
      filters: depth_multiplier,
      kernel_size: [this.kernelShape[1], this.kernelShape[2]],
      strides: this.strides,
      padding,
      data_format,
      activation: 'linear',
      use_bias: false,
      gpu: attrs.gpu
    }
    this.pointwiseConvAttrs = {
      filters,
      kernel_size: [1, 1],
      strides: [1, 1],
      padding,
      data_format,
      activation: 'linear',
      use_bias,
      gpu: attrs.gpu
    }

    // GPU setup
    if (this.gpu) {
      this.activationProgram = webgl2.compileProgram(require(`../../activations/${this.activation}.webgl2.glsl`))
    }
  }

  /**
   * Method for setting layer weights
   * Override `super` method since weights must be set in component Conv2D layers
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this._depthwiseConv = new _DepthwiseConv2D(this.depthwiseConvAttrs)
    this._depthwiseConv.setWeights(weightsArr.slice(0, 1))
    this._pointwiseConv = new Conv2D(this.pointwiseConvAttrs)
    this._pointwiseConv.setWeights(weightsArr.slice(1, 3))
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._call_gpu(x)
    } else {
      this._call_cpu(x)
    }
    return this.output
  }

  /**
   * CPU call
   */
  _call_cpu(x) {
    this._depthwiseConv._call_cpu(x)
    this._pointwiseConv._call_cpu(this._depthwiseConv.output)
    this.output = this._pointwiseConv.output
    this.activationFunc(this.output)
  }

  /**
   * GPU call
   */
  _call_gpu(x) {
    // prevent GPU -> CPU data transfer by specifying non-empty outbound nodes array on these internal Conv2D layers
    this._depthwiseConv.outbound = [null]
    this._pointwiseConv.outbound = [null]

    this._depthwiseConv._call_gpu(x)
    this._pointwiseConv._call_gpu(this._depthwiseConv.outputReshaped)

    // Activation
    if (this.activation === 'linear') {
      this.output = this._pointwiseConv.output
    } else {
      if (!this.output) {
        this.output = new Tensor([], this._pointwiseConv.output.glTextureShape)
        this.output.createGLTexture()
        this.output.glTextureIsTiled = true
        this.output.untiledShape = this._pointwiseConv.output.untiledShape
      }
      this.outputPreactiv = this._pointwiseConv.output
      webgl2.selectProgram(this.activationProgram)
      webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
      const textures = [this.outputPreactiv.glTexture]
      const textureTypes = ['2d']
      const textureNames = ['x']
      webgl2.bindInputTextures(this.activationProgram, textures, textureTypes, textureNames)
      webgl2.runProgram()
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      this.output.reshapeTensorFromTiled()

      // convert back to channels_first ordering if necessary
      if (this.dataFormat === 'channels_first') {
        this.output.tensor = this.output.tensor.transpose(2, 0, 1)
      }
    }
  }
}

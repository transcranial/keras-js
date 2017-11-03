import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * _Pooling1D layer class
 */
export default class _Pooling1D extends Layer {
  /**
   * Creates a _Pooling1D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_Pooling1D'

    const { pool_size = 2, strides = null, padding = 'valid' } = attrs

    this.poolSize = pool_size
    this.strides = strides === null ? this.poolSize : strides
    this.padding = padding

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max'

    // GPU setup
    if (this.gpu) {
      this.poolingProgram = webgl2.compileProgram(require('./_Pooling.webgl2.glsl'))
    }
  }

  /**
   * Layer computational logic
   *
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
   *
   * @param {Tensor} x
   */
  _call_cpu(x) {
    const stepsNew =
      this.padding === 'valid'
        ? Math.floor((x.tensor.shape[0] - this.poolSize + this.strides) / this.strides)
        : Math.floor((x.tensor.shape[0] + this.strides - 1) / this.strides)

    this.output = new Tensor([], [stepsNew, x.tensor.shape[1]])
    const outputStep = new Tensor([], [x.tensor.shape[1]])

    // in padding same, start negative from beyond step 0
    let step =
      this.padding === 'valid'
        ? 0
        : Math.min(0, Math.ceil((x.tensor.shape[0] - (stepsNew - 1) * this.strides - this.poolSize) / 2))

    for (let i = 0; i < stepsNew; i++) {
      let _step = Math.max(0, step)
      let limit = this.poolSize + Math.min(0, step)
      ops.assign(outputStep.tensor, x.tensor.pick(_step, null))

      let count = 1
      for (let j = 1; j < limit; j++) {
        if (_step + j > x.tensor.shape[0] - 1) {
          break
        }
        if (this.poolingFunc === 'max') {
          ops.maxeq(outputStep.tensor, x.tensor.pick(_step + j, null))
        } else if (this.poolingFunc === 'average') {
          ops.addeq(outputStep.tensor, x.tensor.pick(_step + j, null))
        }
        count += 1
      }

      if (this.poolingFunc === 'average') {
        ops.divseq(outputStep.tensor, count)
      }

      ops.assign(this.output.tensor.pick(i, null), outputStep.tensor)
      step += this.strides
    }
  }

  /**
   * Pre-compute index map for GPU pooling function
   *
   * @param {number[]} inputShape
   */
  _createIndexMap(inputShape) {
    if (this.indexMap) {
      return
    }

    const stepsNew =
      this.padding === 'valid'
        ? Math.floor((inputShape[0] - this.poolSize + this.strides) / this.strides)
        : Math.floor((inputShape[0] + this.strides - 1) / this.strides)

    this.outputShape = [stepsNew, inputShape[1]]

    this.indexMap = new Tensor([], [stepsNew, this.poolSize], { type: Int32Array })
    ops.assigns(this.indexMap.tensor, -1)

    // in padding same, start negative from beyond step 0
    let step =
      this.padding === 'valid'
        ? 0
        : Math.min(0, Math.ceil((inputShape[0] - (stepsNew - 1) * this.strides - this.poolSize) / 2))

    for (let i = 0; i < stepsNew; i++) {
      let _step = Math.max(0, step)
      let limit = this.poolSize + Math.min(0, step)

      let inputIndex = _step
      this.indexMap.tensor.set(i, 0, inputIndex)
      for (let j = 1; j < limit; j++) {
        inputIndex = _step + j
        if (inputIndex <= inputShape[0] - 1) {
          this.indexMap.tensor.set(i, j, inputIndex)
        } else {
          break
        }
      }
      step += this.strides
    }

    if (this.gpu) {
      this.indexMap.createGLTexture('2d', 'int')
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _call_gpu(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }
    this.inputShape = x.tensor.shape
    this._createIndexMap(this.inputShape)

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], this.outputShape)
      this.output.createGLTexture()
    }

    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max'

    webgl2.selectProgram(this.poolingProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const uniforms = [...this.output.glTextureShape, this.poolSize, +isMaxPooling]
    const uniformTypes = ['int', 'int', 'int', 'bool']
    const uniformNames = ['rows', 'cols', 'poolSize', 'isMaxPooling']
    webgl2.bindUniforms(this.poolingProgram, uniforms, uniformTypes, uniformNames)
    const textures = [x.glTexture, this.indexMap.glTexture]
    const textureTypes = ['2d', '2d']
    const textureNames = ['x', 'indexMap']
    webgl2.bindInputTextures(this.poolingProgram, textures, textureTypes, textureNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
    }
  }
}

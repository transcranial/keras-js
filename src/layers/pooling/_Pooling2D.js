import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * _Pooling2D layer class
 */
export default class _Pooling2D extends Layer {
  /**
   * Creates a _Pooling2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_Pooling2D'

    const { pool_size = [2, 2], strides = null, padding = 'valid', data_format = 'channels_last' } = attrs

    if (Array.isArray(pool_size)) {
      this.poolSize = pool_size
    } else {
      this.poolSize = [pool_size, pool_size]
    }

    if (Array.isArray(strides)) {
      this.strides = strides
    } else if (strides !== null) {
      this.strides = [strides, strides]
    } else {
      this.strides = this.poolSize
    }

    this.padding = padding
    this.dataFormat = data_format

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
   * Method for computing output dimensions and padding, based on input dimensions, kernel size, and padding mode
   *
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc

   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    if (this.outputShape && this.inputPadding) {
      return
    }

    const [inputRows, inputCols, inputChannels] = inputShape
    const [nbRow, nbCol] = this.poolSize

    const outputRows =
      this.padding === 'same'
        ? Math.floor((inputRows + this.strides[0] - 1) / this.strides[0])
        : Math.floor((inputRows - nbRow + this.strides[0]) / this.strides[0])
    const outputCols =
      this.padding === 'same'
        ? Math.floor((inputCols + this.strides[1] - 1) / this.strides[1])
        : Math.floor((inputCols - nbCol + this.strides[1]) / this.strides[1])

    const paddingRow =
      this.padding === 'same' ? Math.max(0, Math.floor((outputRows - 1) * this.strides[0] + nbRow - inputRows)) : 0
    const paddingCol =
      this.padding === 'same' ? Math.max(0, Math.floor((outputCols - 1) * this.strides[1] + nbCol - inputCols)) : 0
    const paddingRowBefore = Math.floor(paddingRow / 2)
    const paddingRowAfter = paddingRow - paddingRowBefore
    const paddingColBefore = Math.floor(paddingCol / 2)
    const paddingColAfter = paddingCol - paddingColBefore

    this.outputShape = [outputRows, outputCols, inputChannels]
    this.inputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter]
  }

  /**
   * Pad input tensor if necessary, for padding='same'. See above for notes on calculating padding.
   * For max, we pad with -infinity.
   *
   * For average we pad with zero.
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _padInput(x) {
    if (this.padding === 'same') {
      const [inputRows, inputCols, inputChannels] = x.tensor.shape
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      const newRows = inputRows + paddingRowBefore + paddingRowAfter
      const newCols = inputCols + paddingColBefore + paddingColAfter

      let _x = new Tensor([], [newRows, newCols, inputChannels])
      if (this.poolingFunc === 'max') {
        ops.assigns(_x.tensor, Number.NEGATIVE_INFINITY)
      }

      ops.assign(
        _x.tensor
          .hi(inputRows + paddingRowBefore, inputCols + paddingColBefore, inputChannels)
          .lo(paddingRowBefore, paddingColBefore, 0),
        x.tensor
      )
      x.tensor = _x.tensor
    }
    return x
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _call_cpu(x) {
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    this._calcOutputShape(x.tensor.shape)
    this._padInput(x)

    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    const [nbRow, nbCol] = this.poolSize
    this.output = new Tensor([], this.outputShape)
    const patch = new Tensor([], [nbRow, nbCol, inputChannels])

    // keep track of padding since these values are not included in pooling
    // for max, we can ignore since padding values are set to -infinity
    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding

    for (let i = 0, _i = 0; i <= inputRows - nbRow; i += this.strides[0], _i++) {
      let nbRowInPadding = 0
      if (i < paddingRowBefore) {
        nbRowInPadding = paddingRowBefore - i
      } else if (i + nbRow > inputRows - paddingRowAfter) {
        nbRowInPadding = i + nbRow - (inputRows - paddingRowAfter)
      }

      for (let j = 0, _j = 0; j <= inputCols - nbCol; j += this.strides[1], _j++) {
        let nbColInPadding = 0
        if (j < paddingColBefore) {
          nbColInPadding = paddingColBefore - j
        } else if (j + nbCol > inputCols - paddingColAfter) {
          nbColInPadding = j + nbCol - (inputCols - paddingColAfter)
        }
        const nbCellsEffective = (nbRow - nbRowInPadding) * (nbCol - nbColInPadding)

        ops.assign(patch.tensor, x.tensor.hi(i + nbRow, j + nbCol, inputChannels).lo(i, j, 0))
        for (let c = 0; c < inputChannels; c++) {
          if (this.poolingFunc === 'max') {
            this.output.tensor.set(_i, _j, c, ops.sup(patch.tensor.pick(null, null, c)))
          } else if (this.poolingFunc === 'average') {
            this.output.tensor.set(_i, _j, c, ops.sum(patch.tensor.pick(null, null, c)) / nbCellsEffective)
          }
        }
      }
    }

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      this.output.tensor = this.output.tensor.transpose(2, 0, 1)
    }
  }

  /**
   * Convert input tensor to column matrix
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _im2col(x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape
    if (!this._tiledInput) {
      this._tiledInput = new Tensor([], [inputRows * inputCols, inputChannels])
    }

    const patch = new Tensor([], [inputRows, inputCols])
    const patchRaveled = new Tensor([], [inputRows * inputCols])
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(patch.tensor, x.tensor.pick(null, null, c))
      patchRaveled.replaceTensorData(patch.tensor.data)
      ops.assign(this._tiledInput.tensor.pick(null, c), patchRaveled.tensor)
    }

    if (this.gpu) {
      this._tiledInput.createGLTexture()
    }
    return this._tiledInput
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

    let inputRows = inputShape[0]
    let inputCols = inputShape[1]
    const rowIndices = new Tensor([], [inputRows, inputCols])
    let index = 0
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        rowIndices.tensor.set(i, j, index)
        index += 1
      }
    }

    // padding for border mode 'same'
    if (this.padding === 'same') {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding
      inputRows = inputRows + paddingRowBefore + paddingRowAfter
      inputCols = inputCols + paddingColBefore + paddingColAfter
      const _rowIndices = new Tensor([], [inputRows, inputCols])
      ops.assigns(_rowIndices.tensor, -1)
      ops.assign(
        _rowIndices.tensor
          .hi(inputShape[0] + paddingRowBefore, inputShape[1] + paddingColBefore)
          .lo(paddingRowBefore, paddingColBefore),
        rowIndices.tensor
      )
      rowIndices.tensor = _rowIndices.tensor
    }

    const [nbRow, nbCol] = this.poolSize
    const outputRows = this.outputShape[0]
    const outputCols = this.outputShape[1]

    this.indexMap = new Tensor([], [outputRows * outputCols, nbRow * nbCol], { type: Int32Array })

    let patchRow = new Tensor([], [nbRow, nbCol])
    let offset = 0
    for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.strides[1]) {
        ops.assign(patchRow.tensor, rowIndices.tensor.hi(i + nbRow, j + nbCol).lo(i, j))
        this.indexMap.tensor.data.set(patchRow.tensor.data, offset)
        offset += nbRow * nbCol
      }
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
    if (x.glTextureIsTiled) {
      this.inputShape = x.untiledShape
    } else {
      // convert to channels_last ordering
      if (this.dataFormat === 'channels_first') {
        x.tensor = x.tensor.transpose(1, 2, 0)
      }
      this.inputShape = x.tensor.shape
      this._im2col(x)
      x.glTexture = this._tiledInput.glTexture
      x.glTextureShape = this._tiledInput.glTextureShape
    }
    this._calcOutputShape(this.inputShape)
    this._createIndexMap(this.inputShape)

    // create output textures if doesn't already exist
    if (!this.output) {
      const [outputRows, outputCols, inputChannels] = this.outputShape
      const outputTextureShape = [outputRows * outputCols, inputChannels]
      this.output = new Tensor([], outputTextureShape)
      this.output.createGLTexture()
      this.output.glTextureIsTiled = true
      this.output.untiledShape = this.outputShape
    }

    const poolSize = this.poolSize[0] * this.poolSize[1]
    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max'

    webgl2.selectProgram(this.poolingProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const uniforms = [...this.output.glTextureShape, poolSize, +isMaxPooling]
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
      this.output.reshapeTensorFromTiled()

      // convert back to channels_first ordering if necessary
      if (this.dataFormat === 'channels_first') {
        this.output.tensor = this.output.tensor.transpose(2, 0, 1)
      }
    }
  }
}

import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * UpSampling2D layer class
 */
export default class UpSampling2D extends Layer {
  /**
   * Creates a UpSampling2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number|number[]} [attrs.size] - upsampling factor, int or tuple of int (length 2)
   * @param {string} [attrs.data_format] - either 'channels_last' or 'channels_first'
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'UpSampling2D'

    const { size = [2, 2], data_format = 'channels_last' } = attrs

    if (Array.isArray(size)) {
      this.size = size
    } else {
      this.size = [size, size]
    }

    this.dataFormat = data_format

    // GPU setup
    if (this.gpu) {
      this.mapInputProgram = webgl2.compileProgram(require('../../mapInput.glsl'))
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
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    // convert to channels_last ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 0)
    }

    this.inputShape = x.tensor.shape
    this.outputShape = [this.inputShape[0] * this.size[0], this.inputShape[1] * this.size[1], this.inputShape[2]]
    this.output = new Tensor([], this.outputShape)
    for (let i = 0; i < this.size[0]; i++) {
      for (let j = 0; j < this.size[1]; j++) {
        ops.assign(this.output.tensor.lo(i, j, 0).step(this.size[0], this.size[1], 1), x.tensor)
      }
    }

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(2, 0, 1)
      this.output.tensor = this.output.tensor.transpose(2, 0, 1)
    }
  }

  /**
   * Creates row/col index mappings to map input texture to output texture
   *
   * @param {Object} indicesForReshaped
   */
  _createIndexMap(indicesForReshaped) {
    if (this.rowIndexMap && this.colIndexMap) {
      return
    }

    const indicesRow = new Tensor(indicesForReshaped.row.data, indicesForReshaped.row.shape, { type: Int32Array })
    const indicesCol = new Tensor(indicesForReshaped.col.data, indicesForReshaped.col.shape, { type: Int32Array })

    this.rowIndexMap = new Tensor([], this.outputShape, { type: Int32Array })
    this.colIndexMap = new Tensor([], this.outputShape, { type: Int32Array })

    for (let i = 0; i < this.size[0]; i++) {
      for (let j = 0; j < this.size[1]; j++) {
        const sliceStart = this.dataFormat === 'channels_first' ? [0, i, j] : [i, j, 0]
        const step =
          this.dataFormat === 'channels_first' ? [1, this.size[0], this.size[1]] : [this.size[0], this.size[1], 1]
        ops.assign(this.rowIndexMap.tensor.lo(...sliceStart).step(...step), indicesRow.tensor)
        ops.assign(this.colIndexMap.tensor.lo(...sliceStart).step(...step), indicesCol.tensor)
      }
    }

    this.rowIndexMap.reshapeTo2DSquare()
    this.colIndexMap.reshapeTo2DSquare()
    this.rowIndexMap.createGLTexture('2d', 'int')
    this.colIndexMap.createGLTexture('2d', 'int')
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      x.reshapeTo2DSquare()
      x.createGLTexture()
    }
    this.inputShape = x.originalShape
    this.outputShape =
      this.dataFormat === 'channels_first'
        ? [this.inputShape[0], this.inputShape[1] * this.size[0], this.inputShape[2] * this.size[1]]
        : [this.inputShape[0] * this.size[0], this.inputShape[1] * this.size[1], this.inputShape[2]]

    this._createIndexMap(x.indicesForReshaped)

    if (!this.output) {
      this.output = new Tensor([], this.outputShape)
      this.output.reshapeTo2DSquare()
      this.output.createGLTexture()
    }

    webgl2.selectProgram(this.mapInputProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    let textures = [x.glTexture, this.rowIndexMap.glTexture, this.colIndexMap.glTexture]
    let textureTypes = ['2d', '2d', '2d']
    let textureNames = ['x', 'rowIndexMap', 'colIndexMap']
    webgl2.bindInputTextures(this.mapInputProgram, textures, textureTypes, textureNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      this.output.reshapeFrom2DSquare()
    }
  }
}

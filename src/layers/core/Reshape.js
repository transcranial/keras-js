import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import _ from 'lodash'
import ops from 'ndarray-ops'

/**
 * Reshape layer class
 * Note there is no concept of batch size in these layers (single-batch).
 */
export default class Reshape extends Layer {
  /**
   * Creates a Reshape layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number[]} [attrs.target_shape]
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Reshape'

    const { target_shape = [] } = attrs
    this.targetShape = target_shape

    // GPU setup
    if (this.gpu) {
      this.mapInputProgram = webgl2.compileProgram(require('../../mapInput.webgl2.glsl'))
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
    if (this.targetShape.reduce((a, b) => a * b, 1) !== x.tensor.size) {
      throw new Error(`${this.name} [Reshape layer] The total size of new array must be unchanged in reshape layer.`)
    }
    this.output = new Tensor([], this.targetShape)
    this.output.replaceTensorData(x.tensor.data)
  }

  /**
   * Creates row/col index mappings to map input texture to output texture
   *
   * @param {number[]} inputShape
   */
  _createIndexMap(inputShape) {
    if (this.rowIndexMap && this.colIndexMap) {
      return
    }

    let indicesRow = new Tensor([], inputShape)
    let indicesCol = new Tensor([], inputShape)

    if (inputShape.length === 2) {
      for (let i = 0; i < inputShape[0]; i++) {
        ops.assigns(indicesRow.tensor.pick(i, null), i)
      }
    } else if (inputShape.length === 3) {
      for (let i = 0; i < inputShape[0]; i++) {
        for (let j = 0; j < inputShape[1]; j++) {
          ops.assigns(indicesRow.tensor.pick(i, j, null), i * inputShape[1] + j)
        }
      }
    } else if (inputShape.length === 4) {
      for (let i = 0; i < inputShape[0]; i++) {
        for (let j = 0; j < inputShape[1]; j++) {
          for (let k = 0; k < inputShape[2]; k++) {
            ops.assigns(
              indicesRow.tensor.pick(i, j, k, null),
              i * inputShape[1] * inputShape[2] + j * inputShape[2] + k
            )
          }
        }
      }
    }
    for (let c = 0; c < _.last(inputShape); c++) {
      ops.assigns(indicesCol.tensor.pick(...Array(inputShape.length - 1).fill(null), c), c)
    }

    this.rowIndexMap = new Tensor([], this.targetShape, { type: Int32Array })
    this.colIndexMap = new Tensor([], this.targetShape, { type: Int32Array })
    this.rowIndexMap.replaceTensorData(new Int32Array(indicesRow.tensor.data))
    this.colIndexMap.replaceTensorData(new Int32Array(indicesCol.tensor.data))
    if (this.targetShape.length > 2) {
      this.rowIndexMap.reshapeTensorToTiled()
      this.colIndexMap.reshapeTensorToTiled()
    }

    if (this.gpu) {
      this.rowIndexMap.createGLTexture('2d', 'int')
      this.colIndexMap.createGLTexture('2d', 'int')
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      this.inputShape = x.tensor.shape
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture()
      } else if (x.tensor.shape.length > 2 && !x.glTextureIsTiled) {
        x.reshapeTensorToTiled()
        x.createGLTexture()
      }
    } else if (x.glTextureIsTiled) {
      this.inputShape = x.untiledShape
    } else {
      this.inputShape = x.tensor.shape
    }
    this._createIndexMap(this.inputShape)

    if (!this.output) {
      this.output = new Tensor([], this.targetShape)
      if (this.targetShape.length > 2) {
        this.output.reshapeTensorToTiled()
      }
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
      if (this.output.glTextureIsTiled) {
        this.output.reshapeTensorFromTiled(this.axis)
      }
    }
  }
}

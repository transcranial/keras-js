import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import _ from 'lodash'
import ops from 'ndarray-ops'

/**
 * Permute layer class
 * Note there is no concept of batch size in these layers (single-batch), so dim numbers 1 less
 * i.e., dim 1 in keras corresponds to dim 0 here, etc.
 */
export default class Permute extends Layer {
  /**
   * Creates a Permute layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number[]} [attrs.dims]
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Permute'

    const { dims = [] } = attrs
    this.dims = dims.map(dim => dim - 1)

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
    if (x.tensor.shape.length <= 1 || _.isEqual(_.range(x.tensor.shape.length), this.dims)) {
      this.output = x
      return this.output
    }

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
    if (this.dims.length !== x.tensor.shape.length) {
      throw new Error(
        `${this.name} [Permute layer] The specified dims permutation must match the number of dimensions.`
      )
    }

    const outputShape = this.dims.map(i => x.tensor.shape[i])
    this.output = new Tensor([], outputShape)
    ops.assign(this.output.tensor, x.tensor.transpose(...this.dims))
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

    const outputShape = this.dims.map(i => inputShape[i])
    this.rowIndexMap = new Tensor([], outputShape, { type: Int32Array })
    this.colIndexMap = new Tensor([], outputShape, { type: Int32Array })
    ops.assign(this.rowIndexMap.tensor, indicesRow.tensor.transpose(...this.dims))
    ops.assign(this.colIndexMap.tensor, indicesCol.tensor.transpose(...this.dims))
    if (outputShape.length > 2) {
      this.rowIndexMap.reshapeTo2D()
      this.colIndexMap.reshapeTo2D()
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
      } else if (x.tensor.shape.length > 2 && !x.is2DReshaped) {
        x.reshapeTo2D()
        x.createGLTexture()
      }
    } else if (x.is2DReshaped) {
      this.inputShape = x.originalShape
    } else {
      this.inputShape = x.tensor.shape
    }
    this._createIndexMap(this.inputShape)

    if (!this.output) {
      const outputShape = this.dims.map(i => this.inputShape[i])
      this.output = new Tensor([], outputShape)
      if (outputShape.length > 2) {
        this.output.reshapeTo2D()
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
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D(this.axis)
      }
    }
  }
}

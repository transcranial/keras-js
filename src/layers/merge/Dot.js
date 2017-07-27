import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import gemm from 'ndarray-gemm'
import ops from 'ndarray-ops'

/**
 * Dot merge layer class, extends abstract _Merge class
 */
export default class Dot extends _Merge {
  /**
   * Creates a Dot merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Dot'

    this.mode = 'dot'

    const { axes = -1, normalize = false } = attrs

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    if (Array.isArray(axes)) {
      this.dotAxes = [axes[0] <= 0 ? axes[0] : axes[0] - 1, axes[1] <= 0 ? axes[1] : axes[1] - 1]
    } else {
      this.dotAxes = [axes <= 0 ? axes : axes - 1, axes <= 0 ? axes : axes - 1]
    }

    this.normalize = normalize

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(require('./Dot.webgl2.glsl'))
    }
  }

  /**
   * Calculate output shape
   * @param {Number[][]} inputShapes
   */
  _calcOutputShape(inputShapes) {
    let shape1 = inputShapes[0].slice()
    let shape2 = inputShapes[1].slice()
    shape1.splice(this.dotAxes[0], 1)
    shape2.splice(this.dotAxes[1], 1)
    this.outputShape = shape1.concat(shape2)
    if (this.outputShape.length === 1) {
      this.outputShape.push(1)
    }
  }

  /**
   * CPU call
   * @param {Tensor[]} inputs
   */
  _call_cpu(inputs) {
    this._calcOutputShape([inputs[0].tensor.shape, inputs[1].tensor.shape])
    this.output = new Tensor([], this.outputShape)

    if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
      if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
        if (this.normalize) {
          for (let i = 0; i < inputs[0].tensor.shape[1]; i++) {
            ops.divseq(inputs[0].tensor.pick(null, i), ops.norm2(inputs[0].tensor.pick(null, i)))
          }
          for (let i = 0; i < inputs[1].tensor.shape[1]; i++) {
            ops.divseq(inputs[1].tensor.pick(null, i), ops.norm2(inputs[1].tensor.pick(null, i)))
          }
        }
        gemm(this.output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor)
      } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
        if (this.normalize) {
          for (let i = 0; i < inputs[0].tensor.shape[0]; i++) {
            ops.divseq(inputs[0].tensor.pick(i, null), ops.norm2(inputs[0].tensor.pick(i, null)))
          }
          for (let i = 0; i < inputs[1].tensor.shape[0]; i++) {
            ops.divseq(inputs[1].tensor.pick(i, null), ops.norm2(inputs[1].tensor.pick(i, null)))
          }
        }
        gemm(this.output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0))
      }
    } else {
      throw new Error(`${this.name} [${this.layerClass} layer] dot mode for 3+ dim tensors not yet implemented.`)
    }
  }

  /**
   * GPU call
   * @param {Tensor[]} inputs
   */
  _call_gpu(inputs) {
    this._calcOutputShape([inputs[0].glTextureShape, inputs[1].glTextureShape])

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], this.outputShape)
      this.output.createGLTexture()
    }

    const commonDim = inputs[0].glTextureShape[this.dotAxes[0]]

    webgl2.selectProgram(this.mergeProgram)
    webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
    const uniforms = [...this.output.glTextureShape, ...this.dotAxes, commonDim, +this.normalize]
    const uniformTypes = ['int', 'int', 'int', 'int', 'int', 'bool']
    const uniformNames = ['rows', 'cols', 'dotAxis1', 'dotAxis2', 'commonDim', 'normalize']
    webgl2.bindUniforms(this.mergeProgram, uniforms, uniformTypes, uniformNames)
    const textures = [inputs[0].glTexture, inputs[1].glTexture]
    const textureTypes = ['2d', '2d']
    const textureNames = ['input1', 'input2']
    webgl2.bindInputTextures(this.mergeProgram, textures, textureTypes, textureNames)
    webgl2.runProgram()

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
    }
  }
}

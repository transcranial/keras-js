import _Merge from './_Merge'
import Tensor from '../../Tensor'
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
  }

  /**
   * CPU call
   * @param {Tensor[]} inputs
   */
  _call_cpu(inputs) {
    let shape1 = inputs[0].tensor.shape.slice()
    let shape2 = inputs[1].tensor.shape.slice()
    shape1.splice(this.dotAxes[0], 1)
    shape2.splice(this.dotAxes[1], 1)
    const outputShape = shape1.concat(shape2)
    if (outputShape.length === 1) {
      outputShape.push(1)
    }
    this.output = new Tensor([], outputShape)

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
  _call_gpu(inputs) {}
}

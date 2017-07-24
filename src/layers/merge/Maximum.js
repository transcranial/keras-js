import _Merge from './_Merge'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * Maximum merge layer class, extends abstract _Merge class
 */
export default class Maximum extends _Merge {
  /**
   * Creates a Maximum merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Maximum'

    this.mode = 'max'
  }

  /**
   * CPU call
   * @param {Tensor[]} inputs
   */
  _call_cpu(inputs) {
    const outputShape = inputs[0].tensor.shape.slice()
    this.output = new Tensor([], outputShape)

    ops.assign(this.output.tensor, inputs[0].tensor)
    for (let i = 1; i < inputs.length; i++) {
      ops.maxeq(this.output.tensor, inputs[i].tensor)
    }
  }

  /**
   * GPU call
   * @param {Tensor[]} inputs
   */
  _call_gpu(inputs) {}
}

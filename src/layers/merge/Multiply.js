import _Merge from './_Merge'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * Multiply merge layer class, extends abstract _Merge class
 */
export default class Multiply extends _Merge {
  /**
   * Creates a Multiply merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Multiply'

    this.mode = 'mul'
  }

  /**
   * CPU call
   * @param {Tensor[]} inputs
   */
  _call_cpu(inputs) {
    const outputShape = inputs[0].tensor.shape.slice()
    this.output = new Tensor([], outputShape)

    ops.assigns(this.output.tensor, 1)
    for (let i = 0; i < inputs.length; i++) {
      ops.muleq(this.output.tensor, inputs[i].tensor)
    }
  }

  /**
   * GPU call
   * @param {Tensor[]} inputs
   */
  _call_gpu(inputs) {}
}

import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * Multiply merge layer class, extends abstract _Merge class
 */
export default class Multiply extends _Merge {
  /**
   * Creates a Multiply merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Multiply'

    this.mode = 'mul'

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(require('./Multiply.webgl2.glsl'))
    }
  }

  /**
   * CPU call
   *
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
}

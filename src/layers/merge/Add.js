import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * Add merge layer class, extends abstract _Merge class
 */
export default class Add extends _Merge {
  /**
   * Creates a Add merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Add'

    this.mode = 'sum'

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(require('./Add.webgl2.glsl'))
    }
  }

  /**
   * CPU call
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {
    const outputShape = inputs[0].tensor.shape.slice()
    this.output = new Tensor([], outputShape)

    for (let i = 0; i < inputs.length; i++) {
      ops.addeq(this.output.tensor, inputs[i].tensor)
    }
  }
}

import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * Maximum merge layer class, extends abstract _Merge class
 */
export default class Maximum extends _Merge {
  /**
   * Creates a Maximum merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Maximum'

    this.mode = 'max'

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(require('./Maximum.glsl'))
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

    ops.assign(this.output.tensor, inputs[0].tensor)
    for (let i = 1; i < inputs.length; i++) {
      ops.maxeq(this.output.tensor, inputs[i].tensor)
    }
  }
}

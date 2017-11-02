import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'

/**
 * Minimum merge layer class, extends abstract _Merge class
 */
export default class Minimum extends _Merge {
  /**
   * Creates a Minimum merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Minimum'

    this.mode = 'min'

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(require('./Minimum.webgl2.glsl'))
    }
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
      ops.mineq(this.output.tensor, inputs[i].tensor)
    }
  }
}

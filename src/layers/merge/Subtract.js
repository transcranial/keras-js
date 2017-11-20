import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import ops from 'ndarray-ops'
import mergeProgramSource from './Subtract.glsl'

/**
 * Subtract merge layer class, extends abstract _Merge class
 */
export default class Subtract extends _Merge {
  /**
   * Creates a Subtract merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Subtract'

    this.mode = 'diff'

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(mergeProgramSource)
    }
  }

  /**
   * CPU call
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {
    if (inputs.length !== 2) {
      this.throwError('Inputs should be an array of 2 Tensors.')
    }

    const outputShape = inputs[0].tensor.shape.slice()
    this.output = new Tensor([], outputShape)

    ops.sub(this.output.tensor, inputs[0].tensor, inputs[1].tensor)
  }
}

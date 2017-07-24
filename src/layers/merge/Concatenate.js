import _Merge from './_Merge'
import Tensor from '../../Tensor'
import concatFirstAxis from 'ndarray-concat-rows'

/**
 * Concatenate merge layer class, extends abstract _Merge class
 */
export default class Concatenate extends _Merge {
  /**
   * Creates a Concatenate merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Concatenate'

    this.mode = 'concat'

    const { axis = -1 } = attrs

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = axis <= 0 ? axis : axis - 1
  }

  /**
   * CPU call
   * @param {Tensor[]} inputs
   */
  _call_cpu(inputs) {
    const outputShape = inputs[0].tensor.shape.slice()
    let _concatAxis = this.concatAxis < 0 ? outputShape.length + this.concatAxis : this.concatAxis
    if (this.concatAxis === 0) _concatAxis = 0
    inputs.slice(1, inputs.length).forEach(x => {
      const d = x.tensor.shape.slice()[_concatAxis]
      outputShape[_concatAxis] += d
    })
    this.output = new Tensor([], outputShape)

    if (_concatAxis === 0) {
      concatFirstAxis(this.output.tensor, inputs.map(x => x.tensor))
    } else {
      let dimsAxisSwap = [_concatAxis]
      for (let i = 0; i < inputs[0].tensor.shape.length; i++) {
        if (i !== _concatAxis) dimsAxisSwap.push(i)
      }
      concatFirstAxis(
        this.output.tensor.transpose(...dimsAxisSwap),
        inputs.map(x => x.tensor.transpose(...dimsAxisSwap))
      )
    }
  }

  /**
   * GPU call
   * @param {Tensor[]} inputs
   */
  _call_gpu(inputs) {}
}

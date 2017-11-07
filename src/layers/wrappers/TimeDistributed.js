import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * TimeDistributed wrapper layer class
 */
export default class TimeDistributed extends Layer {
  /**
   * Creates a TimeDistributed wrapper layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'TimeDistributed'

    const { layer } = attrs

    if (!layer) this.throwError('wrapped layer is undefined.')
    this.layer = layer
  }

  /**
   * Method for setting layer weights
   * Passes weights to the wrapped layer
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this.layer.setWeights(weightsArr)
  }

  /**
   * Method for layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    const xStepShape = [...x.tensor.shape.slice(1)]
    let xStep = new Tensor([], xStepShape)
    ops.assign(xStep.tensor, x.tensor.pick(0, ...xStepShape.map(s => null)))
    let yStep = this.layer.call(xStep)
    const yStepShape = yStep.tensor.shape.slice()
    let y = new Tensor([], [x.tensor.shape[0], ...yStepShape])
    ops.assign(y.tensor.pick(0, ...yStepShape.map(s => null)), yStep.tensor)

    for (let i = 1, steps = x.tensor.shape[0]; i < steps; i++) {
      let xStep = new Tensor([], xStepShape)
      ops.assign(xStep.tensor, x.tensor.pick(i, ...xStepShape.map(s => null)))
      yStep = this.layer.call(xStep)
      ops.assign(y.tensor.pick(i, ...yStepShape.map(s => null)), yStep.tensor)
    }

    x.tensor = y.tensor
    return x
  }
}

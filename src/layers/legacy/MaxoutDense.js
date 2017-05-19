import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'

/**
 * MaxoutDense layer class
 * From Keras docs: takes the element-wise maximum of nb_feature Dense(input_dim, output_dim) linear layers
 * Note that `nb_feature` is implicit in the weights tensors, with shapes:
 * - W: [nb_feature, input_dim, output_dim]
 * - b: [nb_feature, output_dim]
 */
export default class MaxoutDense extends Layer {
  /**
   * Creates a MaxoutDense layer
   * @param {number} attrs.output_dim - output dimension size
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'MaxoutDense'

    const { output_dim = 1, input_dim = null, bias = true } = attrs
    this.outputDim = output_dim
    this.inputDim = input_dim
    this.bias = bias

    // Layer weights specification
    this.params = this.bias ? ['W', 'b'] : ['W']
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    const nbFeature = this.weights.W.tensor.shape[0]

    let featMax = new Tensor([], [this.outputDim])
    for (let i = 0; i < nbFeature; i++) {
      let y = new Tensor([], [this.outputDim])
      if (this.bias) {
        ops.assign(y.tensor, this.weights.b.tensor.pick(i, null))
      }
      gemv(1.0, this.weights.W.tensor.pick(i, null, null).transpose(1, 0), x.tensor, 1.0, y.tensor)
      ops.maxeq(featMax.tensor, y.tensor)
    }

    x.tensor = featMax.tensor
    return x
  }
}

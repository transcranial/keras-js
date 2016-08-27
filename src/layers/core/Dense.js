import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../engine/Layer'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'

/**
 * Dense layer class
 */
export default class Dense extends Layer {
  /**
   * Creates a Dense layer
   * @param {number} outputDim - output dimension size
   * @param {Object} [attrs] - layer attributes
   */
  constructor (outputDim, attrs = {}) {
    super(attrs)
    const {
      activation = 'linear',
      inputDim = null,
      bias = true
    } = attrs

    this.activation = activations[activation]
    this.outputDim = outputDim
    this.inputDim = inputDim
    this.bias = bias

    // Layer weights specification
    this.params = this.bias ? ['W', 'b'] : ['W']

    // Input shape specification
    if (this.inputDim) {
      this.inputShape = [this.inputDim]
    }
  }

  /**
   * Method for layer computational logic
   *
   * x = W^T * x + b
   *
   * weblas notes:
   * sgemm(M, N, K, alpha, A, B, beta, C), where A, B, C are Float32Array
   * - alpha * A * B + beta * C
   * - A has shape M x N
   * - B has shape N x K
   * - C has shape M x K
   *
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call = x => {
    let y = new Tensor([], [this.outputDim])
    if (this.bias) {
      ops.assign(y.tensor, this.weights.b.tensor)
    }
    if (x._useWeblas) {
      const bias = this.bias
        ? this.weights.b.tensor.data
        : new Float32Array(this.outputDim)
      y.tensor.data = weblas.sgemm(
        1, this.weights.W.tensor.shape[1], x.tensor.shape[0], // M, N, K
        1, x.tensor.data, this.weights.W.tensor.data, // alpha, A, B
        1, bias // beta, C
      )
    } else {
      gemv(1.0, this.weights.W.tensor.transpose(1, 0), x.tensor, 1.0, y.tensor)
    }
    x.tensor = y.tensor

    this.activation(x)

    return x
  }
}

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
   * pipeline.sgemm(alpha, A, B, beta, C), where A, B, C are weblas.pipeline.Tensor here
   * - alpha * A * B^T + beta * C
   *
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call = x => {
    if (x._useWeblas) {
      // x is mutable, so create on every call
      x.createWeblasTensor()
      if (!this.weblasWeights) {
        // layer weights are immutable, so only create if not already existing
        this.createWeblasWeights()
      }

      const bias = this.bias
        ? this.weblasWeights.b
        : new weblas.pipeline.Tensor([1, this.outputDim], new Float32Array(this.outputDim))

      x.weblasTensor = weblas.pipeline.sgemm(
        1.0,
        x.weblasTensor,
        this.weblasWeights.W.transpose(true),
        1.0,
        bias
      )

      // activation function in CPU memory
      x.transferWeblasTensor()
    } else {
      let y = new Tensor([], [this.outputDim])
      if (this.bias) {
        ops.assign(y.tensor, this.weights.b.tensor)
      }
      gemv(1.0, this.weights.W.tensor.transpose(1, 0), x.tensor, 1.0, y.tensor)
      x.tensor = y.tensor
    }

    this.activation(x)

    return x
  }
}

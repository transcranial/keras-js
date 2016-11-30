import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'

/**
 * Dense layer class
 */
export default class Dense extends Layer {
  /**
   * Creates a Dense layer
   * @param {number} attrs.outputDim - output dimension size
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = 'Dense'

    const {
      outputDim = 1,
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
   * Method for setting layer weights. Extends `super` method.
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights (weightsArr) {
    super.setWeights(weightsArr)

    if (this._useWeblas) {
      this.weights.W.createWeblasTensor()
      if (!this.weights.W._gpuMaxSizeExceeded) {
        this.weights.W.weblasTensor = this.weights.W.weblasTensor.transpose()
      }
      if (this.bias) {
        this.weights.b.createWeblasTensor()
      } else {
        this._zerosVec = new Tensor([], [this.weights.W.tensor.shape[1]])
        this._zerosVec.createWeblasTensor()
      }
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
  call (x) {
    let y = new Tensor([], [this.outputDim])

    if (this._useWeblas) {
      x.createWeblasTensor()
    }

    if (this._useWeblas && !(x._gpuMaxSizeExceeded || this.weights.W._gpuMaxSizeExceeded)) {
      const bias = this.bias ? this.weights.b.weblasTensor : this._zerosVec.weblasTensor
      y.tensor.data = weblas.pipeline.sgemm(
        1, x.weblasTensor, this.weights.W.weblasTensor,
        1, bias
      ).transfer()
      x.weblasTensor.delete()
      delete x.weblasTensor
    } else {
      if (this.bias) {
        ops.assign(y.tensor, this.weights.b.tensor)
      }
      gemv(1.0, this.weights.W.tensor.transpose(1, 0), x.tensor, 1.0, y.tensor)
    }
    x.tensor = y.tensor

    this.activation(x)

    return x
  }
}

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
   * @param {number} attrs.units - output dimension size
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Dense'

    const { units = 1, activation = 'linear', input_dim = null, use_bias = true } = attrs

    this.activation = activation
    this.activationFunc = activations[activation]
    this.units = units
    this.inputDim = input_dim
    this.use_bias = use_bias

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel']

    // Input shape specification
    if (this.inputDim) {
      this.inputShape = [this.inputDim]
    }

    // Enable layer gpu +/- pipeline mode if supported
    if (this.gpu && weblas) {
      this._useWeblas = true
      this._pipelineEnabled = false
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)

    if (this._useWeblas) {
      this.weights['kernel'].createWeblasTensor()
      if (!this.weights['kernel']._gpuMaxSizeExceeded) {
        this.weights['kernel'].weblasTensor = this.weights['kernel'].weblasTensor.transpose()
      }
      if (this.use_bias) {
        this.weights['bias'].createWeblasTensor()
      } else {
        this._zerosVec = new Tensor([], [this.weights['kernel'].tensor.shape[1]])
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
  call(x) {
    let y = new Tensor([], [this.units])

    if (this._useWeblas) {
      x.createWeblasTensor()
    }

    if (this._useWeblas && !(x._gpuMaxSizeExceeded || this.weights['kernel']._gpuMaxSizeExceeded)) {
      const bias = this.use_bias ? this.weights['bias'].weblasTensor : this._zerosVec.weblasTensor
      y.tensor.data = weblas.pipeline.sgemm(1, x.weblasTensor, this.weights['kernel'].weblasTensor, 1, bias).transfer()
    } else {
      if (this.use_bias) {
        ops.assign(y.tensor, this.weights['bias'].tensor)
      }
      gemv(1, this.weights['kernel'].tensor.transpose(1, 0), x.tensor, 1, y.tensor)
    }
    x.tensor = y.tensor

    this.activationFunc(x)

    return x
  }
}

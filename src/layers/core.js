import * as activations from '../activations'
import { Layer } from '../engine/topology'

export class Dense extends Layer {
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

    /**
    * Layer weights specification
    */
    this.params = this.bias ? ['W', 'b'] : ['W']

    /**
    * Input shape specification
    */
    if (this.inputDim) {
      this.inputShape = [this.inputDim]
    }
  }

  /**
  * Method for layer computational logic
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
  * @returns {Tensor} `this`
  */
  call = x => {
    if (!x.weblasTensor) {
      x.createWeblasTensor()
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
    this.activation(x)

    return this
  }
}

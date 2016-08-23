import ndarray from 'ndarray'
import squeeze from 'ndarray-squeeze'

/**
* Layer class
*/
export class Layer {
  /**
  * Creates a layer
  * @param {Object} [attrs] - layer attributes
  */
  constructor (attrs = {}) {
    this.name = attrs.name
  }

  inboundNodes = []
  outboundNodes = []

  params = []
  weights = {}

  /**
  * Method for setting layer weights
  * We store the weights as both Tensor instances,
  * as well as weblas pipeline tensors if possible (which are in GPU memory)
  * see https://github.com/waylonflinn/weblas/wiki/Pipeline
  *
  * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
  */
  setWeights = weightsArr => {
    this.params.forEach((p, i) => {
      this.weights[p] = weightsArr[i]
    })
  }

  /**
  * Create weblas pipeline tensor weights
  * 2-D only
  */
  createWeblasWeights = () => {
    this.weblasWeights = {}

    this.params.forEach((p, i) => {
      if (this.weights[p].tensor.shape.length === 1) {
        const shape = [1, this.weights[p].tensor.shape[0]]
        this.weblasWeights[p] = new weblas.pipeline.Tensor(shape, this.weights[p].tensor.data)
      } if (this.weights[p].tensor.shape.length === 2) {
        const shape = this.weights[p].tensor.shape
        this.weblasWeights[p] = new weblas.pipeline.Tensor(shape, this.weights[p].tensor.data)
      }
    })
  }

  /**
  * Transfer weblas pipeline tensor weights
  */
  transferWeblasWeights = () => {
    this.params.forEach((p, i) => {
      if (this.weblasWeights[p]) {
        const shape = this.weblasWeights[p].shape
        const arr = this.weblasWeights[p].transfer(true)
        this.weights[p].tensor = squeeze(ndarray(arr, shape))
      }
    })
  }

  /**
  * Delete weblas pipeline tensor weights
  */
  deleteWeblasWeights = () => {
    this.params.forEach((p, i) => {
      if (this.weblasWeights[p]) {
        this.weblasWeights[p].delete()
        delete this.weblasWeights[p]
      }
    })
    delete this.weblasWeights
  }
}

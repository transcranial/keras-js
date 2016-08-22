import ndarray from 'ndarray'
import squeeze from 'ndarray-squeeze'

export class Layer {
  constructor (attrs = {}) {
    this.name = attrs.name
  }

  inboundNodes = []
  outboundNodes = []

  params = []
  weights = {}
  weblasWeights = {}

  /**
  * Method for setting layer weights
  * We store the weights as both Tensor instances,
  * as well as weblas pipeline tensors if possible (which are in GPU memory)
  * see https://github.com/waylonflinn/weblas/wiki/Pipeline
  *
  * @param {Tensor[]} weightsArr
  */
  setWeights = weightsArr => {
    this.params.forEach((p, i) => {
      this.weights[p] = weightsArr[i]
    })

    // create weblas pipeline tensor weights
    this.createWeblasWeights()
  }

  /**
  * Create weblas pipeline tensor weights
  * 2-D only
  */
  createWeblasWeights = () => {
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
  * Sync weblas pipeline tensor weights
  */
  syncWeblasWeights = () => {
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
  }
}

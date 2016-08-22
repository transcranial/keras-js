import ndarray from 'ndarray'
import squeeze from 'ndarray-squeeze'

export default class Tensor {
  constructor (data, shape, options = {}) {
    this._type = options.type || Float32Array
    const TypedArray = this._type

    if (shape.length && data.length !== shape.reduce((a, b) => a * b, 1)) {
      throw new Error('Specified shape incompatible with data.')
    }

    if (data && data.length && data instanceof TypedArray) {
      this.tensor = ndarray(data, shape)
    } else if (data && data.length && data instanceof Array) {
      this.tensor = ndarray(new TypedArray(data), shape)
    } else {
      this.tensor = ndarray(new TypedArray([]), [])
    }
  }

  /**
  * Reference to weblas pipeline tensor in GPU memory, if available
  * see https://github.com/waylonflinn/weblas/wiki/Pipeline
  */
  weblasTensor = null

  /**
  * Create weblas pipeline tensor
  * 2-D only
  */
  createWeblasTensor = () => {
    if (this.tensor.shape.length === 1) {
      const shape = [1, this.tensor.shape[0]]
      this.weblasTensor = new weblas.pipeline.Tensor(shape, this.tensor.data)
    } else if (this.tensor.shape.length === 2) {
      const shape = this.tensor.shape
      this.weblasTensor = new weblas.pipeline.Tensor(shape, this.tensor.data)
    }
  }

  /**
  * Transfers weblas pipeline tensor from GPU memory
  */
  transferWeblasTensor = () => {
    if (this.weblasTensor) {
      const shape = this.weblasTensor.shape
      const arr = this.weblasTensor.transfer(true)
      this.tensor = squeeze(ndarray(arr, shape))
    }
  }

  /**
  * Delete weblas pipeline tensor
  */
  deleteWeblasTensor = () => {
    if (this.weblasTensor) {
      this.weblasTensor.delete()
      this.weblasTensor = null
    }
  }

}

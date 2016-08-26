import ndarray from 'ndarray'
import squeeze from 'ndarray-squeeze'

const checkShape = (data, shape) => {
  if (data.length && shape.length && data.length !== shape.reduce((a, b) => a * b, 1)) {
    throw new Error('Specified shape incompatible with data.')
  }
}

/**
 * Tensor class
 */
export default class Tensor {
  /**
   * Creates a tensor
   * @param {(TypedArray|Array)} data
   * @param {Array} shape
   * @param {Object} [options]
   */
  constructor (data, shape, options = {}) {
    this._type = options.type || Float32Array

    if (data && data.length && data instanceof this._type) {
      checkShape(data, shape)
      this.tensor = ndarray(data, shape)
    } else if (data && data.length && data instanceof Array) {
      checkShape(data, shape)
      this.tensor = ndarray(new this._type(data), shape)
    } else if (!data.length && shape.length) {
      // if shape present but data not provided, initialize with 0s
      this.tensor = ndarray(new this._type(shape.reduce((a, b) => a * b, 1)), shape)
    } else {
      this.tensor = ndarray(new this._type([]), [])
    }

    // turn on weblas
    if (options.useWeblas && weblas) {
      this._useWeblas = true
      this.createWeblasTensor()
    } else {
      this._useWeblas = false
    }
  }

  /**
   * Create weblas pipeline tensor in GPU memory
   * 2-D only
   * see https://github.com/waylonflinn/weblas/wiki/Pipeline
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
      delete this.weblasTensor
    }
  }

  /**
   * Replaces data in the underlying ndarray.
   */
  replaceTensorData = data => {
    if (data && data.length && data instanceof this._type) {
      this.tensor.data = data
    } else if (data && data.length && data instanceof Array) {
      this.tensor.data = new this._type(data)
    } else {
      this.tensor = new this._type([])
    }
  }

}

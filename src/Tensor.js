import ndarray from 'ndarray'
import squeeze from 'ndarray-squeeze'
import { MAX_TEXTURE_SIZE } from './constants'

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

    if (data && data.length && (data instanceof this._type || data instanceof Array)) {
      checkShape(data, shape)
      this.tensor = ndarray(data, shape)
      this.tensor = ndarray(new this._type(data), shape)
    } else if (!data.length && shape.length) {
      // if shape present but data not provided, initialize with 0s
      this.tensor = ndarray(new this._type(shape.reduce((a, b) => a * b, 1)), shape)
    } else {
      this.tensor = ndarray(new this._type([]), [])
    }
  }

  /**
   * Create weblas pipeline tensor in GPU memory
   * 1-D or 2-D only
   * see https://github.com/waylonflinn/weblas/wiki/Pipeline
   *
   * gl.MAX_TEXTURE_SIZE is a limiting factor.
   * Where this is exceeded, weblas Tensor must be split.
   */
  createWeblasTensor () {
    if (this.weblasTensor) {
      this.weblasTensor.delete()
    }
    if (this.weblasTensorsSplit) {
      this.weblasTensorsSplit.forEach(t => t.delete())
    }

    if (this.tensor.shape.length === 1) {
      const len = this.tensor.shape[0]
      if (len > MAX_TEXTURE_SIZE) {
        this.weblasTensorsSplit = []
        const splitNum = Math.ceil(MAX_TEXTURE_SIZE / len)
        for (let i = 0; i < splitNum; i++) {
          const lo = i * Math.round(len / splitNum)
          const hi = Math.min(len, (i + 1) * Math.round(len / splitNum))
          const splitShape = [1, hi - lo]
          this.weblasTensorsSplit.push(new weblas.pipeline.Tensor(splitShape, this.tensor.data.subarray(lo, hi - lo)))
        }
      } else {
        const shape = [1, len]
        this.weblasTensor = new weblas.pipeline.Tensor(shape, this.tensor.data)
      }
    } else if (this.tensor.shape.length === 2) {
      if (this.tensor.shape.every(s => s > MAX_TEXTURE_SIZE)) {
        throw new Error('[Tensor] cannot create Tensor with both dimensions exceeding MAX_TEXTURE_SIZE')
      }

      const rows = this.tensor.shape[0]
      const cols = this.tensor.shape[1]
      if (rows > MAX_TEXTURE_SIZE) {
        this.weblasTensorsSplit = []
        const splitNum = Math.ceil(rows / MAX_TEXTURE_SIZE)
        for (let i = 0; i < splitNum; i++) {
          const lo = i * MAX_TEXTURE_SIZE
          const hi = Math.min(rows, (i + 1) * MAX_TEXTURE_SIZE)
          const splitShape = [hi - lo, cols]
          this.weblasTensorsSplit.push(
            new weblas.pipeline.Tensor(splitShape, this.tensor.data.subarray(lo * cols, hi * cols))
          )
        }
      } else if (cols > MAX_TEXTURE_SIZE) {
        this.weblasTensorsSplit = []
        const splitNum = Math.ceil(cols / MAX_TEXTURE_SIZE)
        for (let i = 0; i < splitNum; i++) {
          const lo = i * MAX_TEXTURE_SIZE
          const hi = Math.min(cols, (i + 1) * MAX_TEXTURE_SIZE)
          const splitShape = [rows, hi - lo]
          this.weblasTensorsSplit.push(
            new weblas.pipeline.Tensor(splitShape, this.tensor.data.slice(rows * lo, rows * hi))
          )
        }
      } else {
        const shape = this.tensor.shape
        this.weblasTensor = new weblas.pipeline.Tensor(shape, this.tensor.data)
      }
    } else {
      throw new Error('[Tensor] can only create weblas Tensor for 1-D or 2-D only')
    }
  }

  /**
   * Transfers weblas pipeline tensor from GPU memory
   */
  transferWeblasTensor () {
    if (this.weblasTensor) {
      const shape = this.weblasTensor.shape
      const arr = this.weblasTensor.transfer(true)
      this.tensor = squeeze(ndarray(arr, shape))
    }
  }

  /**
   * Delete weblas pipeline tensor
   */
  deleteWeblasTensor () {
    if (this.weblasTensor) {
      this.weblasTensor.delete()
      delete this.weblasTensor
    }
  }

  /**
   * Replaces data in the underlying ndarray.
   */
  replaceTensorData (data) {
    if (data && data.length && data instanceof this._type) {
      this.tensor.data = data
    } else if (data && data.length && data instanceof Array) {
      this.tensor.data = new this._type(data)
    } else {
      throw new Error('[Tensor] invalid input for replaceTensorData method.')
    }
  }

}

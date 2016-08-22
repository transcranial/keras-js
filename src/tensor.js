import ndarray from 'ndarray'

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
}

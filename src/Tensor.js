import { webgl2 } from './WebGL2'
import * as tensorUtils from './utils/tensorUtils'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import squeeze from 'ndarray-squeeze'

/**
 * Tensor class
 */
export default class Tensor {
  /**
   * Creates a tensor
   *
   * @param {(TypedArray|Array)} data
   * @param {number[]} shape
   * @param {Object} [options]
   */
  constructor(data, shape, options = {}) {
    this.arrayType = options.type || Float32Array

    if (data && data.length && (data instanceof this.arrayType || data instanceof Array)) {
      tensorUtils.checkShape(data, shape)
      if (data instanceof this.arrayType) {
        this.tensor = ndarray(data, shape)
      } else if (data instanceof Array) {
        this.tensor = ndarray(new this.arrayType(data), shape)
      }
    } else if (!data.length && shape.length) {
      // if shape present but data not provided, initialize with 0s
      this.tensor = ndarray(new this.arrayType(shape.reduce((a, b) => a * b, 1)), shape)
    } else {
      this.tensor = ndarray(new this.arrayType([]), [])
    }
  }

  /**
   * Creates WebGL2 texture
   *
   * Without args, defaults to gl.TEXTURE_2D and gl.R32F
   *
   * @param {string} type
   * @param {string} format
   */
  createGLTexture(type = '2d', format = 'float') {
    let shape = []
    if (this.tensor.shape.length === 1) {
      shape = [1, this.tensor.shape[0]]
      this.is1D = true
    } else if (this.tensor.shape.length === 2) {
      shape = this.tensor.shape
    } else if (this.tensor.shape.length === 3 && ['2d_array', '3d'].includes(type)) {
      shape = this.tensor.shape
    } else {
      throw new Error('[Tensor] cannot create WebGL2 texture.')
    }

    const gl = webgl2.context

    const options = webgl2.getWebGLTextureOptions({ type, format })
    const { textureTarget, textureInternalFormat, textureFormat, textureType } = options

    this.glTexture = gl.createTexture()
    gl.bindTexture(textureTarget, this.glTexture)
    webgl2.storeRef('texture', this.glTexture)

    if (type === '2d') {
      const data = this.tensor.data
      gl.texImage2D(textureTarget, 0, textureInternalFormat, shape[1], shape[0], 0, textureFormat, textureType, data)
    } else if (type === '2d_array' || type === '3d') {
      const data = tensorUtils.data3DLayoutForGL(this.arrayType, this.tensor, shape)
      gl.texImage3D(
        textureTarget,
        0,
        textureInternalFormat,
        shape[1],
        shape[0],
        shape[2],
        0,
        textureFormat,
        textureType,
        data
      )
    }

    this.glTextureShape = shape
    this.glTextureConfig = { type, format }

    // clamp to edge
    gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    // no interpolation
    gl.texParameteri(textureTarget, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(textureTarget, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  }

  /**
   * Deletes WebGLTexture
   */
  deleteGLTexture() {
    if (this.glTexture) {
      const gl = webgl2.context
      gl.deleteTexture(this.glTexture)
      delete this.glTexture
    }
  }

  /**
   * Replaces data in the underlying ndarray, and the corresponding WebGLTexture if glTexture is present
   *
   * @param {number[]} data
   */
  replaceTensorData(data) {
    if (data && data.length && data instanceof this.arrayType) {
      this.tensor.data.set(data)
    } else if (data && data.length && data instanceof Array) {
      this.tensor.data.set(new this.arrayType(data))
    } else {
      throw new Error('[Tensor] invalid input for replaceTensorData method.')
    }

    if (this.glTexture) {
      const gl = webgl2.context
      const { type, format } = this.glTextureConfig
      const options = webgl2.getWebGLTextureOptions({ type, format })
      const { textureTarget, textureFormat, textureType } = options
      gl.bindTexture(textureTarget, this.glTexture)

      const shape = this.glTextureShape
      if (type === '2d') {
        const data = this.tensor.data
        gl.texSubImage2D(textureTarget, 0, 0, 0, shape[1], shape[0], textureFormat, textureType, data, 0)
      } else if (type === '2d_array' || type === '3d') {
        const data = tensorUtils.data3DLayoutForGL(this.arrayType, this.tensor, shape)
        gl.texSubImage3D(textureTarget, 0, 0, 0, 0, shape[1], shape[0], shape[2], textureFormat, textureType, data, 0)
      }
    }
  }

  /**
   * Transfer data from webgl texture on GPU to ndarray on CPU
   */
  transferFromGLTexture() {
    this.tensor = ndarray(new this.arrayType([]), this.glTextureShape)
    this.tensor.data = webgl2.readData(this.glTextureShape)
    if (this.is1D && this.glTextureShape[0] === 1) {
      // collapse to 1D
      this.tensor = squeeze(this.tensor, [0])
    }
  }

  /**
   * Reshapes data into 2D representation preserving single axis
   *
   * @param {number} axis
   */
  reshapeTo2D(axis = -1) {
    if (axis < 0) {
      axis = this.tensor.shape.length + axis
    }

    const axisSize = this.tensor.shape[axis]
    const otherAxes = [...this.tensor.shape.slice(0, axis), ...this.tensor.shape.slice(axis + 1)]
    const otherAxesSize = otherAxes.reduce((a, b) => a * b, 1)

    const reshaped = ndarray(new this.arrayType(otherAxesSize * axisSize), [otherAxesSize, axisSize])

    const otherAxesData = ndarray(new this.arrayType(otherAxesSize), otherAxes)
    const otherAxesDataRaveled = ndarray(new this.arrayType(otherAxesSize), [otherAxesSize])
    const axisSlices = Array(this.tensor.shape.length).fill(null)
    for (let n = 0; n < axisSize; n++) {
      axisSlices[axis] = n
      ops.assign(otherAxesData, this.tensor.pick(...axisSlices))
      otherAxesDataRaveled.data = otherAxesData.data
      ops.assign(reshaped.pick(null, n), otherAxesDataRaveled)
    }

    this.originalShape = this.tensor.shape
    this.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.tensor.shape, false, axis)
    this.tensor = reshaped
    this.is2DReshaped = true
  }

  /**
   * Reshapes tensor in 2D representation back to original
   *
   * Typically called at the end when data is read back from GPU
   *
   * @param {number} axis
   */
  reshapeFrom2D(axis = -1) {
    if (!this.is2DReshaped) {
      throw new Error('[Tensor] not in reshaped 2D representation.')
    }
    if (!this.originalShape) {
      throw new Error('[Tensor] does not contain originalShape.')
    }

    if (axis < 0) {
      axis = this.originalShape.length + axis
    }

    // second axis is the channel, or common, axis
    const channelDataSize = this.tensor.shape[0]
    const channels = this.tensor.shape[1]

    const reshaped = ndarray(new this.arrayType(this.originalShape.reduce((a, b) => a * b, 1)), this.originalShape)
    const channelDataRaveled = ndarray(new this.arrayType(channelDataSize), [channelDataSize])
    const unraveledChannelShape = [...this.originalShape.slice(0, axis), ...this.originalShape.slice(axis + 1)]
    const unraveledChannel = ndarray(
      new this.arrayType(unraveledChannelShape.reduce((a, b) => a * b, 1)),
      unraveledChannelShape
    )
    const axisSlices = Array(this.originalShape.length).fill(null)
    for (let n = 0; n < channels; n++) {
      ops.assign(channelDataRaveled, this.tensor.pick(null, n))
      unraveledChannel.data = channelDataRaveled.data
      axisSlices[axis] = n
      ops.assign(reshaped.pick(...axisSlices), unraveledChannel)
    }

    this.tensor = reshaped
  }

  /**
   * Reshapes data into 2D square representation (underlying data remaining contiguous)
   */
  reshapeTo2DSquare() {
    const squareDim = Math.ceil(Math.sqrt(this.tensor.size))
    const reshaped = ndarray(new this.arrayType(squareDim ** 2), [squareDim, squareDim])
    reshaped.data.set(this.tensor.data)

    const indicesRowArr = ndarray(new Int32Array(this.tensor.size), this.tensor.shape)
    const indicesColArr = ndarray(new Int32Array(this.tensor.size), this.tensor.shape)
    const indicesRowArrReshaped = ndarray(new Int32Array(squareDim ** 2), [squareDim, squareDim])
    const indicesColArrReshaped = ndarray(new Int32Array(squareDim ** 2), [squareDim, squareDim])
    for (let i = 0; i < squareDim; i++) {
      ops.assigns(indicesRowArrReshaped.pick(i, null), i)
    }
    for (let j = 0; j < squareDim; j++) {
      ops.assigns(indicesColArrReshaped.pick(null, j), j)
    }
    indicesRowArr.data.set(indicesRowArrReshaped.data.subarray(0, indicesRowArr.size))
    indicesColArr.data.set(indicesColArrReshaped.data.subarray(0, indicesColArr.size))

    this.originalShape = this.tensor.shape
    this.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.tensor.shape, true)
    this.tensor = reshaped
    this.is2DReshaped = true
  }

  /**
   * Reshapes tensor in 2D square representation back to original (underlying data remains contiguous)
   */
  reshapeFrom2DSquare() {
    if (!this.is2DReshaped || this.tensor.shape.length !== 2 || this.tensor.shape[0] !== this.tensor.shape[1]) {
      throw new Error('[Tensor] not in reshaped 2D square representation.')
    }
    if (!this.originalShape) {
      throw new Error('[Tensor] does not contain originalShape.')
    }

    const size = this.originalShape.reduce((a, b) => a * b, 1)
    const reshaped = ndarray(new this.arrayType(size), this.originalShape)
    reshaped.data.set(this.tensor.data.subarray(0, size))

    this.tensor = reshaped
  }
}

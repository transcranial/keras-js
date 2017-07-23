import Tensor from './Tensor'
import { webgl2 } from './WebGL2'
import ops from 'ndarray-ops'

/**
 * Layer class
 */
export default class Layer {
  /**
   * Creates a layer
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    this.layerClass = 'Layer'
    this.name = attrs.name
    this.gpu = webgl2.isSupported && attrs.gpu

    this.params = []
    this.weights = {}

    this.inbound = []
    this.outbound = []
  }

  /**
   * Set layer weights
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr, createGLTexture = true) {
    this.params.forEach((p, i) => {
      this.weights[p] = weightsArr[i]

      if (this.gpu && createGLTexture) {
        this.weights[p].createGLTexture()
      }
    })
  }

  /**
   * Layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    return x
  }

  /**
   * Toggle GPU mode on/off
   *
   * @param {Boolean} mode - on/off
   */
  toggleGpu(mode) {
    const newMode = typeof mode === 'undefined' ? !this.gpu : mode
    if (webgl2.isSupported && newMode) {
      this.gpu = true
    } else {
      this.gpu = false
    }
  }

  /**
   * Reshapes tiled data into untiled form.
   * Called at the end when data is read back from GPU (which is in tiled 2D format from texture)
   *
   * @param {Tensor} x
   * @param {Number} axis
   * @returns {Tensor}
   */
  reshapeTensorFromTiled(x, axis = -1) {
    if (!x.glTextureIsTiled) {
      throw new Error('Tensor is not in tiled format.')
    }
    if (!x.untiledShape) {
      throw new Error('Tensor does not contain untiledShape.')
    }

    if (axis < 0) {
      axis = x.untiledShape.length + axis
    }

    // second axis is the channel, or common, axis
    const channelDataSize = x.tensor.shape[0]
    const channels = x.tensor.shape[1]

    const reshaped = new Tensor([], x.untiledShape)
    const channelDataRaveled = new Tensor([], [channelDataSize])
    const untiledChannelShape = [...x.untiledShape.slice(0, axis), ...x.untiledShape.slice(axis + 1)]
    const untiledChannel = new Tensor([], untiledChannelShape)
    const axisSlices = Array(x.untiledShape.length).fill(null)
    for (let n = 0; n < channels; n++) {
      ops.assign(channelDataRaveled.tensor, x.tensor.pick(null, n))
      untiledChannel.replaceTensorData(channelDataRaveled.tensor.data)
      axisSlices[axis] = n
      ops.assign(reshaped.tensor.pick(...axisSlices), untiledChannel.tensor)
    }

    return reshaped
  }
}

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
   * @returns {Tensor}
   */
  reshapeTensorFromTiled(x) {
    if (!x.glTextureIsTiled) {
      throw new Error('Tensor is not in tiled format.')
    }
    if (!x.untiledShape) {
      throw new Error('Tensor does not contain untiledShape.')
    }

    // last axis is channel axis
    const nbPatches = x.tensor.shape[0]
    const channels = x.tensor.shape[1]

    let reshaped = new Tensor([], x.untiledShape)
    let outputChannelRaveled = new Tensor([], [nbPatches])
    let outputChannel = new Tensor([], x.untiledShape.slice(0, -1))
    for (let n = 0; n < channels; n++) {
      ops.assign(outputChannelRaveled.tensor, x.tensor.pick(null, n))
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data)
      const axisSlices = Array(x.untiledShape.length - 1).fill(null)
      ops.assign(reshaped.tensor.pick(...axisSlices, n), outputChannel.tensor)
    }

    return reshaped
  }
}

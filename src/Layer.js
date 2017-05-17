import Tensor from './Tensor'
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

    this.params = []
    this.weights = {}

    // gpu and pipeline flags from Model
    this.gpu = attrs.gpu
    this.pipeline = attrs.pipeline

    // layer flags off by default
    this._useWeblas = false
    this._pipelineEnabled = false
  }

  /**
   * Method for setting layer weights
   * We store the weights as both Tensor instances,
   * as well as weblas pipeline tensors if possible (which are in GPU memory)
   * see https://github.com/waylonflinn/weblas/wiki/Pipeline
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this.params.forEach((p, i) => {
      this.weights[p] = weightsArr[i]
    })
  }

  /**
   * Toggle GPU mode on/off
   * weblas must be available
   * @param {boolean} mode - on/off
   */
  toggleGpu(mode) {
    const newMode = typeof mode === 'undefined' ? !this._useWeblas : mode
    if (newMode && weblas) {
      this._useWeblas = true
    } else {
      this._useWeblas = false
    }
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    return x
  }

  /**
   * Pipeline transfer
   * Typically called at the end of a pipelined layer sequence.

   * @param {Tensor} x
   * @returns {Tensor} x
   */
  transferFromPipeline(x) {
    if (!x.weblasTensor) {
      throw new Error('Variable passed in does not contain weblas tensor.')
    }
    if (!x._fromPipeline) {
      throw new Error('Variable passed in does not contain _fromPipeline.')
    }
    if (!x._actualShape) {
      throw new Error('Variable passed in does not contain _actualShape.')
    }

    // last axis is channel axis
    const channels = x.weblasTensor.shape[1]
    const nbPatches = x._actualShape.slice(0, -1).reduce((a, b) => a * b, 1)

    const tiled = new Tensor([], x.weblasTensor.shape)
    tiled.tensor.data = x.weblasTensor.transfer()

    let output = new Tensor([], x._actualShape)
    let outputChannelRaveled = new Tensor([], [nbPatches])
    let outputChannel = new Tensor([], x._actualShape.slice(0, -1))
    for (let n = 0; n < channels; n++) {
      ops.assign(outputChannelRaveled.tensor, tiled.tensor.pick(null, n))
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data)
      const axisSlices = Array(x._actualShape.length - 1).fill(null)
      ops.assign(output.tensor.pick(...axisSlices, n), outputChannel.tensor)
    }

    output._fromPipeline = false
    if (output._actualShape) {
      delete output._actualShape
    }

    return output
  }
}

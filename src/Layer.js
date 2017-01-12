/**
 * Layer class
 */
export default class Layer {
  /**
   * Creates a layer
   * @param {Object} [attrs] - layer attributes
   */
  constructor (attrs = {}) {
    this.layerClass = 'Layer'
    this.name = attrs.name

    this.params = []
    this.weights = {}

    // turn on weblas
    this._useWeblas = false
    this._pipelineEnabled = false
    if (attrs.gpu && weblas) {
      this._useWeblas = true
      if (attrs.pipeline) {
        this._pipelineEnabled = true
      }
    }
  }

  /**
   * Method for setting layer weights
   * We store the weights as both Tensor instances,
   * as well as weblas pipeline tensors if possible (which are in GPU memory)
   * see https://github.com/waylonflinn/weblas/wiki/Pipeline
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights (weightsArr) {
    this.params.forEach((p, i) => {
      this.weights[p] = weightsArr[i]
    })
  }

  /**
   * Toggle GPU mode on/off
   * weblas must be available
   * @param {boolean} mode - on/off
   */
  toggleGpu (mode) {
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
  call (x) {
    return x
  }
}

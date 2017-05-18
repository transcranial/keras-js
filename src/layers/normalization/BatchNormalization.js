import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'
import checkPipelineSupport from '../../utils/checkPipelineSupport'
import WebGLBatchNorm from '../../ext/normalization/WebGLBatchNorm'

/**
 * BatchNormalization layer class
 */
export default class BatchNormalization extends Layer {
  /**
   * Creates an BatchNormalization layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'BatchNormalization'

    const { epsilon = 0.001, axis = -1, center = true, scale = true } = attrs

    this.epsilon = epsilon
    this.center = center
    this.scale = scale

    // no batch axis, so axis is less 1 compared to representation in keras
    // will be set in call(), as input tensor shape is needed to calculate axis
    // if axis < 0
    this.axis = axis
    this.axisNormalized = false

    // Layer weights specification
    this.params = []
    if (this.scale) {
      this.params.push('gamma')
    }
    if (this.center) {
      this.params.push('beta')
    }
    this.params = this.params.concat(['moving_mean', 'moving_variance'])

    // Enable layer gpu +/- pipeline mode if supported
    if (this.gpu && weblas) {
      this._useWeblas = true
      if (this.pipeline) {
        const isPipelineModeSupported = checkPipelineSupport(this.layerClass, attrs)
        if (isPipelineModeSupported) {
          this._pipelineEnabled = true
          this.webglBatchNorm = new WebGLBatchNorm()
        } else {
          this._pipelineEnabled = false
        }
      }
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)

    if (this._useWeblas) {
      this.params.forEach(param => {
        this.weights[param].createWeblasTensor()
      })
    }
  }

  /**
   * Runs layer computational logic in pipeline mode
   * Only works with a previous convolutional layer with its output containing
   * a weblas pipeline tensor which is a 2-D tiled representation (tile data, channels).
   * The output after normalization is still a 2-D tiled representation (typically as input
   * to convolution or merge layers running in pipeline mode).
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callPipelineMode(x) {
    if (!x._fromPipeline) {
      return this._callRegularMode(x)
    }

    x.weblasTensor = this.webglBatchNorm.call(
      x.weblasTensor,
      this.epsilon,
      this.weights.gamma.weblasTensor,
      this.weights.beta.weblasTensor,
      this.weights.moving_mean.weblasTensor,
      this.weights.moving_variance.weblasTensor
    )

    return x
  }

  /**
   * Runs layer computational logic in regular mode
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callRegularMode(x) {
    if (!this.axisNormalized) {
      this.axis = this.axis < 0 ? x.tensor.shape.length + this.axis : this.axis - 1
      this.axisNormalized = true
    }

    let broadcast = []
    for (let d = 0; d < x.tensor.shape.length; d++) {
      if (d === this.axis) broadcast.push(1)
      else broadcast.push(null)
    }

    // broadcast weights
    let _gamma = new Tensor([], x.tensor.shape)
    let _beta = new Tensor([], x.tensor.shape)
    for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
      broadcast[this.axis] = i
      if (this.scale) {
        ops.assigns(_gamma.tensor.pick(...broadcast), this.weights.gamma.tensor.get(i))
      }
      if (this.center) {
        ops.assigns(_beta.tensor.pick(...broadcast), this.weights.beta.tensor.get(i))
      }
    }

    let _mean = new Tensor([], x.tensor.shape)
    let _std = new Tensor([], x.tensor.shape)

    // feature-wise normalization
    for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
      broadcast[this.axis] = i
      ops.assigns(_mean.tensor.pick(...broadcast), this.weights.moving_mean.tensor.get(i))
      ops.assigns(_std.tensor.pick(...broadcast), this.weights.moving_variance.tensor.get(i) + this.epsilon)
    }
    ops.sqrteq(_std.tensor)

    ops.subeq(x.tensor, _mean.tensor)
    ops.diveq(x.tensor, _std.tensor)
    if (this.scale) {
      ops.muleq(x.tensor, _gamma.tensor)
    }
    if (this.center) {
      ops.addeq(x.tensor, _beta.tensor)
    }

    return x
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (this._pipelineEnabled) {
      return this._callPipelineMode(x)
    } else {
      return this._callRegularMode(x)
    }
  }
}

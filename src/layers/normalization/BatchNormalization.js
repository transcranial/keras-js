import Layer from '../../engine/Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'
import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'

/**
 * BatchNormalization layer class
 */
export default class BatchNormalization extends Layer {
  /**
   * Creates an BatchNormalization layer
   */
  constructor (attrs = {}) {
    super(attrs)
    const {
      epsilon = 1e-5,
      mode = 0,
      axis = -1
    } = attrs

    this.epsilon = epsilon
    this.mode = mode
    this.axis = axis

    // Layer weights specification
    // running mean and std are non_trainable_weights in mode 0
    this.params = this.mode === 0
      ? ['gamma', 'beta', 'running_mean', 'running_std']
      : ['gamma', 'beta']
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    // no batch axis, so axis is less 1 compared to representation in keras
    this.axis = this.axis < 0 ? x.tensor.shape.length + this.axis : this.axis - 1

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
      ops.assigns(_gamma.tensor.pick(...broadcast), this.weights.gamma.tensor.get(i))
      ops.assigns(_beta.tensor.pick(...broadcast), this.weights.beta.tensor.get(i))
    }

    let _mean = new Tensor([], x.tensor.shape)
    let _std = new Tensor([], x.tensor.shape)

    if (this.mode === 0) {
      // feature-wise normalization
      for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
        broadcast[this.axis] = i
        ops.assigns(_mean.tensor.pick(...broadcast), this.weights.running_mean.tensor.get(i))
        ops.assigns(_std.tensor.pick(...broadcast), this.weights.running_std.tensor.get(i) + this.epsilon)
      }
      ops.sqrteq(_std.tensor)
    } else if (this.mode === 1) {
      // sample-wise normalization
      let reducedShape = x.tensor.shape.slice()
      reducedShape.splice(this.axis, 1)

      // mean
      let sampleMean = new Tensor([], reducedShape)
      for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
        broadcast[this.axis] = i
        ops.addeq(sampleMean.tensor, x.tensor.pick(...broadcast))
      }
      ops.divseq(sampleMean.tensor, x.tensor.shape[this.axis])

      // stddev
      let sampleStd = new Tensor([], reducedShape)
      let stdTemp = new Tensor([], reducedShape)
      for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
        broadcast[this.axis] = i
        ops.sub(stdTemp.tensor, x.tensor.pick(...broadcast), sampleMean.tensor)
        ops.powseq(stdTemp.tensor, 2)
        ops.addeq(sampleStd.tensor, stdTemp.tensor)
      }
      ops.divseq(sampleStd.tensor, x.tensor.shape[this.axis])
      ops.addseq(sampleStd.tensor, this.epsilon)
      ops.sqrteq(sampleStd.tensor)
      ops.addseq(sampleStd.tensor, this.epsilon)

      // broadcast
      for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
        broadcast[this.axis] = i
        ops.assign(_mean.tensor.pick(...broadcast), sampleMean.tensor)
        ops.assign(_std.tensor.pick(...broadcast), sampleStd.tensor)
      }
    } else if (this.mode === 2) {
      // feature-wise normalization, using per-batch statistics
      // here, batch size always = 1
      for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
        broadcast[this.axis] = i
        let reduction = flattenDeep(unpack(x.tensor.pick(...broadcast)))
        let axisMean = reduction.reduce((a, b) => a + b, 0) / reduction.length
        let axisStd = reduction.map(x => (x - axisMean) * (x - axisMean)).reduce((a, b) => a + b, 0) / reduction.length
        ops.assigns(_mean.tensor.pick(...broadcast), axisMean)
        ops.assigns(_std.tensor.pick(...broadcast), axisStd + this.epsilon)
      }
      ops.sqrteq(_std.tensor)
    } else {
      throw new Error(`[normalization.BatchNormalization] Invalid mode ${this.mode}.`)
    }

    ops.subeq(x.tensor, _mean.tensor)
    ops.diveq(x.tensor, _std.tensor)
    ops.muleq(x.tensor, _gamma.tensor)
    ops.addeq(x.tensor, _beta.tensor)

    return x
  }
}

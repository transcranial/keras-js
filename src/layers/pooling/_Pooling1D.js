import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * _Pooling1D layer class
 */
export default class _Pooling1D extends Layer {
  /**
   * Creates a _Pooling1D layer
   */
  constructor (attrs = {}) {
    super(attrs)
    this.layerClass = '_Pooling1D'

    const {
      poolLength = 2,
      stride = null,
      borderMode = 'valid'
    } = attrs

    this.poolLength = poolLength
    this.stride = stride === null ? poolLength : stride
    this.borderMode = borderMode

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max'
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call (x) {
    if (this.poolingFunc !== 'max' && this.poolingFunc !== 'average') {
      throw new Error(`[pooling._Pooling1D] pooling function must be max or average.`)
    }

    const stepsNew = this.borderMode === 'valid'
      ? Math.floor((x.tensor.shape[0] - this.poolLength + this.stride) / this.stride)
      : Math.floor((x.tensor.shape[0] + this.stride - 1) / this.stride)

    let y = new Tensor([], [stepsNew, x.tensor.shape[1]])
    let yStep = new Tensor([], [x.tensor.shape[1]])

    // in borderMode same, start negative from beyond step 0
    let step = this.borderMode === 'valid'
      ? 0
      : Math.min(0, Math.ceil((x.tensor.shape[0] - (stepsNew - 1) * this.stride - this.poolLength) / 2))

    for (let i = 0; i < stepsNew; i++) {
      let _step = Math.max(0, step)
      let limit = this.poolLength + Math.min(0, step)
      ops.assign(yStep.tensor, x.tensor.pick(_step, null))

      let count = 1
      for (let j = 1; j < limit; j++) {
        if ((_step + j) > (x.tensor.shape[0] - 1)) {
          break
        }
        if (this.poolingFunc === 'max') {
          ops.maxeq(yStep.tensor, x.tensor.pick(_step + j, null))
        } else if (this.poolingFunc === 'average') {
          ops.addeq(yStep.tensor, x.tensor.pick(_step + j, null))
        }
        count += 1
      }

      if (this.poolingFunc === 'average') {
        ops.divseq(yStep.tensor, count)
      }

      ops.assign(y.tensor.pick(i, null), yStep.tensor)
      step += this.stride
    }

    x.tensor = y.tensor
    return x
  }
}

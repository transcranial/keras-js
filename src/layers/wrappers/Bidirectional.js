import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'
import pick from 'lodash/pick'
import * as recurrentLayers from '../recurrent'

/**
 * Bidirectional wrapper layer class
 */
export default class Bidirectional extends Layer {
  /**
   * Creates a Bidirectional wrapper layer
   * @param {Layer} attrs.layer
   * @param {String} [attrs.merge_mode] - one of concat, mul, sum, ave
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Bidirectional'

    const { layer, merge_mode = 'concat' } = attrs

    if (!layer) throw new Error('[Bidirectional] wrapped layer is undefined.')

    this.forwardLayer = layer

    let backwardLayerAttrs = {
      units: this.forwardLayer.units,
      activation: this.forwardLayer.activation,
      recurrent_activation: this.forwardLayer.recurrentActivation,
      return_sequences: this.forwardLayer.returnSequences,
      go_backwards: !this.forwardLayer.goBackwards,
      stateful: this.forwardLayer.stateful
    }
    this.backwardLayer = new recurrentLayers[layer.layerClass](backwardLayerAttrs)

    this.mergeMode = merge_mode
  }

  /**
   * Method for setting layer weights
   * Passes weights to the wrapped layer
   * Here, the weights array is concatenated from the forward layer and the backward layer
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this.forwardLayer.setWeights(weightsArr.slice(0, weightsArr.length / 2))
    this.backwardLayer.setWeights(weightsArr.slice(weightsArr.length / 2))
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    let xForward = new Tensor(x.tensor.data, x.tensor.shape)
    let xBackward = new Tensor(x.tensor.data, x.tensor.shape)
    let yForward = this.forwardLayer.call(xForward)
    let yBackward = this.backwardLayer.call(xBackward)

    if (this.mergeMode === 'concat') {
      let outShape = yForward.tensor.shape.slice()
      outShape[outShape.length - 1] += yBackward.tensor.shape[outShape.length - 1]
      let y = new Tensor([], outShape)
      if (this.forwardLayer.returnSequences) {
        ops.assign(y.tensor.hi(outShape[0], yForward.tensor.shape[1]).lo(0, 0), yForward.tensor)
        // when returnSequences = true, reverse results of backwardLayer before concat
        ops.assign(y.tensor.hi(outShape[0], outShape[1]).lo(0, yForward.tensor.shape[1]), yBackward.tensor.step(-1))
      } else {
        ops.assign(y.tensor.hi(yForward.tensor.shape[0]).lo(0), yForward.tensor)
        ops.assign(y.tensor.hi(outShape[0]).lo(yForward.tensor.shape[0]), yBackward.tensor)
      }
      x.tensor = y.tensor
    } else if (this.mergeMode === 'sum') {
      let outShape = yForward.tensor.shape.slice()
      let y = new Tensor([], outShape)
      ops.addeq(y.tensor, yForward.tensor)
      ops.addeq(y.tensor, yBackward.tensor)
      x.tensor = y.tensor
    } else if (this.mergeMode === 'mul') {
      let outShape = yForward.tensor.shape.slice()
      let y = new Tensor([], outShape)
      ops.assigns(y.tensor, 1)
      ops.muleq(y.tensor, yForward.tensor)
      ops.muleq(y.tensor, yBackward.tensor)
      x.tensor = y.tensor
    } else if (this.mergeMode === 'ave') {
      let outShape = yForward.tensor.shape.slice()
      let y = new Tensor([], outShape)
      ops.addeq(y.tensor, yForward.tensor)
      ops.addeq(y.tensor, yBackward.tensor)
      ops.divseq(y.tensor, 2)
      x.tensor = y.tensor
    }

    return x
  }
}

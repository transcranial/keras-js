import Tensor from '../../Tensor'
import Layer from '../../Layer'
import gemm from 'ndarray-gemm'
import ops from 'ndarray-ops'
import unsqueeze from 'ndarray-unsqueeze'
import concatFirstAxis from 'ndarray-concat-rows'
import isEqual from 'lodash/isEqual'
import range from 'lodash/range'
import sum from 'lodash/sum'

/**
 * _Merge layer class
 */
export default class _Merge extends Layer {
  /**
   * Creates a _Merge layer
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = '_Merge'
  }

  /**
   * Internal method for validating inputs
   * @param {Tensor[]} inputs
   * @returns {Boolean} valid
   */
  _validateInputs(inputs) {
    const shapes = inputs.map(x => x.tensor.shape.slice())
    if (['sum', 'mul', 'ave', 'max'].indexOf(this.mode) > -1) {
      if (!shapes.every(shape => isEqual(shape, shapes[0]))) {
        throw new Error(
          `${this.name} [${this.layerClass} layer] All input shapes must be the same for mode ${this.mode}.`
        )
      }
    }
    if (this.mode === 'dot') {
      if (inputs.length !== 2) {
        throw new Error(`${this.name} [${this.layerClass} layer] Exactly 2 inputs required for mode ${this.mode}.`)
      }
      if (this.dotAxes[0] < 0) {
        this.dotAxes[0] = shapes[0].length + this.dotAxes[0]
      }
      if (this.dotAxes[1] < 0) {
        this.dotAxes[1] = shapes[1].length + this.dotAxes[1]
      }
      if (shapes[0][this.dotAxes[0]] !== shapes[1][this.dotAxes[1]]) {
        throw new Error(`${this.name} [${this.layerClass} layer] Dimensions incompatibility using dot mode.`)
      }
    } else if (this.mode === 'concat') {
      let nonConcatShapes = shapes.slice()
      let _concatAxis = this.concatAxis < 0 ? nonConcatShapes[0].length + this.concatAxis : this.concatAxis
      if (this.concatAxis === 0) _concatAxis = 0
      range(nonConcatShapes.length).forEach(i => {
        nonConcatShapes[i].splice(_concatAxis, 1)
      })
      if (!nonConcatShapes.every(shape => isEqual(shape, nonConcatShapes[0]))) {
        throw new Error(
          `${this.name} [${this.layerClass} layer] In concat mode, all shapes must be the same except along the concat axis.`
        )
      }
    }
    return true
  }

  /**
   * Runs layer computational logic in pipeline mode
   * Only works with inputs containing weblas pipeline tensors which are 2-D tiled
   * representations (tile data, channels).
   * For now, support only ['sum', 'mul', 'concat', 'ave', 'max']
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _callPipelineMode(inputs) {
    if (!inputs.every(x => x._fromPipeline)) {
      return this._callRegularMode(inputs)
    }

    let output = new Tensor([], inputs[0].weblasTensor.shape)

    output.weblasTensor = this.webglMerge.call(inputs.map(x => x.weblasTensor))

    output._fromPipeline = true
    output._actualShape = inputs[0]._actualShape
    if (this.mode === 'concat') {
      output._actualShape = [...inputs[0]._actualShape.slice(0, -1), sum(inputs.map(x => x._actualShape.slice(-1)[0]))]
    }

    return output
  }

  /**
   * Runs layer computational logic in regular mode
   * @param {Tensor[]} inputs
   * @returns {Tensor} output
   */
  _callRegularMode(inputs) {
    const valid = this._validateInputs(inputs)
    if (!valid) {
      throw new Error(`${this.name} [${this.layerClass} layer] Invalid inputs to call method.`)
    }

    let output
    let outputShape
    if (['sum', 'mul', 'ave', 'max'].indexOf(this.mode) > -1) {
      outputShape = inputs[0].tensor.shape.slice()
      output = new Tensor([], outputShape)
    } else if (this.mode === 'concat') {
      outputShape = inputs[0].tensor.shape.slice()
      let _concatAxis = this.concatAxis < 0 ? outputShape.length + this.concatAxis : this.concatAxis
      if (this.concatAxis === 0) _concatAxis = 0
      inputs.slice(1, inputs.length).forEach(x => {
        const d = x.tensor.shape.slice()[_concatAxis]
        outputShape[_concatAxis] += d
      })
      output = new Tensor([], outputShape)
    } else if (this.mode === 'dot') {
      let shape1 = inputs[0].tensor.shape.slice()
      let shape2 = inputs[1].tensor.shape.slice()
      shape1.splice(this.dotAxes[0], 1)
      shape2.splice(this.dotAxes[1], 1)
      outputShape = shape1.concat(shape2)
      if (outputShape.length === 1) {
        outputShape.push(1)
      }
      output = new Tensor([], outputShape)
    }

    if (this.mode === 'sum') {
      for (let i = 0; i < inputs.length; i++) {
        ops.addeq(output.tensor, inputs[i].tensor)
      }
    } else if (this.mode === 'mul') {
      ops.assigns(output.tensor, 1)
      for (let i = 0; i < inputs.length; i++) {
        ops.muleq(output.tensor, inputs[i].tensor)
      }
    } else if (this.mode === 'ave') {
      for (let i = 0; i < inputs.length; i++) {
        ops.addeq(output.tensor, inputs[i].tensor)
      }
      ops.divseq(output.tensor, inputs.length)
    } else if (this.mode === 'max') {
      ops.assign(output.tensor, inputs[0].tensor)
      for (let i = 1; i < inputs.length; i++) {
        ops.maxeq(output.tensor, inputs[i].tensor)
      }
    } else if (this.mode === 'concat') {
      let _concatAxis = this.concatAxis < 0 ? inputs[0].tensor.shape.length + this.concatAxis : this.concatAxis
      if (this.concatAxis === 0) _concatAxis = 0
      if (_concatAxis === 0) {
        concatFirstAxis(output.tensor, inputs.map(x => x.tensor))
      } else {
        let dimsAxisSwap = [_concatAxis]
        for (let i = 0; i < inputs[0].tensor.shape.length; i++) {
          if (i !== _concatAxis) dimsAxisSwap.push(i)
        }
        concatFirstAxis(output.tensor.transpose(...dimsAxisSwap), inputs.map(x => x.tensor.transpose(...dimsAxisSwap)))
      }
    } else if (this.mode === 'dot') {
      if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
        if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
          if (this.normalize) {
            for (let i = 0; i < inputs[0].tensor.shape[1]; i++) {
              ops.divseq(inputs[0].tensor.pick(null, i), ops.norm2(inputs[0].tensor.pick(null, i)))
            }
            for (let i = 0; i < inputs[1].tensor.shape[1]; i++) {
              ops.divseq(inputs[1].tensor.pick(null, i), ops.norm2(inputs[1].tensor.pick(null, i)))
            }
          }
          gemm(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor)
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          if (this.normalize) {
            for (let i = 0; i < inputs[0].tensor.shape[0]; i++) {
              ops.divseq(inputs[0].tensor.pick(i, null), ops.norm2(inputs[0].tensor.pick(i, null)))
            }
            for (let i = 0; i < inputs[1].tensor.shape[0]; i++) {
              ops.divseq(inputs[1].tensor.pick(i, null), ops.norm2(inputs[1].tensor.pick(i, null)))
            }
          }
          gemm(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0))
        }
      } else {
        throw new Error(`${this.name} [${this.layerClass} layer] dot mode for 3+ dim tensors not yet implemented.`)
      }
    }

    return output
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

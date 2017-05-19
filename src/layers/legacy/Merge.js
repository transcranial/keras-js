import Tensor from '../../Tensor'
import Layer from '../../Layer'
import gemm from 'ndarray-gemm'
import ops from 'ndarray-ops'
import unsqueeze from 'ndarray-unsqueeze'
import concatFirstAxis from 'ndarray-concat-rows'
import isEqual from 'lodash/isEqual'
import range from 'lodash/range'

/**
 * Merge layer class
 */
export default class Merge extends Layer {
  /**
   * Creates a Merge layer
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Merge'

    const { mode = 'sum', concat_axis = -1, dot_axes = -1 } = attrs

    const availableModes = ['sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max']
    if (availableModes.indexOf(mode) > -1) {
      this.mode = mode
    } else {
      throw new Error(`${this.name} [Merge layer] ${mode} not available.`)
    }

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = concat_axis <= 0 ? concat_axis : concat_axis - 1
    if (Array.isArray(dot_axes)) {
      this.dotAxes = [
        dot_axes[0] <= 0 ? dot_axes[0] : dot_axes[0] - 1,
        dot_axes[1] <= 0 ? dot_axes[1] : dot_axes[1] - 1
      ]
    } else {
      this.dotAxes = [dot_axes <= 0 ? dot_axes : dot_axes - 1, dot_axes <= 0 ? dot_axes : dot_axes - 1]
    }
  }

  /**
   * Internal method for validating inputs
   * @param {Tensor[]} inputs
   * @returns {boolean} valid
   */
  _validateInputs(inputs) {
    const shapes = inputs.map(x => x.tensor.shape.slice())
    if (['sum', 'mul', 'ave', 'cos', 'max'].indexOf(this.mode) > -1) {
      if (!shapes.every(shape => isEqual(shape, shapes[0]))) {
        throw new Error(`${this.name} [Merge layer] All input shapes must be the same for mode ${this.mode}.`)
      }
    }
    if (['cos', 'dot'].indexOf(this.mode) > -1) {
      if (inputs.length !== 2) {
        throw new Error(`${this.name} [Merge layer] Exactly 2 inputs required for mode ${this.mode}.`)
      }
      if (this.dotAxes[0] < 0) {
        this.dotAxes[0] = shapes[0].length + this.dotAxes[0]
      }
      if (this.dotAxes[1] < 0) {
        this.dotAxes[1] = shapes[1].length + this.dotAxes[1]
      }
      if (shapes[0][this.dotAxes[0]] !== shapes[1][this.dotAxes[1]]) {
        throw new Error(`${this.name} [Merge layer] Dimensions incompatibility using dot mode.`)
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
          `${this.name} [Merge layer] In concat mode, all shapes must be the same except along the concat axis.`
        )
      }
    }
    return true
  }

  /**
   * Method for layer computational logic
   * @param {Tensor[]} inputs
   * @returns {Tensor} `this`
   */
  call(inputs) {
    const valid = this._validateInputs(inputs)
    if (!valid) {
      throw new Error(`${this.name} [Merge layer] Invalid inputs to call method.`)
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
    } else if (['cos', 'dot'].indexOf(this.mode) > -1) {
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
      ops.assigns(output.tensor, 1.0)
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
          gemm(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor)
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          gemm(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0))
        }
      } else {
        throw new Error(`${this.name} [Merge layer] dot mode for 3+ dim tensors not yet implemented.`)
      }
    } else if (this.mode === 'cos') {
      if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
        let a = new Tensor([], output.tensor.shape)
        let b = new Tensor([], output.tensor.shape)
        if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
          gemm(a.tensor, inputs[0].tensor.transpose(1, 0), inputs[0].tensor)
          gemm(b.tensor, inputs[1].tensor.transpose(1, 0), inputs[1].tensor)
          gemm(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor)
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          gemm(a.tensor, inputs[0].tensor, inputs[0].tensor.transpose(1, 0))
          gemm(b.tensor, inputs[1].tensor, inputs[1].tensor.transpose(1, 0))
          gemm(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0))
        }
        ops.muleq(a.tensor, b.tensor)
        ops.sqrteq(a.tensor)
        ops.diveq(output.tensor, a.tensor)
        output.tensor = unsqueeze(output.tensor, 0)
      } else {
        throw new Error(`${this.name} [Merge layer] cos mode for 3+ dim tensors not yet implemented.`)
      }
    }

    return output
  }
}

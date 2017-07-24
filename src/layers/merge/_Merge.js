import Layer from '../../Layer'
import isEqual from 'lodash/isEqual'
import range from 'lodash/range'

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
   * Layer computational logic
   *
   * @param {Tensor[]} inputs
   * @returns {Tensor}
   */
  call(inputs) {
    if (this.gpu) {
      if (!inputs.every(x => x._fromPipeline)) {
        return this._callRegularMode(inputs)
      }
      this._call_gpu(inputs)
    } else {
      const valid = this._validateInputs(inputs)
      if (!valid) {
        throw new Error(`${this.name} [${this.layerClass} layer] Invalid inputs to call method.`)
      }
      this._call_cpu(inputs)
    }
    return this.output
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
          `${this.name} [${this
            .layerClass} layer] In concat mode, all shapes must be the same except along the concat axis.`
        )
      }
    }
    return true
  }
}

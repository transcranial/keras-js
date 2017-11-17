import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import * as tensorUtils from '../../utils/tensorUtils'
import concatFirstAxis from 'ndarray-concat-rows'
import _ from 'lodash'

/**
 * Concatenate merge layer class, extends abstract _Merge class
 */
export default class Concatenate extends _Merge {
  /**
   * Creates a Concatenate merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Concatenate'

    this.mode = 'concat'

    const { axis = -1 } = attrs

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = axis <= 0 ? axis : axis - 1

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = webgl2.compileProgram(require('./Concatenate.glsl'))
    }
  }

  /**
   * CPU call
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {
    const outputShape = inputs[0].tensor.shape.slice()
    const _concatAxis = this.concatAxis < 0 ? outputShape.length + this.concatAxis : this.concatAxis
    inputs.slice(1, inputs.length).forEach(x => {
      const d = x.tensor.shape.slice()[_concatAxis]
      outputShape[_concatAxis] += d
    })
    this.output = new Tensor([], outputShape)

    if (_concatAxis === 0) {
      concatFirstAxis(this.output.tensor, inputs.map(x => x.tensor))
    } else {
      let dimsAxisSwap = [_concatAxis]
      for (let i = 0; i < inputs[0].tensor.shape.length; i++) {
        if (i !== _concatAxis) dimsAxisSwap.push(i)
      }
      concatFirstAxis(
        this.output.tensor.transpose(...dimsAxisSwap),
        inputs.map(x => x.tensor.transpose(...dimsAxisSwap))
      )
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor[]} inputs
   */
  _callGPU(inputs) {
    const outputShape = inputs[0].glTextureShape.slice()
    let _concatAxis = 1
    if (inputs[0].is2DReshaped) {
      if (this.concatAxis === -1 || this.concatAxis === inputs[0].originalShape.length - 1) {
        _concatAxis = 1
      } else {
        this.throwError('specified axis not supported for now.')
      }
    } else {
      if (this.concatAxis === -1 || this.concatAxis === 1) {
        _concatAxis = 1
      } else if (this.concatAxis === -2 || this.concatAxis === 0) {
        _concatAxis = 0
      } else {
        this.throwError('specified axis not supported for now.')
      }
    }

    // create output textures if doesn't already exist
    if (!this.output) {
      outputShape[_concatAxis] = _.sum(inputs.map(input => input.glTextureShape[_concatAxis]))
      this.output = new Tensor([], outputShape)
      this.output.createGLTexture()
      if (inputs[0].is1D) {
        this.output.is1D = inputs[0].is1D
      } else if (inputs[0].is2DReshaped) {
        this.output.is2DReshaped = inputs[0].is2DReshaped
        this.output.originalShape = inputs[0].originalShape.slice()
        const _concatAxis = this.concatAxis < 0 ? this.output.originalShape.length + this.concatAxis : this.concatAxis
        this.output.originalShape[_concatAxis] = _.sum(inputs.map(input => input.originalShape[_concatAxis]))
        this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(
          this.output.originalShape,
          false,
          _concatAxis
        )
      }
    }
    if (!this.runningOutput) {
      this.runningOutput = new Tensor([], outputShape)
      this.runningOutput.createGLTexture()
    }

    const numInputs = inputs.length

    let offsetStart = 0
    let offsetEnd = inputs[0].glTextureShape[_concatAxis]
    for (let i = 0; i < numInputs; i++) {
      // copy output texture to intermediate output
      webgl2.runProgram({
        program: this.copyTextureProgram,
        output: this.runningOutput,
        inputs: [{ texture: this.output.glTexture, type: '2d', name: 'source' }]
      })

      // run merge program
      webgl2.runProgram({
        program: this.mergeProgram,
        output: this.output,
        inputs: [
          { texture: this.runningOutput.glTexture, type: '2d', name: 'runningOutput' },
          { texture: inputs[i].glTexture, type: '2d', name: 'input1' }
        ],
        uniforms: [
          { value: this.output.glTextureShape[0], type: 'int', name: 'rows' },
          { value: this.output.glTextureShape[1], type: 'int', name: 'cols' },
          { value: _concatAxis, type: 'int', name: 'concatAxis' },
          { value: offsetStart, type: 'int', name: 'offsetStart' },
          { value: offsetEnd, type: 'int', name: 'offsetEnd' }
        ]
      })

      if (i < numInputs - 1) {
        offsetStart += inputs[i].glTextureShape[_concatAxis]
        offsetEnd += inputs[i + 1].glTextureShape[_concatAxis]
      }
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D()
      } else if (this.output.is2DSquareReshaped) {
        this.output.reshapeFrom2DSquare()
      }
    }
  }
}

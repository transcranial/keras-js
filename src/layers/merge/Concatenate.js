import _Merge from './_Merge'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import concatFirstAxis from 'ndarray-concat-rows'
import sum from 'lodash/sum'

/**
 * Concatenate merge layer class, extends abstract _Merge class
 */
export default class Concatenate extends _Merge {
  /**
   * Creates a Concatenate merge layer
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
      this.mergeProgram = webgl2.compileProgram(require('./Concatenate.webgl2.glsl'))
    }
  }

  /**
   * CPU call
   * @param {Tensor[]} inputs
   */
  _call_cpu(inputs) {
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
   * @param {Tensor[]} inputs
   */
  _call_gpu(inputs) {
    const outputShape = inputs[0].glTextureShape.slice()
    const _concatAxis = this.concatAxis < 0 ? outputShape.length + this.concatAxis : this.concatAxis

    // create output textures if doesn't already exist
    if (!this.output) {
      outputShape[_concatAxis] = sum(inputs.map(input => input.glTextureShape[_concatAxis]))
      this.output = new Tensor([], outputShape)
      this.output.createGLTexture()
      if (inputs[0].glTextureIsTiled) {
        this.output.glTextureIsTiled = inputs[0].glTextureIsTiled
        this.output.untiledShape = inputs[0].untiledShape
        const _concatAxis = this.concatAxis < 0 ? this.output.untiledShape.length + this.concatAxis : this.concatAxis
        this.output.untiledShape[_concatAxis] = sum(inputs.map(input => input.untiledShape[_concatAxis]))
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
      webgl2.selectProgram(this.copyTextureProgram)
      webgl2.bindOutputTexture(this.runningOutput.glTexture, this.runningOutput.glTextureShape)
      webgl2.bindInputTextures(this.copyTextureProgram, [this.output.glTexture], ['2d'], ['source'])
      webgl2.runProgram()

      // run merge program
      webgl2.selectProgram(this.mergeProgram)
      webgl2.bindOutputTexture(this.output.glTexture, this.output.glTextureShape)
      const uniforms = [...this.output.glTextureShape, _concatAxis, offsetStart, offsetEnd]
      const uniformTypes = ['int', 'int', 'int', 'int', 'int']
      const uniformNames = ['rows', 'cols', 'concatAxis', 'offsetStart', 'offsetEnd']
      webgl2.bindUniforms(this.mergeProgram, uniforms, uniformTypes, uniformNames)
      const textures = [this.runningOutput.glTexture, inputs[i].glTexture]
      const textureTypes = ['2d', '2d']
      const textureNames = ['runningOutput', 'input1']
      webgl2.bindInputTextures(this.mergeProgram, textures, textureTypes, textureNames)
      webgl2.runProgram()

      if (i < numInputs - 1) {
        offsetStart += inputs[i].glTextureShape[_concatAxis]
        offsetEnd += inputs[i + 1].glTextureShape[_concatAxis]
      }
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      if (this.output.glTextureIsTiled) {
        this.output.reshapeTensorFromTiled()
      }
    }
  }
}

import WebGLLayer from '../WebGLLayer'

export default class WebGLConv2D extends WebGLLayer {
  constructor () {
    super()
    this.program = this.webgl.createProgram(
      require('shader-loader!./conv2d.glsl')
    )
  }

  static TEXTURE_UNIFORM_NAME_0 = 'A'
  static TEXTURE_UNIFORM_NAME_1 = 'B_t'
  static TEXTURE_UNIFORM_NAME_2 = 'C'
  static SHARED_LENGTH_UNIFORM_NAME = 'K'
  static COLUMN_COUNT_UNIFORM_NAME = 'N'
  static PAD_UNIFORM_NAME = 'pad'
  static RELU_ACTIVATION_UNIFORM_NAME = 'relu'

  call (t0, t1, t2, activation) {
    if (t1.shape[1] !== t0.shape[1]) {
      throw new Error('Second dimension must be of same size for input Tensors (second Tensor is transposed).')
    }

    this.webgl.selectProgram(this.program)
    const gl = this.webgl.context

    const M = t0.shape[0]
    const N = t1.shape[0]
    const K = t0.shape[1]

    // create an empty output Tensor
    const tOut = new weblas.pipeline.Tensor([M, N], null)

    // bind our input textures containing matrix data
    this._bindInputTexture(t0.texture, gl.TEXTURE0, WebGLConv2D.TEXTURE_UNIFORM_NAME_0)
    this._bindInputTexture(t1.texture, gl.TEXTURE1, WebGLConv2D.TEXTURE_UNIFORM_NAME_1)
    this._bindInputTexture(t2.texture, gl.TEXTURE2, WebGLConv2D.TEXTURE_UNIFORM_NAME_2)

    const kPad = this.webgl.getPad(K)
    const nPad = this.webgl.getPad(N)

    // bind uniforms
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLConv2D.SHARED_LENGTH_UNIFORM_NAME), K + kPad)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLConv2D.COLUMN_COUNT_UNIFORM_NAME), N)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLConv2D.PAD_UNIFORM_NAME), nPad)
    if (activation === 'relu') {
      gl.uniform1i(gl.getUniformLocation(this.program, WebGLConv2D.RELU_ACTIVATION_UNIFORM_NAME), 1)
    }

    // create our destination texture
    this.webgl.bindOutputTexture(M, (N + nPad) / 4, tOut.texture)

    // initiate calculation
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

    this.webgl.unbindInputTexture(gl.TEXTURE0)
    this.webgl.unbindInputTexture(gl.TEXTURE1)
    this.webgl.unbindInputTexture(gl.TEXTURE2)

    return tOut
  }
}

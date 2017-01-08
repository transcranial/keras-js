import WebGLLayer from '../WebGLLayer'

export default class WebGLBatchNorm extends WebGLLayer {
  constructor () {
    super()
    this.program = this.webgl.createProgram(
      require('shader-loader!./batchnorm.glsl')
    )
  }

  static INPUT_TEXTURE_UNIFORM_NAME = 'X'
  static MEAN_UNIFORM_NAME = 'mean'
  static STD_UNIFORM_NAME = 'std'
  static GAMMA_UNIFORM_NAME = 'gamma'
  static BETA_UNIFORM_NAME = 'beta'
  static LENGTH_UNIFORM_NAME = 'N'
  static PAD_UNIFORM_NAME = 'pad'

  call (t0, mean, std, gamma, beta) {
    this.webgl.selectProgram(this.program)
    const gl = this.webgl.context

    const M = t0.shape[0]
    const N = t0.shape[1]

    // create an empty output Tensor
    const tOut = new weblas.pipeline.Tensor([M, N], null)

    const pad = this.webgl.getPad(N)

    // bind our input textures containing matrix data
    this._bindInputTexture(t0.texture, gl.TEXTURE0, WebGLBatchNorm.INPUT_TEXTURE_UNIFORM_NAME)
    this._bindInputTexture(mean.texture, gl.TEXTURE1, WebGLBatchNorm.MEAN_UNIFORM_NAME)
    this._bindInputTexture(std.texture, gl.TEXTURE2, WebGLBatchNorm.STD_UNIFORM_NAME)
    this._bindInputTexture(gamma.texture, gl.TEXTURE3, WebGLBatchNorm.GAMMA_UNIFORM_NAME)
    this._bindInputTexture(beta.texture, gl.TEXTURE4, WebGLBatchNorm.BETA_UNIFORM_NAME)

    // bind uniforms
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLBatchNorm.LENGTH_UNIFORM_NAME), N)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLBatchNorm.PAD_UNIFORM_NAME), pad)

    // create our destination texture
    this.webgl.bindOutputTexture(M, (N + pad) / 4, tOut.texture)

    // initiate calculation
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

    this.webgl.unbindInputTexture(gl.TEXTURE0)

    return tOut
  }
}

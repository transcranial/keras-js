import WebGLLayer from '../WebGLLayer'

export default class WebGLPooling2D extends WebGLLayer {
  constructor (poolingFunc) {
    super()
    if (poolingFunc === 'max') {
      this.program = this.webgl.createProgram(
        require('./maxpooling2d.glsl')
      )
    } else if (poolingFunc === 'average') {
      this.program = this.webgl.createProgram(
        require('./avgpooling2d.glsl')
      )
    } else {
      throw new Error(`[WebGLPooling2D] pooling function must be max or average.`)
    }
  }

  static INPUT_TEXTURE_UNIFORM_NAME = 'X'
  static POOL_IMAP_TEXTURE_UNIFORM_NAME = 'poolIndexMapping'
  static PAD_UNIFORM_NAME = 'pad'
  static CHANNELS_UNIFORM_NAME = 'channels'
  static POOL_ELEMENTS_UNIFORM_NAME = 'poolElements'

  call (input, poolIndexMapping) {
    this.webgl.selectProgram(this.program)
    const gl = this.webgl.context

    const outputLength = poolIndexMapping.shape[0]
    const poolElements = poolIndexMapping.shape[1]
    const channels = input.shape[1]

    // create an empty output Tensor
    const output = new weblas.pipeline.Tensor([outputLength, channels], null)

    // bind our input textures containing matrix data
    this._bindInputTexture(this.program, input.texture, gl.TEXTURE0, WebGLPooling2D.INPUT_TEXTURE_UNIFORM_NAME)
    this._bindInputTexture(this.program, poolIndexMapping.texture, gl.TEXTURE1, WebGLPooling2D.POOL_IMAP_TEXTURE_UNIFORM_NAME)

    const pad = this.webgl.getPad(channels)

    // bind uniforms
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.PAD_UNIFORM_NAME), pad)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.CHANNELS_UNIFORM_NAME), channels)
    gl.uniform1i(gl.getUniformLocation(this.program, WebGLPooling2D.POOL_ELEMENTS_UNIFORM_NAME), poolElements)

    // create our destination texture
    this.webgl.bindOutputTexture(outputLength, (channels + pad) / 4, output.texture)

    // initiate calculation
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

    this.webgl.unbindInputTexture(gl.TEXTURE0)
    this.webgl.unbindInputTexture(gl.TEXTURE1)

    return output
  }
}

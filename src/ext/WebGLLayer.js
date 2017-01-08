export default class WebGLLayer {
  constructor () {
    this.webgl = weblas.gpu.gl
  }

  /**
   * Create a texture from the given texel data and bind it to our shader program.
   *
   * texture - packed texels data
   * textureUnit - the texture unit to bind to (gl.TEXTURE0, gl.TEXTURE1, etc)
   * name - the uniform name to associate with (must match shader program)
   */
  _bindInputTexture (texture, textureUnit, name) {
    const gl = this.webgl.context

    gl.activeTexture(textureUnit)
    gl.bindTexture(gl.TEXTURE_2D, texture)

    const sampler = gl.getUniformLocation(this.program, name)
    gl.uniform1i(sampler, textureUnit - gl.TEXTURE0)
  }
}

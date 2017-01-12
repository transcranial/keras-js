export default class WebGLLayer {
  constructor () {
    this.webgl = weblas.gpu.gl
  }

  /**
   * Bind WebGL input texture.
   *
   * @param {WebGLProgram} program - shader program
   * @param {WebGLTexture} texture - texels data
   * @param {number} textureUnit - e.g., gl.TEXTURE0
   * @param {string} name - uniform name in shader program
   */
  _bindInputTexture (program, texture, textureUnit, name) {
    const gl = this.webgl.context

    gl.activeTexture(textureUnit)
    gl.bindTexture(gl.TEXTURE_2D, texture)

    const sampler = gl.getUniformLocation(program, name)
    gl.uniform1i(sampler, textureUnit - gl.TEXTURE0)
  }
}

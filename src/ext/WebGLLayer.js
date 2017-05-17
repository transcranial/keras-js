export default class WebGLLayer {
  constructor() {
    this.webgl = weblas.gpu.gl
    this.numTextures = 8
  }

  MAX_NUM_TEXTURES = 8

  /**
   * Bind WebGL input texture.
   *
   * @param {WebGLProgram} program - shader program
   * @param {WebGLTexture} texture - texels data
   * @param {number} textureUnit - e.g., gl.TEXTURE0
   * @param {string} name - uniform name in shader program
   */
  _bindInputTexture(program, texture, textureUnit, name) {
    const gl = this.webgl.context

    gl.activeTexture(textureUnit)
    gl.bindTexture(gl.TEXTURE_2D, texture)

    const sampler = gl.getUniformLocation(program, name)
    gl.uniform1i(sampler, textureUnit - gl.TEXTURE0)
  }

  /**
   * Runs WebGL fragment shader program to perform computation.
   */
  _compute() {
    const gl = this.webgl.context
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)
  }

  /**
   * Clean-up: unbind WebGL input textures.
   */
  _unbindInputTextures() {
    const gl = this.webgl.context
    for (let i = 0; i < this.numTextures; i++) {
      this.webgl.unbindInputTexture(gl.TEXTURE0 + i)
    }
  }
}

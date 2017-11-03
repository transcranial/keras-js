class WebGL2 {
  constructor() {
    this.isSupported = false

    this.vertexShader = null
    this.textureUnitMap = null
    this.textureUnitCounter = 0

    if (typeof window !== 'undefined') {
      this.canvas = document.createElement('canvas')
      this.context = this.canvas.getContext('webgl2')

      const gl = this.context
      if (gl) {
        this.isSupported = true
        gl.getExtension('EXT_color_buffer_float')
        this.MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE)
        this.MAX_TEXTURE_IMAGE_UNITS = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
        this.init()
      } else {
        console.log('Unable to initialize WebGL2 -- your browser may not support it.')
      }
    }
  }

  /**
   * Intialization after WebGL2 detected.
   */
  init() {
    this.textureUnitMap = new Map()
    this.createCommonVertexShader()
  }

  /**
   * Creates and compiles passthrough vertex shader that we will attach
   * to all our programs.
   */
  createCommonVertexShader() {
    const gl = this.context

    const source = require('./vertexShader.webgl2.glsl')

    const vertexShader = gl.createShader(gl.VERTEX_SHADER)
    gl.shaderSource(vertexShader, source)
    gl.compileShader(vertexShader)

    const success = gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)
    if (!success) {
      console.error(gl.getShaderInfoLog(vertexShader))
      gl.deleteShader(vertexShader)
      this.isSupported = false
    }

    this.vertexShader = vertexShader
  }

  /**
   * Compiles fragment shader from source and creates program from it,
   * using our passthrough vertex shader.
   *
   * @param {string} source - fragment shader GLSL source code
   * @returns {WebGLProgram}
   */
  compileProgram(source) {
    const gl = this.context

    // create and compile fragment shader
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)
    gl.shaderSource(fragmentShader, source)
    gl.compileShader(fragmentShader)

    let success = gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)
    if (!success) {
      console.error(gl.getShaderInfoLog(fragmentShader))
      gl.deleteShader(fragmentShader)
      this.isSupported = false
    }

    // create program and attach compiled shaders
    const program = gl.createProgram()
    gl.attachShader(program, this.vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    success = gl.getProgramParameter(program, gl.LINK_STATUS)
    if (!success) {
      console.error(gl.getProgramInfoLog(program))
      this.isSupported = false
    }

    this.setupVertices(program)
    return program
  }

  /**
   * Setup vertices
   *
   * @param {WebGLProgram} program
   */
  setupVertices(program) {
    const gl = this.context

    const position = gl.getAttribLocation(program, 'position')
    const positionVertexObj = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionVertexObj)

    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]),
      gl.STATIC_DRAW
    )
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0)
    gl.enableVertexAttribArray(position)

    const texcoord = gl.getAttribLocation(program, 'texcoord')
    const texcoordVertexObj = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, texcoordVertexObj)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), gl.STATIC_DRAW)
    gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 0, 0)
    gl.enableVertexAttribArray(texcoord)

    const indicesVertexObj = gl.createBuffer()
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indicesVertexObj)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW)
  }

  /**
   * Selects linked program as active program
   *
   * @param {WebGLProgram} program
   */
  selectProgram(program) {
    const gl = webgl2.context
    gl.useProgram(program)
  }

  /**
   * Bind uniforms within program
   *
   * @param {WebGLProgram} program
   * @param {*[]} values
   * @param {string[]} types
   * @param {string[]} names
   */
  bindUniforms(program, values, types, names) {
    const gl = webgl2.context

    values.forEach((val, i) => {
      const loc = gl.getUniformLocation(program, names[i])
      if (types[i] === 'float') {
        gl.uniform1f(loc, val)
      } else if (types[i] === 'int' || types[i] === 'bool') {
        gl.uniform1i(loc, val)
      }
    })
  }

  /**
   * Bind input textures within program
   *
   * @param {WebGLProgram} program
   * @param {WebGLTexture[]} textures
   * @param {string[]} types
   * @param {string[]} names
   */
  bindInputTextures(program, textures, types, names) {
    const gl = webgl2.context

    const targetMap = {
      '2d': gl.TEXTURE_2D,
      '2d_array': gl.TEXTURE_2D_ARRAY,
      '3d': gl.TEXTURE_3D
    }

    textures.forEach((tex, i) => {
      gl.activeTexture(gl.TEXTURE0 + i)
      gl.bindTexture(targetMap[types[i]], tex)
      gl.uniform1i(gl.getUniformLocation(program, names[i]), i)
    })
  }

  /**
   * Bind output texture
   *
   * @param {WebGLTexture} outputTexture
   * @param {number[]} shape
   */
  bindOutputTexture(outputTexture, shape) {
    const gl = this.context

    gl.viewport(0, 0, shape[1], shape[0])

    this.framebuffer = this.framebuffer || gl.createFramebuffer()

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0)
  }

  /**
   * Runs fragment shader program
   */
  runProgram() {
    const gl = this.context
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)
  }

  /**
   * Reads pixel data from framebuffer
   *
   * @param {number[]} shape
   * @returns {Float32Array}
   */
  readData(shape) {
    const gl = this.context
    const buf = new ArrayBuffer(shape[0] * shape[1] * 4 * 4)
    const view = new Float32Array(buf)
    gl.readPixels(0, 0, shape[1], shape[0], gl.RGBA, gl.FLOAT, view)
    const out = []
    for (let i = 0; i < view.length; i += 4) {
      out.push(view[i])
    }
    return new Float32Array(out)
  }
}

const webgl2 = new WebGL2()
const MAX_TEXTURE_SIZE = webgl2.MAX_TEXTURE_SIZE
const MAX_TEXTURE_IMAGE_UNITS = webgl2.MAX_TEXTURE_IMAGE_UNITS

export { webgl2, MAX_TEXTURE_SIZE, MAX_TEXTURE_IMAGE_UNITS }

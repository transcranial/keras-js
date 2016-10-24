const canvas = document.createElement('canvas')
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')

if (!gl) {
  throw new Error('Unable to initialize WebGL. Your browser may not support it.')
}

export const MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE)

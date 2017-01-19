const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

let MAX_TEXTURE_SIZE = 16384;
if (gl) {
  MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
} else {
  console.log('Unable to initialize WebGL. Your browser may not support it.');
}

export { MAX_TEXTURE_SIZE };

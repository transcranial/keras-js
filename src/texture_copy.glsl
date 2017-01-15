// Copy texture
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D source;

void main(void) {
  gl_FragColor = texture2D(source, vec2(outTex.x, outTex.y));
}

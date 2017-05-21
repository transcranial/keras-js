precision highp float;

varying vec2 outTex;
uniform sampler2D X;
uniform int relu;

void main(void) {
  if (relu == 1) {
    gl_FragColor = max(texture2D(X, vec2(outTex.x, outTex.y)), 0.0);
  } else {
    gl_FragColor = texture2D(X, vec2(outTex.x, outTex.y));
  }
}

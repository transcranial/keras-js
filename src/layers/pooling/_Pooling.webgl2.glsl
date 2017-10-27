#version 300 es
precision mediump float;
precision mediump isampler2D;

in vec2 outTex;
uniform sampler2D x;
uniform isampler2D indexMap;
uniform int rows;
uniform int cols;
uniform int poolSize;
uniform bool isMaxPooling;
out vec4 outColor;

void main() {
  int out_x = int(float(cols) * outTex.x);
  int out_y = int(float(rows) * outTex.y);

  float val = 0.;
  int count = 0;
  for (int i = 0; i < poolSize; ++i) {
    int poolIndex = texelFetch(indexMap, ivec2(i, out_y), 0).r;
    if (poolIndex != -1) {
      float val2 = texelFetch(x, ivec2(out_x, poolIndex), 0).r;
      if (isMaxPooling) {
        if (count == 0 || val2 > val) {
          val = val2;
        }
      } else {
        val += val2;
      }
      count += 1;
    }
  }

  if (!isMaxPooling) {
    val /= float(count);
  }

  outColor = vec4(val);
}

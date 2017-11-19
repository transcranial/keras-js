#version 300 es
precision highp float;
precision highp sampler2DArray;
precision highp isampler2D;

in vec2 outTex;
uniform sampler2DArray x;
uniform isampler2D poolIndexMap;
uniform isampler2D fragmentIndexMap;
uniform int channels;
uniform int poolSize;
uniform bool isMaxPooling;
out vec4 outColor;

void main() {
  int out_x = int(float(channels) * outTex.x);
  int out_y = int(float(textureSize(poolIndexMap, 0)[1]) * outTex.y);

  float val = 0.;
  int count = 0;
  for (int i = 0; i < poolSize; ++i) {
    int poolIndex = texelFetch(poolIndexMap, ivec2(i, out_y), 0).r;
    int fragmentIndex = texelFetch(fragmentIndexMap, ivec2(i, out_y), 0).r;
    if (poolIndex != -1 && fragmentIndex != -1) {
      poolIndex = int(mod(float(poolIndex), float(textureSize(x, 0)[1])));
      float val2 = texelFetch(x, ivec3(out_x, poolIndex, fragmentIndex), 0).r;
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

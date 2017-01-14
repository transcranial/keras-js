// 2D Max Pooling op.
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D X;
uniform sampler2D poolIndexMapping;
uniform int inputRows;
uniform int channels;
uniform int channelsPad;
uniform int poolElements;
uniform int poolElementsPad;

float select_index(vec4 v, int index) {
  float val = 0.0;
  if (index == 0) {
    val = v.r;
  } else if (index == 1) {
    val = v.g;
  } else if (index == 2) {
    val = v.b;
  } else if (index == 3) {
    val = v.a;
  }
  return val;
}

void fix_pad(inout vec4 v, int pad) {
  v.a = 0.0;
  if (pad == 2) {
    v.b = 0.0;
  } else if (pad == 3) {
    v.b = 0.0;
    v.g = 0.0;
  }
}

void main(void) {
  // index of first element in pixel (matrix space)
  float col = floor(outTex.x * float(channels + channelsPad) - 1.5);

  float poolIndexCoordX;
  vec4 poolIndices;
  int poolIndexRGBA;
  float poolIndex;
  vec4 mappedValues;
  float inputCoordY;
  const float min = -1.0e+08;
  vec4 currentMax = vec4(min, min, min, min);
  for (int i = 0; i < 100; i += 1) {
    if (i >= poolElements) {
      break;
    }

    poolIndexCoordX = (float(i) + 0.5) / float(poolElements + poolElementsPad);
    poolIndices = texture2D(poolIndexMapping, vec2(poolIndexCoordX, outTex.y));
    poolIndexRGBA = int(mod(float(i), 4.0));
    poolIndex = select_index(poolIndices, poolIndexRGBA);

    if (poolIndex != -1.0) {
      inputCoordY = (poolIndex + 0.5) / float(inputRows);
      mappedValues = texture2D(X, vec2(outTex.x, inputCoordY));
    }

    currentMax = max(currentMax, mappedValues);
  }

  // set pad values to 0.0, if in padded region of output texture
  if (channelsPad > 0 && col + 4.0 > float(channels)) {
    fix_pad(mappedValues, channelsPad);
  }

  gl_FragColor = currentMax;
}

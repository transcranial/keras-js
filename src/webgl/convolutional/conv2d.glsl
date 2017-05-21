// 2D convolution fragment shader - based on im2col + gemm implementation
// The input texture, X, is already configured as column matrix, after
// input_transform.glsl is run on it if necessary. The output texture is in column
// matrix configuration, and will need to be reshaped or transformed prior to the
// next layer.

// The following code is adapted from weblas, specifically the sgemm parts.
// https://github.com/waylonflinn/weblas
//
// The MIT License (MIT)
//
// Copyright (c) 2015 Waylon Flinn
// Modified by Leon Chen, 2017
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

precision highp float;

varying vec2 outTex;
uniform sampler2D X;
uniform sampler2D W;
uniform sampler2D b;
uniform int inputCols;
uniform int outputCols;
uniform int inputColPad;
uniform int outputColPad;
uniform int relu;

// sum of products between elements in row i (from X) x col j (from B)
//
// Calculate the dot product between the row (from X) and column (from B)
// identified by the passed indeces (output texture coordinate space).
// We loop over elements in the row and column and sum the product
// using the glsl `dot` function to process four elements at a time.
// This four element optimization requires that the matrix B be
// transposed before texel packing and that both matrices be padded
// (with zeros) to a multiple of four (4) in their shared dimension.
float dot_rowcol (float y, float x, sampler2D X, sampler2D W, int K) {
  float delta_t = 1. / float(K); // space (on texture) between elements
  float sum = 0.; // sum for this row/column pair
  float z = 0.5 * (4.0 * delta_t);// position for shared dimension on source textures

  for (int l = 0; l < 4096; ++l) {
    if (l >= K / 4) break; // stop when we finish the row/column
    // l is in pixel space, so we divide by four

    // retrieve next four elements from each texture
    vec4 a_ik = texture2D(X, vec2(z, y));
    vec4 b_kj = texture2D(W, vec2(z, x));

    // use `dot` to process four elements at a time
    sum += dot(a_ik, b_kj);
    z += (4.0 * delta_t); // (z + 0.5)*delta
  }
  return sum;
}

void main(void) {

  // get the implied row and column from .y and .x of passed (output)
  // texture coordinate. These map directly to input texture space when
  // the relevant dimensions are the same.
  float row_t = outTex.y;
  float col_t = outTex.x;
  vec4 b_v = texture2D(b, vec2(col_t, 0.5));

  vec4 sum_v = vec4(0.0, 0.0, 0.0, 0.0);
  float col = (col_t * float(outputCols + outputColPad) - 2.0); // index of first element in pixel (matrix space)
  sum_v.r = dot_rowcol(row_t, (col + 0.5) / float(outputCols), X, W, inputCols + inputColPad);
  // in the padding region?
  if (outputColPad > 0 && (col + 4.0) > float(outputCols)) {
    // pad
    if (outputColPad < 3) {
      sum_v.g = dot_rowcol(row_t, (col + 1.5) / float(outputCols), X, W, inputCols + inputColPad);
    }
    if (outputColPad < 2) {
      sum_v.b = dot_rowcol(row_t, (col + 2.5) / float(outputCols), X, W, inputCols + inputColPad);
    }
  } else {
    sum_v.g = dot_rowcol(row_t, (col + 1.5) / float(outputCols), X, W, inputCols + inputColPad);
    sum_v.b = dot_rowcol(row_t, (col + 2.5) / float(outputCols), X, W, inputCols + inputColPad);
    sum_v.a = dot_rowcol(row_t, (col + 3.5) / float(outputCols), X, W, inputCols + inputColPad);
  }

  if (relu == 1) {
    gl_FragColor = max(sum_v + b_v, 0.0);
  } else {
    gl_FragColor = sum_v + b_v;
  }
}

// fragment shader that calculates the matrix product and writes each
// element to a pixel component in a floating point texture.
// the output RGBA canvas.
// readPixel is used to read the bytes.

precision highp float;

varying vec2 outTex; // texture coords of row/column to calculate
uniform sampler2D A; // texture with data from padded A
uniform sampler2D B_t; // texture with data from padded transpose of B
uniform sampler2D C; // texture with data from C
uniform int K; // number of elements in shared dimension
uniform int N; // number of columns in output
uniform int pad; // additional columns to nearest multiple of four
uniform int relu;

// sum of products between elements in row i (from A) x col j (from B)
//
// Calculate the dot product between the row (from A) and column (from B)
// identified by the passed indeces (output texture coordinate space).
// We loop over elements in the row and column and sum the product
// using the glsl `dot` function to process four elements at a time.
// This four element optimization requires that the matrix B be
// transposed before texel packing and that both matrices be padded
// (with zeros) to a multiple of four (4) in their shared dimension.
float dot_rowcol (float y, float x, sampler2D A, sampler2D B_t, int K) {
  float delta_t = 1. / float(K); // space (on texture) between elements
  float sum = 0.; // sum for this row/column pair
  float z = 0.5 * (4.0 * delta_t);// position for shared dimension on source textures

  for (int l = 0; l < 4096; ++l) {
    if (l >= K / 4) break; // stop when we finish the row/column
    // l is in pixel space, so we divide by four

    // retrieve next four elements from each texture
    vec4 a_ik = texture2D(A, vec2(z, y));
    vec4 b_kj = texture2D(B_t, vec2(z, x));

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
  vec4 c_v = texture2D(C, vec2(col_t, 0.5));

  vec4 sum_v = vec4(0.0, 0.0, 0.0, 0.0);
  float col = (col_t * float(N + pad) - 2.0); // index of first element in pixel (matrix space)
  sum_v.r = dot_rowcol(row_t, (col + 0.5) / float(N), A, B_t, K);
  // in the padding region?
  if (pad > 0 && (col + 4.0) > float(N)) {
    // pad
    if (pad < 3) {
      sum_v.g = dot_rowcol(row_t, (col + 1.5) / float(N), A, B_t, K);
    }
    if (pad < 2) {
      sum_v.b = dot_rowcol(row_t, (col + 2.5) / float(N), A, B_t, K);
    }
  } else {
    sum_v.g = dot_rowcol(row_t, (col + 1.5) / float(N), A, B_t, K);
    sum_v.b = dot_rowcol(row_t, (col + 2.5) / float(N), A, B_t, K);
    sum_v.a = dot_rowcol(row_t, (col + 3.5) / float(N), A, B_t, K);
  }

  if (relu == 1) {
    gl_FragColor = max(sum_v + c_v, 0.0);
  } else {
    gl_FragColor = sum_v + c_v;
  }
}

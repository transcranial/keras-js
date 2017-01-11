precision highp float;

varying vec2 outTex; // texture coords of row/column to calculate
uniform sampler2D A;
uniform sampler2D indexMappingRow;
uniform sampler2D indexMappingCol;
uniform int N; // number of columns in output
uniform int pad; // additional columns to nearest multiple of four

void main(void) {
  float row_t = outTex.y;
  float col_t = outTex.x;
	float col = (col_t * float(N + pad) - 2.0); // index of first element in pixel (matrix space)

  vec4 mapped_row_t = texture2D(indexMappingRow, vec2(col_t, row_t));
  vec4 mapped_col_t = texture2D(indexMappingCol, vec2(col_t, row_t));

  vec4 mapped_input_val = vec4(0.0, 0.0, 0.0, 0.0);
  mapped_input_val.r = texture2D(A, vec2(mapped_col_t.r, mapped_row_t.r)).r;
  if (pad > 0 && (col + 4.0) > float(N)) {
    if (pad < 3) {
      mapped_input_val.g = texture2D(A, vec2(mapped_col_t.g, mapped_row_t.g)).g;
    }
    if (pad < 2) {
      mapped_input_val.b = texture2D(A, vec2(mapped_col_t.b, mapped_row_t.b)).b;
    }
  } else {
    mapped_input_val.g = texture2D(A, vec2(mapped_col_t.g, mapped_row_t.g)).g;
    mapped_input_val.b = texture2D(A, vec2(mapped_col_t.b, mapped_row_t.b)).b;
    mapped_input_val.a = texture2D(A, vec2(mapped_col_t.a, mapped_row_t.a)).a;
  }

  gl_FragColor = mapped_input_val;
}

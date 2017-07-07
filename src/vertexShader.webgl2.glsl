#version 300 es
precision highp float;

in vec3 position;
in vec2 texcoord;
out vec2 out_tex;

void main () {
  gl_Position = vec4(position, 1.0);
	out_tex = texcoord;
}

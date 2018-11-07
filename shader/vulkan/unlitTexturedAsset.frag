#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable


layout(binding = 11) uniform sampler2D texSampler;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTc;
layout (location = 2) in vec3 inViewVec;
layout (location = 3) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec3 color = texture(texSampler, inTc).rgb;
	outFragColor = vec4(color, 1.0);
}

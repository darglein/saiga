#version 450

#ifndef POINT_SIZE
#define POINT_SIZE 5.0f
#endif

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inNormal;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec4 inData;



layout (binding = 7) uniform UBO2 
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
} ubo;


layout (push_constant) uniform PushConstants {
	mat4 model;
} pushConstants;

layout(location=0) out VertexData
{
  vec3 color;
} outData;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main() 
{
	outData.color = vec3(inColor);
        vec4 vp = ubo.view * pushConstants.model * vec4(inPos.xyz, 1.0);
        gl_Position = ubo.projection * vp;

        // scale pointsize with inverse distance to camera
        gl_PointSize = 10 * float(POINT_SIZE) / length(vec3(vp));
}

#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inNormal;
layout (location = 2) in vec2 inTc;
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

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outTc;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;

out gl_PerVertex
{
	vec4 gl_Position;
};

void main() 
{
	outNormal = vec3(inNormal);
	outTc = inTc;
	gl_Position = ubo.projection * ubo.view * pushConstants.model * vec4(inPos.xyz, 1.0);
	
	vec4 pos = ubo.view * pushConstants.model * vec4(vec3(inPos), 1.0);
	outNormal = mat3(ubo.view * pushConstants.model) * vec3(inNormal);
	vec3 lPos = mat3(ubo.view) * ubo.lightPos.xyz;
	outLightVec = lPos;//lPos;// - pos.xyz;
	outViewVec = -pos.xyz;		
}

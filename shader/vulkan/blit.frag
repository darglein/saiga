#version 450

#extension GL_GOOGLE_include_directive : require



layout(binding = 11) uniform sampler2D colorTexture;

layout (location = 0) out vec4 outColor;

layout(location=0) in VertexData
{
    vec2 tc;
} inData;


void main() 
{
    outColor = texture(colorTexture,inData.tc);
}



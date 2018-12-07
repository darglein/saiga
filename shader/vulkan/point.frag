#version 450

//layout (location = 1) in vec3 inColor;

layout(location=0) in VertexData
{
  vec3 color;
} inData;


layout (location = 0) out vec4 outFragColor;

void main() 
{   
        outFragColor = vec4(inData.color, 1.0);
}

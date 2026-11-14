#version 450


layout(location=0) in vec4 inPosition;
layout(location=1) in vec4 inNormal;
layout(location=2) in vec4 inColor;
layout(location=3) in vec4 inData;


out gl_PerVertex
{
    vec4 gl_Position;
};

layout(location=0) out VertexData
{
    vec2 tc;
} outData;

void main() 
{
    // Vulkan NDC: y = -1 is the top of the viewport. Texture2D::fromImage uploads with flipY = true (default), so the
    // first texture row (tc.y = 0) is the bottom row of the image. Therefore tc.y = 1 at the top shows the image
    // (stored top row first) upright.
    outData.tc   = inPosition.xy * 0.5f + 0.5f;
    outData.tc.y = 1.0 - outData.tc.y;
    gl_Position = vec4(inPosition.x,inPosition.y,0,1);


    gl_Position.z = 0.5;
}

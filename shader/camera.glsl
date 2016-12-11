

layout (std140) uniform cameraData
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 camera_position;
};


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;


out vec2 texCoord;
out vec4 pos;

out vec3 viewDir;
out vec4 eyePos;

void main() {
    texCoord = in_tex;
    pos = vec4(in_position,1);
    gl_Position = vec4(in_position,1);


    mat4 invView = inverse(view);
    vec4 worldPos = inverse(proj) * pos;
    worldPos /= worldPos.w;
    worldPos =  invView * worldPos;


    eyePos = invView[3];
    viewDir = vec3(worldPos-eyePos);
}



##GL_FRAGMENT_SHADER

#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 params;

in vec2 texCoord;
in vec4 pos;
in vec3 viewDir;
in vec4 eyePos;

layout(location=0) out vec4 out_color;

void main() {
    float horizonHeight = params.x;
    float skyboxDistance = params.y;


    //todo render without srgb
    vec4 darkBlueSky = vec4(43,99,192,255) / 255.0f;
    darkBlueSky = pow(darkBlueSky,vec4(2.2f));

    vec4 blueSky = vec4(97,161,248,255) / 255.0f;
    blueSky = pow(blueSky,vec4(2.2f));

    vec4 lightBlueSky = vec4(177,212,254,255) / 255.0f;
    lightBlueSky = pow(lightBlueSky,vec4(2.2f));



    //direction of current viewing ray
    vec3 dir = normalize(viewDir);

    //intersection point of viewing ray with cylinder around viewer with radius=skyboxDistance
    vec3 skyboxPos = vec3(eyePos) + dir * (skyboxDistance / length(vec2(dir.x,dir.z))) - horizonHeight;

    //this gives the tangens of the viewing ray towards the ground
    float h = skyboxPos.y/skyboxDistance;

    //exponential gradient
    float a = -exp(-h*3)+1;


    out_color = mix(blueSky,darkBlueSky,a);

    //fade out bottom border to black
    if(h<0)
        out_color = mix(blueSky,vec4(0),-h*100.0f);

}



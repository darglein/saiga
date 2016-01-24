
##GL_VERTEX_SHADER

#version 400
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

void main() {
    texCoord = in_tex;
    pos = vec4(in_position,1);
    gl_Position = vec4(in_position,1);
}



##GL_FRAGMENT_SHADER

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

in vec2 texCoord;
in vec4 pos;

layout(location=0) out vec4 out_color;

void main() {
    vec4 darkBlueSky = vec4(43,99,192,255) / 255.0f;
    darkBlueSky = pow(darkBlueSky,vec4(2.2f));

    vec4 blueSky = vec4(97,161,248,255) / 255.0f;
    blueSky = pow(blueSky,vec4(2.2f));

    vec4 lightBlueSky = vec4(177,212,254,255) / 255.0f;
    lightBlueSky = pow(lightBlueSky,vec4(2.2f));




    mat4 invView = inverse(view);
    vec4 worldPos = inverse(proj) * pos;
    worldPos /= worldPos.w;
    worldPos =  invView * worldPos;

    vec4 eyePos = invView[3];

    vec3 dir = normalize(vec3(worldPos)-vec3(eyePos));
//    vec3 dir = normalize(vec3(worldPos));

    float cosAlpha = dir.y;
    float alpha = acos(cosAlpha);
    out_color =  blueSky;

//    out_color = mix(colorBottom,colorTop,1.0f-alpha/(3.141f/2.0f));
//    out_color = mix(blueSky,darkBlueSky,cosAlpha);

    if(cosAlpha>0.5f)
         out_color =  darkBlueSky;

    if(cosAlpha<0)
        out_color =  vec4(0,0,0,1);

//    out_color = worldPos;
}



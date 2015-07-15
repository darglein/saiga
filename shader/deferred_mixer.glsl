##start
##vertex

#version 150
#extension GL_ARB_explicit_attrib_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

out vec2 tc;

void main() {
    tc = in_tex;
//    gl_Position = vec4(in_position,1);
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}


##end

##start
##fragment

#version 150

in vec2 tc;

uniform mat4 view;


uniform sampler2D deferred_diffuse;
uniform sampler2D deferred_normal;
uniform sampler2D deferred_depth;
uniform sampler2D deferred_position;

out vec4 color;



void main(){


    vec4 diffColor = texture( deferred_diffuse, tc );
    vec4 position = texture( deferred_position, tc );
    float depth = texture( deferred_depth, tc ).x;
    vec3 normal = texture( deferred_normal, tc ).xyz;
    normal = normal*2.0f - 1.0f;




    vec4 ambColor = diffColor;
    vec4 specColor = vec4(1)*0.5;

    vec3 lightdir = normalize(vec3(-1,-3,-2));
            lightdir = -vec3(view*vec4(lightdir,0));
//    //directional light
    vec3 L = normalize(lightdir);
    vec3 vertexMV = vec3(position);
    vec3 E = normalize(-vertexMV);
    vec3 R = normalize(-reflect(L,normal));
    vec3 N = normalize(normal);
    N = normalize(N);
    //calculate Ambient Term:
    vec4 Iamb = ambColor*0.2;

    //calculate Diffuse Term:
    vec4 Idiff = diffColor * max(dot(N,L), 0.0);
    Idiff = clamp(Idiff, 0.0, 1.0);

    // calculate Specular Term:
    vec4 Ispec = specColor* pow(max(dot(R,E),0.0),10);
    Ispec = clamp(Ispec, 0.0, 1.0);

    color = vec4(Iamb+ Idiff + Ispec);

}

##end

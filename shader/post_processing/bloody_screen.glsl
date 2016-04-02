#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 400

uniform float intensity;

#include "post_processing_helper_fs.glsl"


float intensityRect(vec2 tc){
    vec2 d = abs(tc-vec2(0.5f));



    return clamp(max(d.x,d.y),0,1);
//    return clamp(d.x+d.y,0,1);
//    return clamp(d.y,0,1);
}

// float intensityCircle(vec2 tc){

	 // vec2 d = (abs(tc-vec2(0.5f)));

	// d.y*=0.8f;
	
	// float l = length(d)-0.34f;
	
	// float v = l;
    // return clamp(v,0,1);
// }


void main() {

    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec3 c = texelFetch( image, tci ,0).rgb;

//    vec3 c = texture( image, tc ).rgb;
	
	float a = intensityRect(tc);
	a = smoothstep(0.3,0.5,a);
    out_color = vec4(c + intensity*vec3(a,0,0),1);
}



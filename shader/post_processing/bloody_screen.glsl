
##GL_VERTEX_SHADER

#version 400
layout(location=0) in vec3 in_position;


out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc*0.5f+0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 400


uniform sampler2D image;

uniform vec4 screenSize;
uniform float intensity;


in vec2 tc;

layout(location=0) out vec3 out_color;


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


    vec3 c = texture( image, tc ).rgb;
	
	float a = intensityRect(tc);
	a = smoothstep(0.3,0.5,a);
    out_color = c + intensity*vec3(a,0,0);
}



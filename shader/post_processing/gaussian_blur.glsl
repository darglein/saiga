##start
##vertex

#version 400
layout(location=0) in vec3 in_position;


out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc*0.5f+0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}


##end

##start
##fragment

#version 400


uniform sampler2D image;

uniform vec4 screenSize;

in vec2 tc;

layout(location=0) out vec3 out_color;

const float gauss[9] = float[](0.077847,	0.123317,	0.077847,
				 0.123317,	0.195346,	0.123317,
				  0.077847,	0.123317,	0.077847);

const float sobelx[9] = float[](1,0,-1,
				 2,0,-2,
				 1,0,-1);

const float sobely[9] = float[](1,2,1,
				 0,0,0,
				 -1,-2,-1);




vec3 filter3x3(const float filter[9]){
    vec3 color;

    color += filter[0]*texture( image, tc + vec2(-screenSize.z,-screenSize.w) ).rgb;
    color += filter[1]*texture( image, tc + vec2(0,-screenSize.w) ).rgb;
    color += filter[2]*texture( image, tc + vec2(screenSize.z,-screenSize.w) ).rgb;

    color += filter[3]*texture( image, tc + vec2(-screenSize.z,0) ).rgb;
    color += filter[4]*texture( image, tc + vec2(0,0) ).rgb;
    color += filter[5]*texture( image, tc + vec2(screenSize.z,0) ).rgb;

    color += filter[6]*texture( image, tc + vec2(-screenSize.z,screenSize.w) ).rgb;
    color += filter[7]*texture( image, tc + vec2(0,screenSize.w) ).rgb;
    color += filter[8]*texture( image, tc + vec2(screenSize.z,screenSize.w) ).rgb;

    return color;
}

void main() {



//    out_color = texture( image, tc  ).rgb;
//     out_color = filter3x3(sobelx) + filter3x3(sobely);
    out_color = filter3x3(gauss);
}

##end

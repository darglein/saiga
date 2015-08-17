##start
##vertex

#version 150
#extension GL_ARB_explicit_attrib_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;

out vec3 normal;
out vec3 normalW;
out vec3 vertexMV;
out vec3 vertex;
out vec2 texCoord;
uniform float wobble;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    normalW = normalize(vec3(model * vec4( in_normal, 0 )));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = proj*view *model* vec4(in_position,1);
}


##end

##start
##fragment

#version 150
#extension GL_ARB_explicit_attrib_location : enable
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler2D image;
uniform float highlight;
uniform float wobble;

in vec3 normal;
in vec3 normalW;
in vec3 vertexMV;
in vec3 vertex;
in vec2 texCoord;

layout(location=0) out vec4 out_color;

const float C_PI    = 3.1415;
const float C_2PI   = 2.0 * C_PI;
const float C_2PI_I = 1.0 / (2.0 * C_PI);
const float C_PI_2  = C_PI / 2.0;

vec3 blackAndWhite(vec3 color){
//    float light = dot(color,vec3(1))/3.0f;
    float light = dot(color,vec3(0.21f,0.72f,0.07f));
    return vec3(light);
}

vec2 Distort(vec2 p)
{
    float theta  = atan(p.y, p.x);
    float radius = length(p);
	
	float w;
	if (wobble < 0.2f){
		w = wobble/0.2f;
	} else {
		w = 1- ((wobble-0.2f)/0.8f);
	}
	
	
    radius = pow(radius, 1.f+w*1.f);
    p.x = radius * cos(theta);
    p.y = radius * sin(theta);
    return 0.5 * (p + 1.0);
}


void main() {
	
	vec2 tex = texCoord;
  vec2 xy = 2.0 * tex.xy - 1.0;
  vec2 uv;
  float d = length(xy);
  if (d < 1.5)
  {
    uv = Distort(xy);  }
  else
  {
    uv = tex;
  }
  
  mat2 RotationMatrix = mat2( cos(1.0), -sin(1.0),
                               sin(1.0),  cos(1.0));
	
	vec2 texrot = RotationMatrix*tex;
	
    vec3 diffColor = texture(image, uv).rgb;
	float factor = 0.f;
 	if (	texrot.y < 1-2*wobble){
		//diffColor = vec3(1,0,0);
		factor = 1.f;
	} 

//    out_color = vec4(diffColor,1);
    out_color =  vec4(mix(diffColor, blackAndWhite(diffColor), (1.0f-highlight) ),1);

}

##end

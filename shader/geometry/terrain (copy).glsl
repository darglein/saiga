##start
##vertex

#version 400
layout(location=0) in vec3 in_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;

uniform sampler2D image;


uniform vec4 ScaleFactor, FineBlockOrig;
uniform vec2 ViewerPos, AlphaOffset, OneOverWidth;
uniform float ZScaleFactor, ZTexScaleFactor;

out vec3 vertexMV;

void main() {

    // convert from grid xy to world xy coordinates
     //  ScaleFactor.xy: grid spacing of current level
     //  ScaleFactor.zw: origin of current block within world
    vec2 worldPos = in_position.xz * ScaleFactor.xy + ScaleFactor.zw;

    // sample the vertex texture
    float height = 0;//texture(image,tc).r;
    height = height*ZScaleFactor;

    vec4 position = vec4(worldPos.x+ViewerPos.x,height,worldPos.y+ViewerPos.y,1);
    vertexMV = vec3(view * position);
    gl_Position = proj*view * position;

//    vec2 tc = vec2(in_position.x,in_position.z)+vec2(ViewerPos.x/50.0f,ViewerPos.y/50.0f);
//    tc = tc*0.5f+vec2(0.5f);
//    float height = texture(image,tc).r*50.0f;

//    vec4 position = vec4(in_position.x,height,in_position.z,1);
//    position = model * position;
//    position += vec4(ViewerPos.x,0,ViewerPos.y,0);
//    vertexMV = vec3(view * position);
//    gl_Position = proj*view * position;
}


##end

##start
##geometry
#version 400
in vec3 vertexMV[3];
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;
out vec3 vertexMV2;
out vec3 normal;
void main()
{
    vec3 n;

    n = cross(vertexMV[1]-vertexMV[0],vertexMV[2]-vertexMV[0]);
    n = normalize(n);
  for(int i=0; i<3; i++)
  {
      vertexMV2 = vertexMV[i];
      normal = n;

    gl_Position = gl_in[i].gl_Position;
    EmitVertex();
  }
  EndPrimitive();
}
##end


##start
##fragment

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 color;

in vec3 normal;
in vec3 vertexMV2;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

void main() {

    vec4 diffColor = vec4(color);

    out_color =  vec3(diffColor);
    out_normal = normalize(normal)*0.5f+0.5f;
    out_position = vertexMV2;
}

##end

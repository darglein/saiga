##start
##vertex

#version 400
layout(location=0) in vec3 in_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;


out vec3 vertexMV;

void main() {
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    gl_Position = proj*view *model* vec4(in_position,1);
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


in vec3 vertexMV2;
in vec3 normal;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

void main() {

    vec4 diffColor = color;

    out_color =  vec3(diffColor);
    out_normal = normalize(normal)*0.5f+0.5f;
//    out_normal = vec3(0,1,0);
    out_position = vertexMV2;
}

##end

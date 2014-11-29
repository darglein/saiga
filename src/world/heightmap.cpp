#include "world/heightmap.h"

Heightmap::Heightmap(int w, int h){
    heightmap.width = n+1;
    heightmap.height = n+1;
    heightmap.bitDepth = 8;
    heightmap.channels = 1;
    heightmap.create();
}

void Heightmap::createNoiseHeightmap(){
    PerlinNoise noise;
    for(unsigned int x=0;x<heightmap.width;++x){
        for(unsigned int y=0;y<heightmap.height;++y){
            float xf = (float)x/(float)heightmap.width;
            float yf = (float)y/(float)heightmap.height;
            xf *= 10;
            yf *= 10;
            float h = noise.noise(xf,yf,0);
            h = h*256.0f;
            u_int8_t n = (u_int8_t)h;
            heightmap.setPixel(x,y,n);
        }
    }
}

void Heightmap::createTestHeightmap(){
    for(unsigned int x=0;x<heightmap.width;++x){
        for(unsigned int y=0;y<heightmap.height;++y){
            float xf = (float)x/(float)heightmap.width;
            float yf = (float)y/(float)heightmap.height;
            float h =0;
            h+=xf*0.5f;
            h+=yf*0.5f;
            h = h*256.0f;
            u_int8_t n = (u_int8_t)h;
            heightmap.setPixel(x,y,n);
        }
    }
}
 Texture* Heightmap::createTexture(){


    Texture* texture = new Texture();
    texture->fromImage(heightmap);
    texture->setWrap(GL_CLAMP_TO_EDGE);
    return texture;

}


 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMesh(){

    unsigned int w = 100;
    unsigned int h = 100;


     return createGridMesh(w,h,vec2(2.0f/(w-1),2.0f/(h-1)),vec2(1.0));

}


 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMesh2(){
     return createGridMesh(m,m,vec2(1.0f/(m-1)),vec2(0.5));
}



 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMeshFixUpV(){
     return createGridMesh(3,m,vec2(1.0f/(m-1)),vec2(0.5));
}



 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMeshFixUpH(){
     return createGridMesh(m,3,vec2(1.0f/(m-1)),vec2(0.5));
}

 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMeshTrim(){

     mesh_t* mesh = new mesh_t();

    int w = 2*m+1;
    int h = 2;
    vec2 d = vec2(1.0f/(m-1));
    vec2 o = vec2(0.5);

     float dw = d.x;
     float dh = d.y;
     for(unsigned int y=0;y<h;y++){
         for(unsigned int x=0;x<w;x++){
             float fx = (float)x*dw-o.x;
             float fy = (float)y*dh-o.y;
             Vertex v(vec3(fx,0.0f,fy));
             mesh->addVertex(v);
         }
     }


     for(unsigned int y=0;y<h-1;y++){
         for(unsigned int x=0;x<w-1;x++){
             GLuint quad[] = {y*w+x,(y+1)*w+x,(y+1)*w+x+1,y*w+x+1};
             mesh->addQuad(quad);
         }
     }

    int offset = mesh->vertices.size();

     w = 2;
     h = 2*m;

      dw = d.x;
      dh = d.y;
      for(unsigned int y=0;y<h;y++){
          for(unsigned int x=0;x<w;x++){
              float fx = (float)x*dw-o.x;
              float fy = (float)(y+1)*dh-o.y;
              Vertex v(vec3(fx,0.0f,fy));
              mesh->addVertex(v);
          }
      }


      for(unsigned int y=0;y<h-1;y++){
          for(unsigned int x=0;x<w-1;x++){
              GLuint quad[] = {offset+y*w+x,offset+(y+1)*w+x,offset+(y+1)*w+x+1,offset+y*w+x+1};
              mesh->addQuad(quad);
          }
      }

     return std::shared_ptr<Heightmap::mesh_t>(mesh);
}


 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMeshTrimi(){

    auto mesh = createMeshTrim();
    mesh->transform(glm::rotate(mat4(),glm::radians(180.0f),vec3(0,1,0)));
    return mesh;
}

 std::shared_ptr<Heightmap::mesh_t> Heightmap::createMeshCenter(){
     int m = this->m*2;
     return createGridMesh(m,m,vec2(1.0/(m-2)),vec2(0.5));
}


 std::shared_ptr<Heightmap::mesh_t> Heightmap::createGridMesh(unsigned int w, unsigned int h, vec2 d, vec2 o){
     mesh_t* mesh = new mesh_t();

     //creating uniform grid with w*h vertices
     //the resulting mesh will fill the quad (-1,0,-1) - (1,0,1)
     float dw = d.x;
     float dh = d.y;
     for(unsigned int y=0;y<h;y++){
         for(unsigned int x=0;x<w;x++){
             float fx = (float)x*dw-o.x;
             float fy = (float)y*dh-o.y;
             Vertex v(vec3(fx,0.0f,fy));
             mesh->addVertex(v);
         }
     }


     for(unsigned int y=0;y<h-1;y++){
         for(unsigned int x=0;x<w-1;x++){
             GLuint quad[] = {y*w+x,(y+1)*w+x,(y+1)*w+x+1,y*w+x+1};
             mesh->addQuad(quad);
         }
     }

     return std::shared_ptr<Heightmap::mesh_t>(mesh);
 }

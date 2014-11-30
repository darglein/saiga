#include "world/heightmap.h"

Heightmap::Heightmap(int w, int h):w(w),h(h){
    heightmap.width = w;
    heightmap.height = h;
    heightmap.bitDepth = 16;
    heightmap.channels = 1;
    heightmap.create();
    heights = new float[w*h];

    normalmap.width = w;
    normalmap.height = h;
    normalmap.bitDepth = 8;
    normalmap.channels = 3;
    normalmap.create();
}

void Heightmap::createNoiseHeightmap(){
    PerlinNoise noise;
    for(unsigned int x=0;x<heightmap.width;++x){
        for(unsigned int y=0;y<heightmap.height;++y){
            float xf = (float)x/(float)heightmap.width;
            float yf = (float)y/(float)heightmap.height;




            xf *= 30;
            yf *= 30;
            float h = noise.fBm(xf,yf,0,2);

            setHeight(x,y,h);

        }
    }

    normalizeHeightMap();

    for(unsigned int x=0;x<heightmap.width;++x){
        for(unsigned int y=0;y<heightmap.height;++y){

            float h = getHeight(x,y);
            h = h*65536.0f;
            h = glm::clamp(h,0.0f,65535.0f);
            u_int16_t n = (u_int16_t)h;
            heightmap.setPixel(x,y,n);
        }
    }
}

void Heightmap::normalizeHeightMap(){
    float diff = maxH-minH;

    float m = minH;

    cout<<"min "<<minH<<" max "<<maxH<<endl;
    for(unsigned int x=0;x<heightmap.width;++x){
        for(unsigned int y=0;y<heightmap.height;++y){
            float h = getHeight(x,y);
            h = h-m;
            h = h/diff;
            setHeight(x,y,h);
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

void Heightmap::createNormalmap(){
    for(int x=0;x<heightmap.width;++x){
        for(int y=0;y<heightmap.height;++y){

            vec3 x1 (x+1,getHeightScaled(x+1,y),y);
            vec3 x2 (x-1,getHeightScaled(x-1,y),y);

            vec3 y1 (x,getHeightScaled(x,y+1),y+1);
            vec3 y2 (x,getHeightScaled(x,y-1),y-1);

            vec3 n = glm::cross(y2-y1,x2-x1);

            //            cout<<"Normal "<<n<<" "<<x<<" "<<y<<" "<<x1<<" "<<x2<<" "<<y1<<" "<<y2<<endl;
            n = glm::normalize(n);
            n = 0.5f * (n+vec3(1.0f)); //now in range 0,1
            n = n*255.0f; //range 0,255
            n = glm::clamp(n,vec3(0),vec3(255));

            normalmap.setPixel(x,y,(u_int8_t)n.x,(u_int8_t)n.y,(u_int8_t)n.z);

        }
    }
}

float Heightmap::getHeight(int x, int y){
    x = glm::clamp(x,0,(int)(w-1));
    y = glm::clamp(y,0,(int)(h-1));

    return heights[x+y*w];
}

float Heightmap::getHeightScaled(int x, int y){
    return getHeight(x,y)*heightScale;
}

void Heightmap::setHeight(int x, int y, float v){
    minH = glm::min(v,minH);
    maxH = glm::max(v,maxH);
    heights[x+y*w] = v;
}

void Heightmap::createTextures(){
    texheightmap = new Texture();
    texheightmap->fromImage(heightmap);
    texheightmap->setWrap(GL_CLAMP_TO_EDGE);
    texheightmap->setFiltering(GL_LINEAR);

    texnormalmap = new Texture();
    texnormalmap->fromImage(normalmap);
    texnormalmap->setWrap(GL_CLAMP_TO_EDGE);
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



std::shared_ptr<Heightmap::mesh_t> Heightmap::createMeshDegenerated(){
    mesh_t* mesh = new mesh_t();

    int w = (n+1)/2;
    float dx = 2.0f/(m-1);

    vec2 d[] = {vec2(dx,0),vec2(dx,0),vec2(0,dx),vec2(0,dx)};
    vec2 o[] = {vec2(0.5),vec2(0.5,-3.5-(dx)),vec2(0.5),vec2(-3.5-(dx),0.5)};

    int orientation[] = {1,0,0,1};



    for(int i=0;i<4;i++){
        int offset = mesh->vertices.size();

        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*d[i].x-o[i].x;
            float fy = (float)x*d[i].y-o[i].y;
            Vertex v(vec3(fx,0.0f,fy));
            mesh->addVertex(v);
            if(x<w-1){
                //add vertex between
                fx = fx+0.5f*d[i].x;
                fy = fy+0.5f*d[i].y;
                Vertex v(vec3(fx,0.0f,fy));
                mesh->addVertex(v);
            }

        }


        for(unsigned int x=0;x<w-1;x++){
            //add degenerated triangle
            int idx = 2*x;
            GLuint face1[] = {offset+idx,offset+idx+1,offset+idx+2};
            GLuint face2[] = {offset+idx,offset+idx+2,offset+idx+1};

            if(orientation[i])
                mesh->addFace(face1);
            else
                mesh->addFace(face2);
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

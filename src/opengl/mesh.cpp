#include "opengl/mesh.h"



void MaterialMesh::draw(const mat4 &model, const Camera &cam){
    shader->bind();

    //static_cast casts at compile time. use if you're 100% sure that the object is from that type ;)
    static_cast<MaterialShader*>(shader)->uploadAll(model,cam.view,cam.proj);


    buffer.bind();

    //    cout<<"Drawing "<<triangleGroups.size()<<" Triangle Groups."<<endl;
    for(TriangleGroup &tg : triangleGroups){
        int* start = 0 ;
        start += tg.startFace*3;
        int count = tg.faces*3;
        Material* material = tg.mat;
        if(material == NULL)
            static_cast<MaterialShader*>(shader)->uploadMaterial(Material());
            else
        static_cast<MaterialShader*>(shader)->uploadMaterial(*material);

        buffer.draw(count,(void*)start);
//        glDrawElements(buffer.draw_mode,count, GL_UNSIGNED_INT, (void*)start);


    }

    buffer.unbind();
    shader->unbind();
}

FBMesh::FBMesh(){

}

void FBMesh::createQuadMesh(){
    VertexNT vertices[] = {
        VertexNT(vec3(-1,-1,0),vec3(0,0,1),vec2(0,0)),
        VertexNT(vec3(1,-1,0),vec3(0,0,1),vec2(1,0)),
        VertexNT(vec3(1,1,0),vec3(0,0,1),vec2(1,1)),
        VertexNT(vec3(-1,1,0),vec3(0,0,1),vec2(0,1))
    };
    GLuint indices[] = {0,1,2,3};
    buffer.set(vertices,4,indices,4);
    buffer.setDrawMode(GL_QUADS);
}

void FBMesh::bindUniforms(){

    shader->uploadFramebuffer(framebuffer);
}

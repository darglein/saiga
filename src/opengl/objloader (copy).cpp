#include "opengl/objloader.h"

bool splitAtFirst(string &str, string &erg, char de){
    unsigned int index = 0;
    for(;index<str.length();index++){
        if(str[index] == de){
            break;
        }
    }
    if(index==0)
        return false;
    erg = str.substr(0,index);
    if(index<str.length())
        str = str.substr(index+1);
    else
        str = "";

    //check if str has leading de's
    while(str[0]==de){
        str = str.substr(1);
    }
    return true;
}

bool ObjLoader::extractIndices(char* line, int &v1,int &v2,int &v3){
    char* cva1 = line;
    char* cva2 = line;
    char* cva3;
    //read until first '/' or end
    while(cva2[0]!=0 && cva2[0]!='/' ){
        cva2++;
    }
    if(cva2[0]=='/'){
        cva2[0] = 0;
        cva2++;
    }
    cva3 = cva2;
    while(cva3[0]!=0 && cva3[0]!='/' ){
        cva3++;
    }
    if(cva3[0]=='/'){
        cva3[0] = 0;
        cva3++;
    }
    //obj indexing starts counting from 1
    v1 = atoi(cva1) - 1;
    v2 = atoi(cva2) - 1;
    v3 = atoi(cva3) - 1;
    return true;
}

#define TOKENS 4
char tokens[] = {' ',',','\t','\r'};

inline bool isToken(char c){
    for(int i=0;i<TOKENS;i++){
        if(c==tokens[i])
            return true;
    }
    return false;
}

bool skipTokens(char* &str){
    while(str[0]!=0 && isToken(str[0])){
        if(str[0]=='#')
            return false;
        str++;
    }

    return true;
}

bool readUntilToken(char* &str,string &out_value){
    out_value = "";
    while(str[0]!=0 && str[0]!='#' && !isToken(str[0])){
        out_value.append(str,1);
        str++;
    }
    return out_value.length()>0;
}

float nextFloat(char* &str){
    string ergs;
    readUntilToken(str,ergs);
    skipTokens(str);
    return atof(ergs.c_str());
}

vec3 readVec3(char* &str){
    vec3 v;
    v.x = nextFloat(str);
    v.y = nextFloat(str);
    v.z = nextFloat(str);
    return v;
}
vec2 readVec2(char* &str){
    vec2 v;
    v.x = nextFloat(str);
    v.y = nextFloat(str);
    return v;
}



MaterialMesh* ObjLoader::loadFromFile(const string &path){


    outIndices.clear();
    outVertices.clear();
//    mesh.clear();
    triangleGroups.resize(0);



    std::fstream stream;
    stream.exceptions ( std::fstream::failbit | std::fstream::badbit );


    try {
        stream.open (path, std::fstream::in);
    }
    catch (std::fstream::failure e) {
        return NULL;
    }


//    cout<<"objloader: loading file "<<path<<endl;

    char buffer[1024];
    string line;
    string header;

    //set first group
    triangleGroups.push_back(TriangleGroup());
    triangleGroups[0].startFace = 0;
    triangleGroups[0].mat = NULL;

    try {
        while(stream.getline(buffer,1024)){ //lines

            parseLine(buffer);
        }

    }
    catch (std::fstream::failure &e) {
//        cout<<"Loading finished"<<endl;
    }

    //end last group
    TriangleGroup &group = triangleGroups[triangleGroups.size()-1];
    group.faces = faces.size()-group.startFace;

    createOutput();


    cout<<"Obj loading finished. Vertices: "<<outVertices.size()<<" Indices: "<<outIndices.size()<<" Faces: "<<faces.size()<<endl;
//    cout<<"Obj loading finished. "<<mesh<<endl;

    if(maxCorners>3)
        cout<<"Warning, this model is not triangulated. Maximum number of vertices per face: "<<maxCorners<<endl;

    MaterialMesh* mmesh = new MaterialMesh();

//    mesh.createBuffers(mmesh->buffer);
    mmesh->buffer.set(outVertices,outIndices);
    mmesh->buffer.setDrawMode(GL_TRIANGLES);
//    mesh->buffer.createGLBuffers();
    mmesh->triangleGroups.swap(triangleGroups); //destroys Objloader::triangleGroups in the process

    return mmesh;
}


void ObjLoader::parseLine(char* line){
    //    cout<<line<<endl;
    string header;
    skipTokens(line);
    readUntilToken(line,header);
    skipTokens(line);

    if(header == "#"){

    }else if(header == "usemtl"){

        Material* mat = materialLoader->load(line);
        if(mat ==NULL){
            cout<<"Could not find material: "<<line<<endl;
            return;
        }

        if(faces.size()==0){
//            cout<<"use material: "<<line<<" "<<faces.size()<<" id: "<<matId<<endl;
            triangleGroups[0].startFace = faces.size();
            triangleGroups[0].mat = mat;
            return;
        }
        TriangleGroup &currentGroup = triangleGroups[triangleGroups.size()-1];
        if(currentGroup.mat == mat){
            //duplicated material
            return;
        }

//        cout<<"use material: "<<line<<" "<<faces.size()<<" id: "<<matId<<endl;
        //finish last group and create new one
        currentGroup.faces = faces.size()-currentGroup.startFace;

        TriangleGroup newGroup;
        newGroup.startFace = faces.size();
        newGroup.mat = mat;
        triangleGroups.push_back(newGroup);

    }else if(header == "mtllib"){
        cout<<"load material library: "<<line<<" "<<endl;
        materialLoader->loadLibrary(line);
    }else if(header == "g"){
        //        cout<<"Found Group: "<<line<<endl;
    }else if(header == "o"){
        //        cout<<"Found Object: "<<line<<endl;
        //cout<<endl;
    }else if(header == "s"){

    }else if(header == "v"){
        parseV(line);
    }else if(header == "vt"){
        parseT(line);
    }else if(header == "vn"){
        parseN(line);
    }else if(header == "f"){
        parseF(line);

    }else{
        //        cout<<"warning unknown .obj command: ";
        //        cout<<line<<endl;
    }

}

void ObjLoader::parseV(char* line){
    vertices.push_back(readVec3(line));
}

void ObjLoader::parseN(char* line){
    normals.push_back(readVec3(line));
}

void ObjLoader::parseT(char* line){
    texCoords.push_back(readVec2(line));
}

void ObjLoader::parseF(char* line){
    //    cout<<line<<endl;
    string value;
    int cornerCount = 0;
    IndexedVertex currentVertex,startVertex, lastVertex;
    char buffer[1024];
    Face face;
    while(readUntilToken(line,value)){

        int n = value.copy(buffer,1024);
        buffer[n] = 0;
        //        cout<<n<<" "<<buffer<<" "<<(int)buffer[0]<<endl;
        cornerCount++;
        int vert,tex,norm;
        if(!extractIndices(buffer,vert,tex,norm))
            continue;

        //        cout<<vert<<" "<<tex<<" "<<norm<<endl;
        currentVertex.v = vert;
        currentVertex.n = norm;
        currentVertex.t = tex;
        if(cornerCount<=3){
            if(cornerCount==1)
                startVertex = currentVertex;
            face.vertices[cornerCount-1] = currentVertex;
            if(cornerCount==3)
                faces.push_back(face);
        }else{
            Face f;
            f.vertices[0] = lastVertex;
            f.vertices[1] = currentVertex;
            f.vertices[2] = startVertex;
            faces.push_back(f);
        }

        lastVertex = currentVertex;


        //        lastIndex = index;

        maxCorners = glm::max(maxCorners,cornerCount);


        if(!skipTokens(line))
            return;
    }
}

void ObjLoader::createOutput(){
//    cout<<"Triangle groups: "<<triangleGroups.size()<<endl;
    aabb voxel_bounds;
    voxel_bounds.makeNegative();



//    mesh.vertices.resize(vertices.size());

    vertices_used.resize(vertices.size());
    outVertices.resize(vertices.size());


    for(unsigned int i=0;i<vertices_used.size();i++){
        vertices_used[i] = false;
        voxel_bounds.growBox(vertices[i]);
    }
    cout<<"Object bounds: "<<voxel_bounds<<endl;


    for(Face &f : faces){
        GLuint fa[3];
        for(int i=0;i<3;i++){
            IndexedVertex &currentVertex = f.vertices[i];
            int vert = currentVertex.v;
            int norm = currentVertex.n;
            int tex = currentVertex.t;

            VertexNT verte;
            if(vert>=0)
                verte.position =vertices[vert];
            if(norm>=0)
                verte.normal = normals[norm];
            if(tex>=0)
                verte.texture =texCoords[tex];


            int index = -1;
            if(vertices_used[vert]){
                if(verte==outVertices[vert]){
                    index = vert;
                }
            }else{
                outVertices[vert] = verte;
                index = vert;
                vertices_used[vert] = true;
            }

            if(index==-1){
                index = outVertices.size();
                outVertices.push_back(verte);
            }
            fa[i] = index;
            outIndices.push_back(index);
        }
//        mesh.addFace(fa);
    }
}

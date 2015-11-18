#pragma once

#include <saiga/assets/asset.h>

class SAIGA_GLOBAL ColoredAsset : public BasicAsset<VertexNC,GLuint>{
};


class SAIGA_GLOBAL TexturedAsset : public BasicAsset<VertexNC,GLuint>{
};

class SAIGA_GLOBAL AnimatedAsset : public BasicAsset<VertexNC,GLuint>{
};

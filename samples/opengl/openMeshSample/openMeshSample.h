/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/window/SampleWindowForward.h"


using namespace Saiga;

class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();
    ~Sample();

    void reduce();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;


    virtual void render(Camera* camera, RenderPass render_pass) override;


   private:
    bool useAspectRatio   = true;
    float ratio           = 3;
    float errorTolerance  = 1;
    bool useQuadric       = false;
    float quadricMaxError = 0.001;
    bool useHausdorf      = false;
    float hausError       = 0.01;
    bool useNormalDev     = false;
    float normalDev       = 20;
    bool useNormalFlip    = true;
    float maxNormalDev    = 8;
    bool useRoundness     = false;
    float minRoundness    = 0.4;

    bool showReduced = false;
    bool writeToFile = false;
    bool wireframe   = true;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject sphere;


    TriangleMesh<VertexNC, GLuint> baseMesh;
    TriangleMesh<VertexNC, GLuint> reducedMesh;
};

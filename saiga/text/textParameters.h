#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"

struct SAIGA_GLOBAL TextParameters{

    vec4 color = vec4(1,1,1,1);
    vec4 outlineColor = vec4(0,0,0,0);
    vec4 glowColor = vec4(0,0,0,0);

    vec4 outlineData = vec4(0.5f,0.5f,0.5f,0.5f);
    vec2 softEdgeData = vec2(0.5f,0.5f);
    vec2 glowData = vec2(0.5f,0.5f);
    float alpha = 1.0f;

    void setOutline(const vec4& outlineColor, float width, float smoothness);
    void setGlow(const vec4& glowColor, float width);
    void setColor(const vec4& color, float smoothness);
    void setAlpha(float alpha);
} GLM_ALIGN(16);

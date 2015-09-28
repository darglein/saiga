#include "saiga/rendering/overlay/textDebugOverlay.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/geometry/triangle_mesh.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/text/text_generator.h"
#include "saiga/text/dynamic_text.h"
#include "saiga/text/textShader.h"

TextDebugOverlay::TextDebugOverlay(): overlay(1600,900){

    overlay.setTextShader(ShaderLoader::instance()->load<TextShader>("deferred_text.glsl"));

}

void TextDebugOverlay::init(TextGenerator *textGenerator)
{
    this->textGenerator = textGenerator;
    text = textGenerator->createDynamicText(23);
    text->color = vec4(1,0,0,1);
    vec3 size = text->getSize();
    textGenerator->updateText(text,"ASFASFASF     ",0);
    text->translateGlobal(vec3(10,overlay.height-size.y-20,0));
    text->calculateModel();
    overlay.addText(text);
}

void TextDebugOverlay::render()
{
    overlay.render();
}


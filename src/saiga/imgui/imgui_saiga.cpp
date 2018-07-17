#include "saiga/imgui/imgui.h"

#if defined(SAIGA_VULKAN_INCLUDED) || defined(SAIGA_OPENGL_INCLUDED)
#error This module must be independent of any graphics API.
#endif

namespace ImGui {

//void Texture(std::shared_ptr<Saiga::raw_Texture> texture, const ImVec2& size){
//    ImGui::Image((void*)(intptr_t)texture->getId(),size,vec2(0,1),vec2(1,0));
//}

}

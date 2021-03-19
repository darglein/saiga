/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_renderer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/all.h"
#include "saiga/opengl/texture/Texture.h"

namespace Saiga
{
class SAIGA_OPENGL_API ImGui_GL_Renderer : public ImGuiRenderer
{
   public:
    ImGui_GL_Renderer(const ImGuiParameters& params);
    virtual ~ImGui_GL_Renderer();

    void render();

   protected:
    std::shared_ptr<Shader> shader;
    std::shared_ptr<Texture> texture;
    IndexedVertexBuffer<ImDrawVert, ImDrawIdx> buffer;

    virtual void renderDrawLists(ImDrawData* draw_data);
};



// A system which tracks OpenGL time on a per frame basis.
// The GLrender has one object which should be used by every subprocess.
class SAIGA_OPENGL_API GLTimerSystem
{
   public:
    // One measurement is given by the start and end tick (in ns)
    using Measurement = std::pair<uint64_t, uint64_t>;

    struct TimeData
    {
        MultiFrameOpenGLTimer timer;

        // Past times stored in ms
        std::vector<Measurement> measurements_ms;
        Measurement last_measurement = {0, 0};
        Measurement capture          = {0, 0};
        int depth                    = 0;

        int count = 0;

        TimeData(int& current_depth);
        void AddTime(Measurement t);

        void Start()
        {
            depth = current_depth++;
            timer.startTimer();
        }
        void Stop()
        {
            timer.stopTimer();
            current_depth--;
            SAIGA_ASSERT(depth == current_depth);
            AddTime(timer.LastMeasurement());
        }

       private:
        int& current_depth;
    };



    struct ScopedTimingSection
    {
        ScopedTimingSection(TimeData& sec) : sec(sec) { sec.Start(); }
        ~ScopedTimingSection() { sec.Stop(); }
        TimeData& sec;
    };



    GLTimerSystem();


    ScopedTimingSection CreateScope(const std::string& name);
    TimeData& GetTimer(const std::string& name);


    void BeginFrame();
    void EndFrame();

    void Imgui();



   private:
    int current_depth = 0;
    std::map<std::string, std::shared_ptr<TimeData>> data;

    bool has_capture  = false;
    bool capture_next = false;
};



}  // namespace Saiga

namespace ImGui
{
SAIGA_OPENGL_API void Texture(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y,
                              const ImVec4& tint_col   = ImVec4(1, 1, 1, 1),
                              const ImVec4& border_col = ImVec4(0, 0, 0, 0));

}

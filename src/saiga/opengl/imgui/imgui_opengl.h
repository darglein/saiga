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


    struct TimeStats
    {
        int depth;
        std::string Name;
        // statistics (all in ms)
        float stat_last   = 0;
        float stat_min    = 0;
        float stat_max    = 0;
        float stat_median = 0;
        float stat_mean   = 0;
        float stat_sdev   = 0;
    };

    struct TimeData
    {
        MultiFrameOpenGLTimer timer;

        TimeStats stats;

        std::vector<Measurement> measurements_ms;
        Measurement last_measurement = {0, 0};
        Measurement capture          = {0, 0};

        int count   = 0;
        bool active = false;

        TimeData(int& current_depth, int samples);
        void AddTime(Measurement t);

        void Start()
        {
            if (current_depth >= 0)
            {
                stats.depth = current_depth++;
                timer.startTimer();
            }
        }
        void Stop()
        {
            if (current_depth >= 0)
            {
                timer.stopTimer();
                current_depth--;
                SAIGA_ASSERT(stats.depth == current_depth);
                AddTime(timer.LastMeasurement());
            }
        }

        std::vector<float> ComputeTimes();

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
    int num_samples = 100;

    // we count the number of frames so that the expensive statistic recomputation is only done
    // once every #num_samples frames.
    int current_frame = 0;


    int current_depth = 0;
    std::map<std::string, std::shared_ptr<TimeData>> data;

    bool has_capture = false;
    bool capturing   = true;
    int current_view = 1;

    bool render_window       = true;
    bool normalize_time      = true;
    float absolute_scale_fps = 60;
};



}  // namespace Saiga

namespace ImGui
{
SAIGA_OPENGL_API void Texture(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y,
                              const ImVec4& tint_col   = ImVec4(1, 1, 1, 1),
                              const ImVec4& border_col = ImVec4(0, 0, 0, 0));

}

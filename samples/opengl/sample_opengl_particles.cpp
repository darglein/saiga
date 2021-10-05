/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"
#include "saiga/core/model/all.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/window/message_box.h"
#include "saiga/opengl/world/pointCloud.h"
using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;


    enum CopyMethod
    {
        CM_subData = 0,
        CM_map     = 1,
    };
    CopyMethod cm = CM_subData;

   public:
    Sample()
    {
        shader_simple = shaderLoader.load<MVPShader>("geometry/colored_points.glsl");


        gpu_timers = {{ImGui::TimeGraph("GPU Update Buffer"), MultiFrameOpenGLTimer(false)}};

        for (auto& g : gpu_timers)
        {
            g.second.create();
        }

        Create();
        std::cout << "Program Initialized!" << std::endl;
    }

    void Create()
    {
        h_buffer.resize(n_points);

        for (auto& v : h_buffer)
        {
            v.position = Random::MatrixUniform<vec3>(0, size);
            v.color    = Random::MatrixUniform<vec3>(-0.2, 0.2);
        }
        buffers.clear();
        buffers.resize(num_multi_buffers);

        for (int i = 0; i < num_multi_buffers; ++i)
        {
            buffers[i] = std::make_shared<VertexBuffer<PointVertex>>();
            buffers[i]->setDrawMode(GL_POINTS);
            buffers[i]->set(h_buffer, buffer_type);
        }
        current_draw_buffer = 0;
    }

    void simulate(float dt)
    {
        float t;
        {
            ScopedTimer tim(t);
            for (auto& v : h_buffer)
            {
                v.position += dt * v.color;
            }
        }
        tg.addTime(t);
    }

    void updatebuffer()
    {
        float t;
        {
            ScopedTimer tim(t);
            gpu_timers[0].second.Start();

            current_draw_buffer = (current_draw_buffer + 1) % num_multi_buffers;
            auto buffer         = buffers[current_draw_buffer];


            size_t buffer_size = h_buffer.size() * sizeof(PointVertex);


            if (cm == CM_subData)
            {
                glBindBuffer(buffer->target, buffer->buffer);

                if (orphaning_buffer)
                {
                    glBufferData(buffer->target, buffer_size, 0, buffer_type);
                }
                glBufferSubData(buffer->target, 0, buffer_size, h_buffer.data());
            }
            else if (cm == CM_map)
            {
                glBindBuffer(buffer->target, buffer->buffer);
                auto ptr = buffer->mapBuffer(GL_READ_WRITE);
                memcpy(ptr, h_buffer.data(), buffer_size);
                buffer->unmapBuffer();
            }


            gpu_timers[0].second.Stop();
            // std::cout << gpu_timers[0].second.getTimeNS() << std::endl;
            gpu_timers[0].first.addTime(gpu_timers[0].second.getTimeMS());
        }
        tg_buf_cpu.addTime(t);
    }

    void update(float dt) override
    {
        Base::update(dt);


        if (run_simulation) simulate(dt);
        updatebuffer();
    }

    void render(RenderInfo render_info) override
    {
        Base::render(render_info);
        if (render_info.render_pass == RenderPass::Forward)
        {
            if (render_points)
            {
                if (shader_simple->bind())
                {
                    shader_simple->uploadModel(mat4::Identity());

                    auto buffer = buffers[current_draw_buffer];
                    buffer->bindAndDraw();

                    shader_simple->unbind();
                }
            }
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            if (ImGui::Begin("Saiga Sample"))
            {
                ImGui::InputInt("n_points", &n_points);

                {
                    static int curr_item              = 0;
                    std::vector<std::string> elements = {"GL_STREAM_DRAW", "GL_STATIC_DRAW", "GL_DYNAMIC_DRAW"};
                    ImGui::Combo("Buffer Usage", &curr_item, elements);

                    std::vector<GLenum> types = {GL_STREAM_DRAW, GL_STATIC_DRAW, GL_DYNAMIC_DRAW};
                    buffer_type               = types[curr_item];
                }

                if (ImGui::InputInt("num_multi_buffers", &num_multi_buffers))
                {
                    num_multi_buffers = clamp(num_multi_buffers, 1, 10);
                    Create();
                }

                if (ImGui::Button("create"))
                {
                    Create();
                }

                {
                    static int curr_item              = 0;
                    std::vector<std::string> elements = {"subdata", "map"};
                    ImGui::Combo("Upload method", &curr_item, elements);

                    cm = CopyMethod(curr_item);
                }

                ImGui::Checkbox("orphaning_buffer", &orphaning_buffer);
                ImGui::Checkbox("run_simulation", &run_simulation);
                ImGui::Checkbox("render_points", &render_points);


                tg.renderImGui();
                tg_buf_cpu.renderImGui();

                for (auto& t : gpu_timers)
                {
                    t.first.renderImGui();
                }
            }
            ImGui::End();
        }
    }

   private:
    std::vector<std::pair<ImGui::TimeGraph, MultiFrameOpenGLTimer>> gpu_timers;

    bool run_simulation   = true;
    bool render_points    = true;
    int num_multi_buffers = 1;
    bool orphaning_buffer = true;
    GLenum buffer_type    = GL_STREAM_DRAW;
    float size            = 4;
    int n_points          = 1000000;
    std::shared_ptr<MVPShader> shader_simple;

    int current_draw_buffer = 0;
    std::vector<std::shared_ptr<VertexBuffer<PointVertex>>> buffers;



    std::vector<PointVertex> h_buffer;
    ImGui::TimeGraph tg         = {"Simulation Time"};
    ImGui::TimeGraph tg_buf_cpu = {"Update Buffer Time (CPU)"};
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}

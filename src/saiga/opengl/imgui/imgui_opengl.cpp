/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_opengl.h"



namespace Saiga
{
template <>
void VertexBuffer<ImDrawVert>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);


    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ImDrawVert), (void*)(sizeof(float) * 0));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(ImDrawVert), (void*)(sizeof(float) * 2));
    glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(ImDrawVert), (void*)(sizeof(float) * 4));
}

ImGui_GL_Renderer::ImGui_GL_Renderer(const ImGuiParameters& params) : ImGuiRenderer(params)
{
    shader = shaderLoader.load<Shader>("imgui_gl.glsl");

    std::vector<ImDrawVert> test(5);
    std::vector<ImDrawIdx> test2(5);
    buffer.set(test, test2, GL_DYNAMIC_DRAW);

    {
        // Build texture atlas
        ImGuiIO& io = ImGui::GetIO();
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width,
                                     &height);  // Load as RGBA 32-bits for OpenGL3 demo because it is more likely to be
                                                // compatible with user's existing shader.
        texture = std::make_shared<Texture>();
        texture->create(width, height, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, pixels);
        io.Fonts->TexID = (void*)(intptr_t)texture->getId();
    }
}

ImGui_GL_Renderer::~ImGui_GL_Renderer() {}

void ImGui_GL_Renderer::render()
{
    renderDrawLists(ImGui::GetDrawData());
}


void ImGui_GL_Renderer::renderDrawLists(ImDrawData* draw_data)
{
    assert_no_glerror();

    // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer
    // coordinates)
    ImGuiIO& io   = ImGui::GetIO();
    int fb_width  = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
    int fb_height = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
    if (fb_width == 0 || fb_height == 0) return;
    draw_data->ScaleClipRects(io.DisplayFramebufferScale);

    // Backup GL state
    GLint last_blend_src;
    glGetIntegerv(GL_BLEND_SRC, &last_blend_src);
    GLint last_blend_dst;
    glGetIntegerv(GL_BLEND_DST, &last_blend_dst);
    GLint last_blend_equation_rgb;
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &last_blend_equation_rgb);
    GLint last_blend_equation_alpha;
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &last_blend_equation_alpha);
    GLint last_viewport[4];
    glGetIntegerv(GL_VIEWPORT, last_viewport);
    GLboolean last_enable_blend        = glIsEnabled(GL_BLEND);
    GLboolean last_enable_cull_face    = glIsEnabled(GL_CULL_FACE);
    GLboolean last_enable_depth_test   = glIsEnabled(GL_DEPTH_TEST);
    GLboolean last_enable_scissor_test = glIsEnabled(GL_SCISSOR_TEST);

    // Setup render state: alpha-blending enabled, no face culling, no depth testing, scissor enabled
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);
    glActiveTexture(GL_TEXTURE0);

    // Setup orthographic projection matrix
    glViewport(0, 0, (GLsizei)fb_width, (GLsizei)fb_height);
    const float ortho_projection[4][4] = {
        {2.0f / io.DisplaySize.x, 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f / -io.DisplaySize.y, 0.0f, 0.0f},
        {0.0f, 0.0f, -1.0f, 0.0f},
        {-1.0f, 1.0f, 0.0f, 1.0f},
    };
    shader->bind();

    glUniform1i(1, 0);
    glUniformMatrix4fv(0, 1, GL_FALSE, &ortho_projection[0][0]);



    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list         = draw_data->CmdLists[n];
        const ImDrawIdx* idx_buffer_offset = 0;

        buffer.VertexBuffer<ImDrawVert>::fill(cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.size(), GL_DYNAMIC_DRAW);
        buffer.IndexBuffer<ImDrawIdx>::fill(cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.size(), GL_DYNAMIC_DRAW);

        buffer.bind();

        for (const ImDrawCmd* pcmd = cmd_list->CmdBuffer.begin(); pcmd != cmd_list->CmdBuffer.end(); pcmd++)
        {
            if (pcmd->UserCallback)
            {
                pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D, (GLuint)(intptr_t)pcmd->TextureId);
                glScissor((int)pcmd->ClipRect.x, (int)(fb_height - pcmd->ClipRect.w),
                          (int)(pcmd->ClipRect.z - pcmd->ClipRect.x), (int)(pcmd->ClipRect.w - pcmd->ClipRect.y));
                glDrawElements(GL_TRIANGLES, (GLsizei)pcmd->ElemCount,
                               sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, idx_buffer_offset);
            }
            idx_buffer_offset += pcmd->ElemCount;
        }
        buffer.unbind();
    }
    shader->unbind();

    // Restore modified GL state
    glBlendEquationSeparate(static_cast<GLenum>(last_blend_equation_rgb),
                            static_cast<GLenum>(last_blend_equation_alpha));
    glBlendFunc(static_cast<GLenum>(last_blend_src), static_cast<GLenum>(last_blend_dst));
    if (static_cast<bool>(last_enable_blend))
        glEnable(GL_BLEND);
    else
        glDisable(GL_BLEND);
    if (static_cast<bool>(last_enable_cull_face))
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);
    if (static_cast<bool>(last_enable_depth_test))
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    if (static_cast<bool>(last_enable_scissor_test))
        glEnable(GL_SCISSOR_TEST);
    else
        glDisable(GL_SCISSOR_TEST);
    glViewport(last_viewport[0], last_viewport[1], (GLsizei)last_viewport[2], (GLsizei)last_viewport[3]);

    assert_no_glerror();
}

GLTimerSystem::GLTimerSystem() {}

GLTimerSystem::ScopedTimingSection GLTimerSystem::CreateScope(const std::string& name)
{
    return GLTimerSystem::ScopedTimingSection(GetTimer(name));
}

GLTimerSystem::TimeData& GLTimerSystem::GetTimer(const std::string& name)
{
    std::shared_ptr<TimeData>& td = data[name];
    if (!td)
    {
        td = std::make_shared<TimeData>(current_depth);
    }
    return *td;
}

void GLTimerSystem::BeginFrame()
{
    GetTimer("Frame").Start();
}

void GLTimerSystem::EndFrame()
{
    GetTimer("Frame").Stop();
    SAIGA_ASSERT(current_depth == 0);

    if (capture_next)
    {
        for (auto& st : data)
        {
            st.second->capture = st.second->last_measurement;
        }
        capture_next = false;
        has_capture  = true;
    }
}

void GLTimerSystem::Imgui()
{
    if (ImGui::Begin("Frame Time"))
    {
        //        ImGui::Text("Render Time");

        if (ImGui::Button("Capture"))
        {
            console << "Capturing Timings in next frame..." << std::endl;
            capture_next = true;
        }

        if (has_capture)
        {
            TimeData& total_time = GetTimer("Frame");
            auto total_diff      = total_time.capture.second - total_time.capture.first;


            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            // Here we are using InvisibleButton() as a convenience to 1) advance the cursor and 2) allows us to use
            // IsItemHovered() But you can also draw directly and poll mouse/keyboard by yourself. You can manipulate
            // the cursor using GetCursorPos() and SetCursorPos(). If you only use the ImDrawList API, you can notify
            // the owner window of its extends by using SetCursorPos(max).
            ImVec2 canvas_pos  = ImGui::GetCursorScreenPos();     // ImDrawList API uses screen coordinates!
            ImVec2 canvas_size = ImGui::GetContentRegionAvail();  // Resize canvas to what's available
            if (canvas_size.x < 50.0f) canvas_size.x = 50.0f;
            if (canvas_size.y < 50.0f) canvas_size.y = 50.0f;
            draw_list->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                                     IM_COL32(150, 150, 150, 255));
            draw_list->AddRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                               IM_COL32(255, 255, 255, 255));


            float box_h = 30;

            for (auto& st : data)
            {
                TimeData& td = *st.second;

                float x1_rel = float(td.capture.first - total_time.capture.first) / float(total_diff);
                float x2_rel = float(td.capture.second - total_time.capture.first) / float(total_diff);



                ImVec2 box_pos  = canvas_pos + ImVec2(canvas_size.x * x1_rel, td.depth * box_h);
                ImVec2 box_size = ImVec2(canvas_size.x * (x2_rel - x1_rel), box_h);
                draw_list->AddRect(box_pos, box_pos + box_size, IM_COL32(255, 255, 255, 255));

                draw_list->AddText(box_pos, IM_COL32(255, 255, 255, 255), st.first.c_str());

                if (!st.second->measurements_ms.empty())
                {
                    // ImGui::Text("%s: %f", st.first.c_str(), st.second->measurements_ms.back());
                }
            }
        }
    }
    ImGui::End();
}

GLTimerSystem::TimeData::TimeData(int& current_depth) : measurements_ms(1000, {0, 0}), current_depth(current_depth)
{
    timer.create();
}

void GLTimerSystem::TimeData::AddTime(Measurement t)
{
    last_measurement                                = t;
    measurements_ms[count % measurements_ms.size()] = t;
    count++;
}



}  // namespace Saiga

void ImGui::Texture(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y, const ImVec4& tint_col,
                    const ImVec4& border_col)
{
    size_t tid     = texture->getId();
    ImTextureID id = (ImTextureID)tid;

    if (flip_y)
    {
        Image(id, size, ImVec2(0, 1), ImVec2(1, 0), tint_col, border_col);
    }
    else
    {
        Image(id, size, ImVec2(0, 0), ImVec2(1, 1), tint_col, border_col);
    }
}

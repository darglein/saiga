/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_opengl.h"

#include "saiga/core/util/statistics.h"


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
        td             = std::make_shared<TimeData>(current_depth, num_samples);
        td->stats.Name = name;
    }
    return *td;
}

void GLTimerSystem::BeginFrame()
{
    if (capturing)
    {
        for (auto& st : data)
        {
            st.second->active = false;
        }
        current_depth = 0;
    }
    else
    {
        current_depth = -1;
    }
    GetTimer("Frame").Start();
}

void GLTimerSystem::EndFrame()
{
    GetTimer("Frame").Stop();

    if (capturing)
    {
        SAIGA_ASSERT(current_depth == 0);
        for (auto& st : data)
        {
            st.second->capture = st.second->last_measurement;
        }
        has_capture = true;
    }
}

enum TimingTableColumnId
{
    TimingTableColumnId_ID,
    TimingTableColumnId_Name,
    TimingTableColumnId_Last,
    TimingTableColumnId_Min,
    TimingTableColumnId_Max,
    TimingTableColumnId_Median,
    TimingTableColumnId_Mean,
};

struct FrameGraphElement
{
    GLTimerSystem::TimeData* data;

    // Bounding box of the rectangle
    vec2 box_min;
    vec2 box_max;

    bool hover(vec2 mouse)
    {
        return mouse(0) >= box_min(0) && mouse(0) <= box_max(0) && mouse(1) >= box_min(1) && mouse(1) <= box_max(1);
    }

    bool highlighted = false;
};

void GLTimerSystem::Imgui()
{
    //    ImGui::ShowDemoWindow();
    if (!render_window) return;

    if (current_frame % num_samples == 0)
    {
        // recompute statistics
        for (auto& st : data)
        {
            TimeData& td = *st.second;

            std::vector<float> times;

            for (auto& m : td.measurements_ms)
            {
                auto delta = m.second - m.first;

                if (m.first > 0 && delta > 0)
                {
                    float delta_ms = float(delta) / (1000 * 1000);
                    times.push_back(delta_ms);
                }
            }

            Statistics stats(times);
            td.stats.stat_min    = stats.min;
            td.stats.stat_max    = stats.max;
            td.stats.stat_median = stats.median;
            td.stats.stat_mean   = stats.mean;
            td.stats.stat_sdev   = stats.sdev;
        }
    }

    current_frame++;
    TimeData& total_time = GetTimer("Frame");

    if (ImGui::Begin("Frame Time", &render_window))
    {
        ImGui::PushItemWidth(120);
        const std::vector<std::string> names = {"Table View", "Graph View"};
        ImGui::PushID("lsdgsdg");
        ImGui::Combo("", &current_view, names);
        ImGui::PopID();

        ImGui::PopItemWidth();

        ImGui::SameLine();
        ImGui::Checkbox("Capture", &capturing);
        ImGui::SameLine();
        ImGui::Checkbox("Normalize Time", &normalize_time);

        ImGui::SameLine();
        ImGui::PushItemWidth(80);
        if (ImGui::InputInt("#Samples", &num_samples, 0))
        {
            num_samples = std::max(1, num_samples);
            // resize the sample array of all elements
            for (auto& st : data)
            {
                TimeData& td = *st.second;
                td.count     = 0;
                td.measurements_ms.clear();
                td.measurements_ms.resize(num_samples, {0, 0});
            }
        }
        ImGui::PopItemWidth();

        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(
                "Saiga OpenGL Frame Timer\n"
                " - Displays current and past timer measurements\n"
                " - Timers can be nested by calling start/stop while an other timer is running\n"
                " - The statistics (mean/median/...) are computed over the last '#Samples' frames\n"
                " - Usage:\n"
                "      auto tim = timer->CreateScope(\"Geometry\");\n"
                "\n"
                "Table View\n"
                " - Shows all active timers in a table\n"
                " - Click on column names to sort\n"
                " - Shift Click for multi sort\n"
                "\n"
                "Graph View\n"
                " - Plots the last captured frame\n"
                " - Other statistics are shown at mouse hover\n"
                " - Nested Timer calls are drawn below each other\n"
                " - Use the 'normalize time' checkbox to scale the width by the total frame time\n"
                "");
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }

        if (has_capture)
        {
            TimeData* draw_tooltip = nullptr;
            if (current_view == 0)
            {
                // Table
                // const float TEXT_BASE_WIDTH  = ImGui::CalcTextSize("A").x;
                // const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();

                std::vector<TimeData*> items;

                for (auto& st : data)
                {
                    TimeData& td = *st.second;
                    if (td.active)
                    {
                        items.push_back(&td);
                    }
                }


                // Options
                static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                                               ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable |
                                               ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg |
                                               ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                                               ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY;

                //                if (ImGui::BeginTable("Timing Table", 7, flags, ImVec2(0.0f, TEXT_BASE_HEIGHT * 4),
                //                0.0f))
                if (ImGui::BeginTable("Timing Table", 7, flags, ImVec2(0.0f, 0), 0.0f))
                {
                    ImGui::TableSetupColumn("Name",
                                            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f,
                                            TimingTableColumnId_Name);
                    ImGui::TableSetupColumn("Depth", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_ID);
                    ImGui::TableSetupColumn("Last", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Last);
                    ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Min);
                    ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Max);
                    ImGui::TableSetupColumn("Median", ImGuiTableColumnFlags_WidthFixed, 0.0f,
                                            TimingTableColumnId_Median);
                    ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Mean);
                    ImGui::TableSetupScrollFreeze(0, 1);  // Make row always visible
                    ImGui::TableHeadersRow();

                    if (ImGuiTableSortSpecs* sorts_specs = ImGui::TableGetSortSpecs())
                    {
                        // Note: We are not using the sort_specs->dirty mechanism because the table is rebuild every
                        // frame
                        std::sort(items.begin(), items.end(), [sorts_specs](const TimeData* a_, const TimeData* b_) {
                            auto& a = a_->stats;
                            auto& b = b_->stats;
                            for (int n = 0; n < sorts_specs->SpecsCount; n++)
                            {
                                // Here we identify columns using the ColumnUserID value that we ourselves
                                // passed to TableSetupColumn() We could also choose to identify columns based
                                // on their index (sort_spec->ColumnIndex), which is simpler!
                                const ImGuiTableColumnSortSpecs* sort_spec = &sorts_specs->Specs[n];

                                int delta = 0;
                                switch (sort_spec->ColumnUserID)
                                {
                                    case TimingTableColumnId_ID:
                                        delta = a.depth - b.depth;
                                        break;
                                    case TimingTableColumnId_Name:
                                        delta = strcmp(a.Name.c_str(), b.Name.c_str());
                                        break;
                                    case TimingTableColumnId_Last:
                                        delta = a.stat_last - b.stat_last < 0 ? -1 : 1;
                                        break;
                                    case TimingTableColumnId_Min:
                                        delta = a.stat_min - b.stat_min < 0 ? -1 : 1;
                                        break;
                                    case TimingTableColumnId_Max:
                                        delta = a.stat_max - b.stat_max < 0 ? -1 : 1;
                                        break;
                                    case TimingTableColumnId_Median:
                                        delta = a.stat_median - b.stat_median < 0 ? -1 : 1;
                                        break;
                                    case TimingTableColumnId_Mean:
                                        delta = a.stat_mean - b.stat_mean < 0 ? -1 : 1;
                                        break;
                                    default:
                                        SAIGA_EXIT_ERROR("invalid column");
                                        break;
                                }

                                if (delta > 0) return sort_spec->SortDirection == ImGuiSortDirection_Ascending;
                                if (delta < 0) return sort_spec->SortDirection == ImGuiSortDirection_Descending;
                            }

                            return a.depth < b.depth;
                        });
                    }

                    // Demonstrate using clipper for large vertical lists
                    ImGuiListClipper clipper;
                    clipper.Begin(items.size());
                    while (clipper.Step())
                    {
                        for (int row_n = clipper.DisplayStart; row_n < clipper.DisplayEnd; row_n++)
                        {
                            // Display a data item
                            TimeData* item_ = items[row_n];
                            TimeStats* item = &item_->stats;
                            ImGui::PushID(item->Name.c_str());
                            ImGui::TableNextRow();

                            ImGui::TableNextColumn();


                            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_TableRowBg));
                            static int item_is_selected;
                            if (ImGui::Selectable(
                                    item->Name.c_str(), &item_is_selected,
                                    ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap))
                            {
                            }
                            if (ImGui::IsItemHovered())
                            {
                                draw_tooltip = item_;
                            }
                            ImGui::PopStyleColor();
                            //                            ImGui::TextUnformatted(item->Name.c_str());

                            ImGui::TableNextColumn();
                            ImGui::Text("%04d", item->depth);

                            ImGui::TableNextColumn();
                            ImGui::Text("%f", item->stat_last);
                            ImGui::TableNextColumn();
                            ImGui::Text("%f", item->stat_min);
                            ImGui::TableNextColumn();
                            ImGui::Text("%f", item->stat_max);
                            ImGui::TableNextColumn();
                            ImGui::Text("%f", item->stat_median);
                            ImGui::TableNextColumn();
                            ImGui::Text("%f", item->stat_mean);


                            ImGui::PopID();
                        }
                    }

                    ImGui::EndTable();
                }
            }
            else if (current_view == 1)
            {
                // Graph
                auto total_diff = total_time.capture.second - total_time.capture.first;


                // Extract timedata from map and sort by start time to get correct
                // back to front rendering
                std::vector<FrameGraphElement> tds;
                for (auto& td : data)
                {
                    FrameGraphElement e;
                    e.data = td.second.get();
                    tds.push_back(e);
                }
                std::sort(tds.begin(), tds.end(), [](const FrameGraphElement& a, const FrameGraphElement& b) {
                    return a.data->last_measurement.first < b.data->last_measurement.first;
                });



                if (!normalize_time)
                {
                    ImGui::SliderFloat("Absolute FPS", &absolute_scale_fps, 1, 300);
                    total_diff = (1.f / absolute_scale_fps) * (1000 * 1000 * 1000);
                }


                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 canvas_pos     = ImGui::GetCursorScreenPos();
                ImVec2 canvas_size    = ImGui::GetContentRegionAvail();

                ImGuiIO& io = ImGui::GetIO();
                vec2 mouse  = io.MousePos;
                {
                    // Draw canvas
                    if (canvas_size.x < 50.0f) canvas_size.x = 50.0f;
                    if (canvas_size.y < 50.0f) canvas_size.y = 50.0f;
                    draw_list->AddRectFilled(canvas_pos,
                                             ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                                             IM_COL32(50, 50, 50, 255));
                    //                    draw_list->AddRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x,
                    //                    canvas_pos.y + canvas_size.y),
                    //                                       IM_COL32(255, 255, 255, 255));
                }



                float box_h = 30;

                for (FrameGraphElement& el : tds)
                {
                    float x1_rel = float(el.data->capture.first - total_time.capture.first) / float(total_diff);
                    float x2_rel = float(el.data->capture.second - total_time.capture.first) / float(total_diff);



                    vec2 box_pos  = canvas_pos + vec2(canvas_size.x * x1_rel, el.data->stats.depth * (box_h + 4));
                    vec2 box_size = vec2(canvas_size.x * (x2_rel - x1_rel), box_h);

                    el.box_min = box_pos;
                    el.box_max = box_pos + box_size;

                    if (el.hover(mouse))
                    {
                        el.highlighted = true;
                        draw_tooltip   = el.data;
                    }
                }

                for (FrameGraphElement& el : tds)
                {
                    ImU32 c1 = IM_COL32(100, 100, 100, 255);
                    ImU32 c2 = IM_COL32(150, 150, 150, 255);

                    ImU32 color = el.highlighted ? c2 : c1;


                    draw_list->AddRectFilled(el.box_min, el.box_max, color, 4);
                    draw_list->AddRect(el.box_min, el.box_max, IM_COL32(255, 255, 255, 255), 4);

                    vec2 ts = ImGui::CalcTextSize(el.data->stats.Name.c_str());

                    float w = el.box_max(0) - el.box_min(0);
                    float h = el.box_max(1) - el.box_min(1);


                    if (ts.x() + 4 < w)
                    {
                        vec2 text_pos = el.box_min + vec2((w - ts(0)) * 0.5, (h - ts(1) - 2) * 0.5);
                        //                    draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255),
                        //                    el.data->stats.Name.c_str());


                        draw_list->AddText(ImGui::GetFont(), 20, text_pos, ImGui::GetColorU32(ImGuiCol_Text),
                                           el.data->stats.Name.c_str());
                    }
                }
            }

            if (draw_tooltip)
            {
                auto td = draw_tooltip;
                ImGui::SetNextWindowSize(vec2(300, 400));
                ImGui::BeginTooltip();

                ImGui::Columns(2, 0, false);
                ImGui::SetColumnWidth(0, 100);
                ImGui::SetColumnWidth(1, 200);


                ImGui::TextUnformatted("Name");


                ImGui::NextColumn();
                ImGui::TextUnformatted(td->stats.Name.c_str());
                ImGui::NextColumn();


                ImGui::TextUnformatted("Begin (ns)");
                ImGui::NextColumn();
                ImGui::Text("%lu", td->last_measurement.first - total_time.last_measurement.first);
                ImGui::NextColumn();

                ImGui::TextUnformatted("End (ns)");
                ImGui::NextColumn();
                ImGui::Text("%lu", td->last_measurement.second - total_time.last_measurement.first);
                ImGui::NextColumn();



                ImGui::TextUnformatted("Duration (ms)");
                ImGui::NextColumn();
                ImGui::Text("%f", td->stats.stat_last);
                ImGui::NextColumn();


                ImGui::Separator();

                ImGui::TextUnformatted("Averaged over");
                ImGui::NextColumn();
                ImGui::Text("%d frames (all ms)", num_samples);
                ImGui::NextColumn();

                ImGui::TextUnformatted("Min/Max");
                ImGui::NextColumn();
                ImGui::Text("[%f, %f]", td->stats.stat_min, td->stats.stat_max);
                ImGui::NextColumn();

                ImGui::TextUnformatted("Sdev.");
                ImGui::NextColumn();
                ImGui::Text("%f", td->stats.stat_sdev);
                ImGui::NextColumn();

                ImGui::TextUnformatted("Median");
                ImGui::NextColumn();
                ImGui::Text("%f", td->stats.stat_median);
                ImGui::NextColumn();

                ImGui::TextUnformatted("Mean");
                ImGui::NextColumn();
                ImGui::Text("%f", td->stats.stat_mean);
                ImGui::NextColumn();

                ImGui::TextUnformatted("1000/mean");
                ImGui::NextColumn();
                ImGui::Text("%f", 1000.f / td->stats.stat_mean);
                ImGui::NextColumn();

                ImGui::Columns(1);
                ImGui::Separator();

                auto values = td->ComputeTimes();

                float graph_scale = std::max(1.f, std::round(td->stats.stat_max + 1.f));
                ImGui::PlotLines("", values.data(), values.size(), td->count % td->measurements_ms.size(), 0, 0,
                                 graph_scale, ImVec2(0, 80));
                ImGui::Text("Graph scale: %f", graph_scale);


                //                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                //                ImGui::TextUnformatted(
                //                    "Timing tables.\n"
                //                    "Click on column names to sort.\n"
                //                    "Shift Click for multi sort");
                //                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::End();
}

GLTimerSystem::TimeData::TimeData(int& current_depth, int samples)
    : measurements_ms(samples, {0, 0}), current_depth(current_depth)
{
    timer.create();
}

void GLTimerSystem::TimeData::AddTime(Measurement t)
{
    stats.stat_last                                 = (t.second - t.first) / float(1000 * 1000);
    last_measurement                                = t;
    measurements_ms[count % measurements_ms.size()] = t;
    active                                          = true;
    count++;
}

std::vector<float> GLTimerSystem::TimeData::ComputeTimes()
{
    std::vector<float> values;
    for (auto m : measurements_ms)
    {
        auto delta = m.second - m.first;

        if (m.first > 0 && delta > 0)
        {
            float delta_ms = float(delta) / (1000 * 1000);
            values.push_back(delta_ms);
        }
    }
    return values;
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

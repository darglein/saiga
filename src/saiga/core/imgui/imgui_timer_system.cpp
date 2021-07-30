/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_timer_system.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/statistics.h"
#include "saiga/core/util/tostring.h"

#include <fstream>

namespace Saiga
{
void TimerSystem::TimeData::ComputeStatistics()
{
    Statistics st(ComputeTimes());
    stat_min    = st.min;
    stat_max    = st.max;
    stat_median = st.median;
    stat_mean   = st.mean;
    stat_sdev   = st.sdev;
}


std::vector<float> TimerSystem::TimeData::ComputeTimes()
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
void TimerSystem::TimeData::Start()
{
    if (current_depth >= 0)
    {
        // This timer has already been recorded.
        SAIGA_ASSERT(active == false);
        SAIGA_ASSERT(depth == -1);
        depth      = current_depth++;
        name_stack = name_stack + name;
        timer->Start();

        // std::cout << "> " << depth << " " << name << " | " << name_stack << std::endl;
    }
}
void TimerSystem::TimeData::Stop()
{
    if (current_depth >= 0)
    {
        timer->Stop();
        current_depth--;
        SAIGA_ASSERT(name_stack.size() >= name.size());
        name_stack.resize(name_stack.size() - name.size());

        // std::cout << "< " << depth << " " << name << " | " << name_stack << std::endl;

        SAIGA_ASSERT(depth == current_depth);
        AddTime(timer->LastMeasurement());
    }
}
void TimerSystem::TimeData::AddTime(TimerSystem::Measurement t)
{
    stat_last                                       = (t.second - t.first) / float(1000 * 1000);
    last_measurement                                = t;
    measurements_ms[count % measurements_ms.size()] = t;
    active                                          = true;
    count++;
}
void TimerSystem::TimeData::ResizeSamples(int num_samples)
{
    count = 0;
    measurements_ms.clear();
    measurements_ms.resize(num_samples, {0, 0});
}

void TimerSystem::BeginFrame()
{
    if (capturing && render_window)
    {
        for (auto& st : data)
        {
            st.second->active = false;
            st.second->depth  = -1;
        }
        current_depth = 0;
    }
    else
    {
        current_depth = -1;
    }

    BeginFrameImpl();
    SAIGA_ASSERT(current_name_stack.empty());
    GetTimer("Frame", false).Start();
}


void TimerSystem::EndFrame()
{
    GetTimer("Frame", false).Stop();
    SAIGA_ASSERT(current_name_stack.empty());
    EndFrameImpl();

    if (capturing && render_window)
    {
        SAIGA_ASSERT(current_depth == 0);
    }
    // Set depth to -1 so that we don't measure time outside of begin/end sections
    current_depth = -1;
}

void TimerSystem::Imgui()
{
    SAIGA_ASSERT(current_name_stack.empty());
    TimeData& total_time = GetTimer("Frame", false);

    // if(!total_time.active) return;

    std::vector<TimeData*> timers = ActiveTimers();
    TimerSystem::Imgui(system_name, timers, &total_time);
}
TimerSystem::TimeData& TimerSystem::GetTimer(const std::string& name, bool rel_path)
{
    SAIGA_ASSERT(!name.empty());

    std::string full_name         = rel_path ? current_name_stack + name : name;
    std::shared_ptr<TimeData>& td = data[full_name];

    if (!td)
    {
        td = std::make_shared<TimeData>(CreateTimer(), current_depth, current_name_stack);
        td->ResizeSamples(num_samples);
        td->name      = name;
        td->full_name = full_name;
    }
    return *td;
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
    TimerSystem::TimeData* data;

    // Bounding box of the rectangle
    vec2 box_min = vec2::Zero();
    vec2 box_max = vec2::Zero();

    bool hover(vec2 mouse)
    {
        return mouse(0) >= box_min(0) && mouse(0) <= box_max(0) && mouse(1) >= box_min(1) && mouse(1) <= box_max(1);
    }

    bool highlighted = false;
};

void TimerSystem::Imgui(const std::string& name, ArrayView<TimeData*> timers, TimeData* total_time)
{
    if (!render_window) return;

    if (current_frame % num_samples == 0)
    {
        // recompute statistics
        for (auto& st : timers)
        {
            st->ComputeStatistics();
        }
    }

    current_frame++;

    if (ImGui::Begin(name.c_str(), &render_window))
    {
        ImGui::PushItemWidth(120);
        const std::vector<std::string> names = {"Table View", "Timeline View"};
        ImGui::PushID(4574);
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
            for (auto& st : timers)
            {
                st->ResizeSamples(num_samples);
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
                "      auto tim = timer->Measure(\"Geometry\");\n"
                "\n"
                "Table View\n"
                " - Shows all active timers in a table\n"
                " - Click on column names to sort\n"
                " - Shift Click for multi sort\n"
                "\n"
                "Timeline View\n"
                " - Plots the last captured frame\n"
                " - Other statistics are shown at mouse hover\n"
                " - Nested Timer calls are drawn below each other\n"
                " - Use the 'normalize time' checkbox to scale the width by the total frame time\n"
                "");
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }


        if (current_view == 0)
        {
            ImguiTable(timers, total_time);
        }
        else if (current_view == 1)
        {
            ImguiTimeline(timers, total_time);
        }
    }
    ImGui::End();
}

void TimerSystem::ImguiTable(ArrayView<TimeData*> timers, TimeData* total_time)
{
    TimeData* draw_tooltip = nullptr;

    // Options
    const ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable |
                                  ImGuiTableFlags_Sortable | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                                  ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY;

    if (ImGui::BeginTable("Timing Table", 7, flags, ImVec2(0.0f, 0), 0.0f))
    {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f,
                                TimingTableColumnId_Name);
        ImGui::TableSetupColumn("Depth", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_ID);
        ImGui::TableSetupColumn("Last", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Last);
        ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Min);
        ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Max);
        ImGui::TableSetupColumn("Median", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Median);
        ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthFixed, 0.0f, TimingTableColumnId_Mean);
        ImGui::TableSetupScrollFreeze(0, 1);  // Make row always visible
        ImGui::TableHeadersRow();

        if (ImGuiTableSortSpecs* sorts_specs = ImGui::TableGetSortSpecs())
        {
            // Note: We are not using the sort_specs->dirty mechanism because the table is rebuild every
            // frame
            std::sort(timers.begin(), timers.end(), [sorts_specs](const TimeData* a, const TimeData* b) {
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
                            delta = a->depth - b->depth;
                            break;
                        case TimingTableColumnId_Name:
                            delta = strcmp(a->name.c_str(), b->name.c_str());
                            break;
                        case TimingTableColumnId_Last:
                            delta = a->stat_last - b->stat_last < 0 ? -1 : 1;
                            break;
                        case TimingTableColumnId_Min:
                            delta = a->stat_min - b->stat_min < 0 ? -1 : 1;
                            break;
                        case TimingTableColumnId_Max:
                            delta = a->stat_max - b->stat_max < 0 ? -1 : 1;
                            break;
                        case TimingTableColumnId_Median:
                            delta = a->stat_median - b->stat_median < 0 ? -1 : 1;
                            break;
                        case TimingTableColumnId_Mean:
                            delta = a->stat_mean - b->stat_mean < 0 ? -1 : 1;
                            break;
                        default:
                            SAIGA_EXIT_ERROR("invalid column");
                            break;
                    }

                    if (delta > 0) return sort_spec->SortDirection == ImGuiSortDirection_Ascending;
                    if (delta < 0) return sort_spec->SortDirection == ImGuiSortDirection_Descending;
                }

                return a->name < b->name;
            });
        }

        // Demonstrate using clipper for large vertical lists
        ImGuiListClipper clipper;
        clipper.Begin(timers.size());
        while (clipper.Step())
        {
            for (int row_n = clipper.DisplayStart; row_n < clipper.DisplayEnd; row_n++)
            {
                // Display a data item
                TimeData* item = timers[row_n];
                ImGui::PushID(item->name.c_str());
                ImGui::TableNextRow();

                ImGui::TableNextColumn();


                ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_TableRowBg));
                //                static int item_is_selected;
                if (ImGui::Selectable(item->name.c_str(), false,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap))
                {
                }
                if (ImGui::IsItemHovered())
                {
                    draw_tooltip = item;
                }
                ImGui::PopStyleColor();
                //                            ImGui::TextUnformatted(item->name.c_str());

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


    if (draw_tooltip)
    {
        ImguiTooltip(draw_tooltip, total_time);
    }
}
void TimerSystem::ImguiTimeline(ArrayView<TimeData*> timers, TimeData* total_time)
{
    TimeData* draw_tooltip = nullptr;
    // Graph
    auto total_diff = total_time->last_measurement.second - total_time->last_measurement.first;


    // Extract timedata from map and sort by start time to get correct
    // back to front rendering
    std::vector<FrameGraphElement> tds;
    for (auto& td : timers)
    {
        FrameGraphElement e;
        //                    e.data = td.second.get();
        e.data = td;
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
        draw_list->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                                 IM_COL32(50, 50, 50, 255));
        //                    draw_list->AddRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x,
        //                    canvas_pos.y + canvas_size.y),
        //                                       IM_COL32(255, 255, 255, 255));
    }



    float box_h = 30;

    for (FrameGraphElement& el : tds)
    {
        float x1_rel = float(el.data->last_measurement.first - total_time->last_measurement.first) / float(total_diff);
        float x2_rel = float(el.data->last_measurement.second - total_time->last_measurement.first) / float(total_diff);



        vec2 box_pos  = canvas_pos + vec2(canvas_size.x * x1_rel, el.data->depth * (box_h + 4));
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
        if (el.data->name.empty()) continue;

        vec2 box_size = el.box_max - el.box_min;
        vec2 ts       = ImGui::CalcTextSize(el.data->name.c_str());

        if (ts.x() + 2 < box_size.x())
        {
            vec2 text_pos = el.box_min + (box_size - ts) * 0.5;
            draw_list->AddText(0, 0, text_pos, ImGui::GetColorU32(ImGuiCol_Text), el.data->name.c_str());

            // Debug box around text
            // draw_list->AddRect(text_pos, vec2(text_pos + ts), IM_COL32(255, 0, 0, 255));
        }
        else
        {
            // Try using only the first letter
            std::string name = el.data->name.substr(0, 1) + ".";
            ts               = ImGui::CalcTextSize(name.c_str());
            if (ts.x() + 2 < box_size.x())
            {
                vec2 text_pos = el.box_min + (box_size - ts) * 0.5;
                draw_list->AddText(0, 0, text_pos, ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
            }
        }
    }

    if (draw_tooltip)
    {
        ImguiTooltip(draw_tooltip, total_time);
    }
}
void TimerSystem::ImguiTooltip(TimeData* td, TimeData* total_time)
{
    ImGui::SetNextWindowSize(vec2(300, 400));
    ImGui::BeginTooltip();

    ImGui::Columns(2, 0, false);
    ImGui::SetColumnWidth(0, 100);
    ImGui::SetColumnWidth(1, 200);


    ImGui::TextUnformatted("Name");
    ImGui::NextColumn();
    ImGui::TextUnformatted(td->name.c_str());
    ImGui::NextColumn();

    ImGui::TextUnformatted("Full Name");
    ImGui::NextColumn();
    ImGui::TextUnformatted(td->full_name.c_str());
    ImGui::NextColumn();


    ImGui::TextUnformatted("Begin (ns)");
    ImGui::NextColumn();
    ImGui::Text("%lu", td->last_measurement.first - total_time->last_measurement.first);
    ImGui::NextColumn();

    ImGui::TextUnformatted("End (ns)");
    ImGui::NextColumn();
    ImGui::Text("%lu", td->last_measurement.second - total_time->last_measurement.first);
    ImGui::NextColumn();



    ImGui::TextUnformatted("Duration (ms)");
    ImGui::NextColumn();
    ImGui::Text("%f", td->stat_last);
    ImGui::NextColumn();


    ImGui::Separator();

    ImGui::TextUnformatted("Averaged over");
    ImGui::NextColumn();
    ImGui::Text("%d frames (all ms)", num_samples);
    ImGui::NextColumn();

    ImGui::TextUnformatted("Min/Max");
    ImGui::NextColumn();
    ImGui::Text("[%f, %f]", td->stat_min, td->stat_max);
    ImGui::NextColumn();

    ImGui::TextUnformatted("Sdev.");
    ImGui::NextColumn();
    ImGui::Text("%f", td->stat_sdev);
    ImGui::NextColumn();

    ImGui::TextUnformatted("Median");
    ImGui::NextColumn();
    ImGui::Text("%f", td->stat_median);
    ImGui::NextColumn();

    ImGui::TextUnformatted("Mean");
    ImGui::NextColumn();
    ImGui::Text("%f", td->stat_mean);
    ImGui::NextColumn();

    ImGui::TextUnformatted("1000/mean");
    ImGui::NextColumn();
    ImGui::Text("%f", 1000.f / td->stat_mean);
    ImGui::NextColumn();

    ImGui::Columns(1);
    ImGui::Separator();

    auto values = td->ComputeTimes();

    float graph_scale = std::max(1.f, std::round(td->stat_max + 1.f));
    ImGui::PlotLines("", values.data(), values.size(), td->count % td->measurements_ms.size(), 0, 0, graph_scale,
                     ImVec2(0, 80));
    ImGui::Text("Graph scale: %f", graph_scale);
    ImGui::EndTooltip();
}
void TimerSystem::PrintTable(std::ostream& strm)
{
    for (auto t : data)
    {
        if (t.second->active) t.second->ComputeStatistics();
    }

    Table tab({30, 5, 10, 10, 10, 10}, strm);
    tab.setFloatPrecision(5);
    tab << "Name"
        << "N"
        << "Mean"
        << "Median"
        << "Min"
        << "Max";

    auto timers = ActiveTimers();
    for (auto& st : timers)
    {
        auto values = st->ComputeTimes();
        Statistics s(values);
        tab << st->name << values.size() << s.mean << s.median << s.min << s.max;
    }
}
std::vector<TimerSystem::TimeData*> TimerSystem::ActiveTimers()
{
    std::vector<TimeData*> result;


    for (auto& st : data)
    {
        if (st.second->active)
        {
            result.push_back(st.second.get());
        }
    }

    std::sort(result.begin(), result.end(), [](TimeData* a, TimeData* b) { return a->stat_mean > b->stat_mean; });


    return result;
}
}  // namespace Saiga

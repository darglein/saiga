/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/app/DebugUi.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/app/DebugUiStyle.h"
#include "saiga/engine/core/Log.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace Saiga
{
namespace Engine
{
namespace
{
std::string lower(std::string s)
{
    for (char& c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

// Fixed windows: placed every frame, never moved, resized (the splitter is our own), docked or saved by ImGui. They
// come to the front like any window when clicked (not forced every frame: that hid the popups opened inside them)
constexpr ImGuiWindowFlags kFixedWindow = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking;
}  // namespace

DebugUi::DebugUi(McpServer* server, std::string stateFile)
    : consolePtr(std::make_unique<DebugConsole>(server, engineLog())), file(std::move(stateFile))
{
    for (const char* name : {"Game", "Performance", "Engine", "Console", "Lighting"}) tabs.push_back(Tab{name, {}, {}});
}

DebugUi::~DebugUi()
{
    save();
}

void DebugUi::select(int tab)
{
    if (tab >= 0 && tab < tabCount())
    {
        selectedTab = tab;
        pendingTab.clear();
        pendingTabIndex = -1;
    }
    else if (tab >= tabCount())
    {
        // A game tab that is added later (SAIGA_DEBUG_TAB before renderHooks.init); it replaces a saved tab name
        pendingTabIndex = tab;
        pendingTab.clear();
    }
}

int DebugUi::findTab(const std::string& name) const
{
    for (int i = 0; i < tabCount(); ++i)
        if (lower(tabs[size_t(i)].name) == lower(name)) return i;
    return -1;
}

int DebugUi::addTab(const std::string& name, std::function<void()> draw)
{
    tabs.push_back(Tab{name, std::move(draw), {}});
    const int index = tabCount() - 1;
    // The saved or requested tab of a game tab (the state is read before the game adds its tabs)
    if (index == pendingTabIndex || (!pendingTab.empty() && lower(pendingTab) == lower(name))) select(index);
    return index;
}

void DebugUi::setTabContent(DebugTab tab, std::function<void()> draw)
{
    tabs[size_t(tab)].draw = std::move(draw);
}

void DebugUi::addSection(DebugTab tab, const std::string& title, std::function<void()> draw, bool defaultOpen)
{
    addSection(int(tab), title, std::move(draw), defaultOpen);
}

void DebugUi::addSection(int tab, const std::string& title, std::function<void()> draw, bool defaultOpen)
{
    if (tab < 0 || tab >= tabCount()) return;
    tabs[size_t(tab)].sections.push_back({title, std::move(draw), defaultOpen});
}

void DebugUi::markItem(const std::string& name)
{
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    markedItems[name] = ImVec2((a.x + b.x) / 2, (a.y + b.y) / 2);
}

bool DebugUi::itemCenter(const std::string& name, ImVec2& center) const
{
    auto it = markedItems.find(name);
    if (it == markedItems.end()) return false;
    center = it->second;
    return true;
}

bool DebugUi::section(const std::string& key, const char* title, bool defaultOpen)
{
    endSubSection();
    auto it = sections.find(key);
    if (it == sections.end()) it = sections.emplace(key, defaultOpen).first;
    ImGui::PushID(key.c_str());
    const bool open = DebugUiStyle::section(title, it->second);
    ImGui::PopID();
    return open;
}

bool DebugUi::subSection(const std::string& key, const char* title, bool defaultOpen)
{
    endSubSection();
    auto it = sections.find(key);
    if (it == sections.end()) it = sections.emplace(key, defaultOpen).first;
    ImGui::PushID(key.c_str());
    ImGui::SetNextItemOpen(it->second, ImGuiCond_Always);
    it->second = ImGui::TreeNodeEx(title, ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen);
    ImGui::PopID();
    if (it->second)
    {
        subIndent = ImGui::GetFontSize();
        ImGui::Indent(subIndent);
    }
    return it->second;
}

void DebugUi::endSubSection()
{
    if (subIndent <= 0) return;
    ImGui::Unindent(subIndent);
    subIndent = 0;
}

InputState DebugUi::handleKeys(const InputState& now, const InputState& previous, bool captured)
{
    if (!enabledFlag) return now;
    InputState out = now;
    for (int k = 0; k < 10; ++k)
    {
        const size_t key = size_t(kKeyF1 + k);
        if (now.keys[key] && !previous.keys[key]) select(k);
        out.keys[key] = false;
    }
    const size_t pause = size_t(kKeyPause);
    if (!captured && now.keys[pause] && !previous.keys[pause] && pauseKey) pauseKey();
    const size_t grave = size_t(kKeyGrave);
    if (!captured && now.keys[grave] && !previous.keys[grave])
    {
        select(DebugTab::Console);
        // Not in this frame: the key's character is in this frame's ImGui input and would be typed into the line
        focusConsoleDelay = 2;
    }
    return out;
}

float DebugUi::barHeight() const
{
    return ImGui::GetCurrentContext() ? ImGui::GetFontSize() + 10 : 0;
}

float DebugUi::panelSize(float em) const
{
    const float display = ImGui::GetIO().DisplaySize.x;
    const float w       = panelWidth > 0 ? panelWidth : 28 * em;
    return std::clamp(w, 0.15f * display, 0.7f * display);
}

void DebugUi::draw()
{
    if (!enabledFlag || !ImGui::GetCurrentContext()) return;
    const float em = ImGui::GetFontSize();
    DebugUiStyle::push();
    drawBar(em);
    if (selectedTab != int(DebugTab::Game)) drawPanel(em);
    DebugUiStyle::pop();
    if (focusConsoleDelay > 0 && selectedTab != int(DebugTab::Console)) focusConsoleDelay = 0;
    saveIfChanged();
}

void DebugUi::drawBar(float em)
{
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const float height   = barHeight();
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(display.x, height), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.6f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 1));
    const ImGuiWindowFlags flags = kFixedWindow | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("##debug bar", nullptr, flags))
    {
        for (int i = 0; i < tabCount(); ++i)
        {
            if (i > 0) ImGui::SameLine(0, 2);
            const int key = tabKey(i);
            std::string label =
                (key > 0 ? "F" + std::to_string(key) + " " : std::string()) + tabs[size_t(i)].name + "##tab";
            const bool sel = i == selectedTab;
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  sel ? ImGui::GetColorU32(ImGuiCol_ButtonActive) : IM_COL32(0, 0, 0, 0));
            ImGui::PushID(i);
            if (ImGui::Button(label.c_str())) select(i);
            ImGui::PopID();
            ImGui::PopStyleColor();
        }
        if (summary)
        {
            ImGui::SameLine(0, 2 * em);
            const float avail = ImGui::GetContentRegionAvail().x;
            if (avail > 2 * em) summary(avail);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
}

void DebugUi::drawPanel(float em)
{
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const float top      = barHeight();
    const float size     = panelSize(em);
    // Docked at the right edge, from below the bar to the bottom
    const ImVec2 pos(display.x - size, top);
    const ImVec2 extent(size, std::max(display.y - top, 1.0f));
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(extent, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.95f);
    Tab& tab = tabs[size_t(selectedTab)];
    if (ImGui::Begin("##debug panel", nullptr, kFixedWindow | ImGuiWindowFlags_NoScrollbar))
    {
        // No title: the bar shows the selected tab
        // Read once: a section may select another tab while it is drawn (BeginChild and EndChild must pair)
        const bool console = selectedTab == int(DebugTab::Console);
        if (console)
        {
            bool focus = false;
            if (focusConsoleDelay > 0 && --focusConsoleDelay == 0) focus = true;
            consolePtr->draw(focus);
        }
        else if (ImGui::BeginChild("##content", ImVec2(0, 0), ImGuiChildFlags_None))
        {
            ImGui::PushID(selectedTab);
            if (tab.draw) tab.draw();
            for (auto& sec : tab.sections)
                if (section(lower(tab.name) + "/section/" + sec.title, sec.title.c_str(), sec.defaultOpen)) sec.draw();
            if (!tab.draw && tab.sections.empty()) DebugUiStyle::emptyState("nothing to show");
            ImGui::PopID();
        }
        if (!console) ImGui::EndChild();
    }
    ImGui::End();
    drawSplitter(em, pos, extent);
}

void DebugUi::drawSplitter(float em, ImVec2 pos, ImVec2 extent)
{
    // The panel's width: a thin window of its own on the panel's left edge (the panel clips its padding); the panel
    // itself cannot be moved
    const float t = 6;
    const ImVec2 p(pos.x - t / 2, pos.y), s(t, extent.y);
    ImGui::SetNextWindowPos(p, ImGuiCond_Always);
    ImGui::SetNextWindowSize(s, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1, 1));
    if (ImGui::Begin("##debug splitter", nullptr,
                     kFixedWindow | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
    {
        ImGui::InvisibleButton("##split", s);
        const bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
        if (hovered || active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(p.x + t / 2, p.y), ImVec2(p.x + t / 2, p.y + s.y),
            hovered || active ? ImGui::GetColorU32(ImGuiCol_SeparatorHovered) : IM_COL32(255, 255, 255, 40),
            hovered || active ? 3.0f : 1.0f);
        if (active) panelWidth = panelSize(em) - ImGui::GetIO().MouseDelta.x;
        // Double-click: the default width
        if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) panelWidth = 0;
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
}

// ----------------------------------------------------------------------------------------------------- State

json11::Json DebugUi::state() const
{
    json11::Json::object sec;
    for (const auto& [k, v] : sections) sec[k] = v;
    json11::Json::array history, hidden;
    for (const auto& h : consolePtr->history) history.push_back(h);
    for (const auto& c : consolePtr->hiddenCategories) hidden.push_back(c);
    return json11::Json::object{
        {"tab", pendingTab.empty() ? tabs[size_t(selectedTab)].name : pendingTab},
        {"panel_width", double(panelWidth)},
        {"sections", sec},
        {"console", json11::Json::object{{"history", history},
                                         {"hidden_categories", hidden},
                                         {"mcp_only", consolePtr->mcpOnly},
                                         {"search", consolePtr->search}}},
    };
}

void DebugUi::setState(const json11::Json& j)
{
    if (!j.is_object()) return;
    const int t = findTab(j["tab"].string_value());
    if (t >= 0)
        selectedTab = t;
    else
        pendingTab = j["tab"].string_value();  // a game tab, added later
    panelWidth = float(std::max(j["panel_width"].number_value(), 0.0));
    for (const auto& [k, v] : j["sections"].object_items())
        if (v.is_bool()) sections[k] = v.bool_value();
    const json11::Json& c = j["console"];
    consolePtr->history.clear();
    for (const auto& h : c["history"].array_items())
        if (h.is_string()) consolePtr->history.push_back(h.string_value());
    while (consolePtr->history.size() > DebugConsole::kHistorySize)
        consolePtr->history.erase(consolePtr->history.begin());
    consolePtr->hiddenCategories.clear();
    for (const auto& h : c["hidden_categories"].array_items())
        if (h.is_string()) consolePtr->hiddenCategories.insert(h.string_value());
    consolePtr->mcpOnly = c["mcp_only"].bool_value();
    consolePtr->search  = c["search"].string_value();
}

void DebugUi::load()
{
    if (file.empty()) return;
    std::ifstream in(file);
    if (!in) return;
    std::stringstream ss;
    ss << in.rdbuf();
    std::string err;
    json11::Json j = json11::Json::parse(ss.str(), err);
    if (!err.empty())
    {
        SAIGA_LOG("app", "warning: %s is not valid JSON (%s); the debug UI starts with its defaults", file.c_str(),
                  err.c_str());
        // Written again only after a change (a hand-edited file is not replaced by the defaults at exit)
        savedState = state().dump();
        return;
    }
    setState(j);
    savedState = state().dump();
}

void DebugUi::save()
{
    if (file.empty()) return;
    const std::string s = state().dump();
    if (s == savedState) return;
    std::ofstream out(file);
    if (!out) return;
    out << s << "\n";
    out.flush();
    // Only a complete write counts (a failed one is tried again at the next change check)
    if (out) savedState = s;
}

void DebugUi::saveIfChanged()
{
    if (file.empty()) return;
    const double now = ImGui::GetTime();
    if (now - lastSaveCheck < 1.0) return;
    lastSaveCheck = now;
    save();
}

}  // namespace Engine
}  // namespace Saiga

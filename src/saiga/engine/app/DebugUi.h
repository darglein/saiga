/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/app/DebugConsole.h"
#include "saiga/engine/app/InputMap.h"
#include "saiga/export.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
class McpServer;

namespace Engine
{
// The engine's tabs (indices of DebugUi's tabs; game tabs follow from 4 on)
enum class DebugTab : int
{
    Game        = 0,
    Performance = 1,
    Engine      = 2,
    Console     = 3
};

/**
 * The developer UI of the engine app (docs/engine_design.md §17.1; S19c): a one-line bar at the top of the screen with
 * the tabs (F1 Game, F2 Performance, F3 Engine, F4 Console, game tabs on F5..F10) and a performance summary, and one
 * panel docked at the right edge of the screen that shows the selected tab. The tabs are tabs, not toggles: an
 * F-key selects its tab, the selected tab's key changes nothing. F1 Game shows no panel (the game alone, with its own
 * GUI). The panel is never a floating window and cannot be moved: it is placed at the right edge every frame from the
 * display size; only its width can be dragged at its left edge. The grave key (`) selects the console and focuses its
 * command line.
 *
 * The state (selected tab, panel width, open sections, the console's filters and history) is kept in a JSON file
 * (empty name: not kept). Exists in every app mode (also without ImGui or a device), so games can register tabs and
 * sections unconditionally; disabled it draws nothing and passes the keys through. Main thread.
 */
class SAIGA_ENGINE_API DebugUi
{
   public:
    // server: the console's tools (may be null); stateFile: where the state is kept (empty: nowhere)
    explicit DebugUi(McpServer* server = nullptr, std::string stateFile = "");
    ~DebugUi();
    DebugUi(const DebugUi&)            = delete;
    DebugUi& operator=(const DebugUi&) = delete;

    void setEnabled(bool e) { enabledFlag = e; }
    bool enabled() const { return enabledFlag; }

    // ---- Tabs
    int tabCount() const { return int(tabs.size()); }
    const std::string& tabName(int tab) const { return tabs[size_t(tab)].name; }
    // The tab's F-key number (1..10), 0 if it has none
    int tabKey(int tab) const { return tab < 10 ? tab + 1 : 0; }
    int selected() const { return selectedTab; }
    bool isSelected(DebugTab t) const { return selectedTab == int(t); }
    void select(int tab);
    void select(DebugTab t) { select(int(t)); }
    // By name, case-insensitive; -1 if unknown
    int findTab(const std::string& name) const;
    // A game tab (F5..F10 while keys are left; later tabs only by click), shown in the panel; returns its index
    int addTab(const std::string& name, std::function<void()> draw);
    // The content of an engine tab (the app sets Performance and Engine; the console is built in)
    void setTabContent(DebugTab tab, std::function<void()> draw);
    // A section appended to a tab, drawn after the tab's content in registration order, with a persisted header
    // (collapsed at first unless defaultOpen: the details a tab does not need at a glance)
    void addSection(DebugTab tab, const std::string& title, std::function<void()> draw, bool defaultOpen = false);
    void addSection(int tab, const std::string& title, std::function<void()> draw, bool defaultOpen = false);
    // The performance summary on the right of the bar; gets the width it may use
    void setSummary(std::function<void(float width)> draw) { summary = std::move(draw); }
    // Clicking the summary selects this tab
    static constexpr DebugTab kSummaryTab = DebugTab::Performance;

    // The sections games added to a tab
    int sectionCount(DebugTab tab) const { return int(tabs[size_t(tab)].sections.size()); }
    // A section header whose open state is kept in the state file (key: unique per tab, e.g. "engine/Simulation")
    bool section(const std::string& key, const char* title, bool defaultOpen = true);

    // ---- Keys
    // now and previous: the raw device states of this and the last frame (previous unfiltered, so a held F-key is one
    // press). F1..F10 select tabs (also while ImGui has the keyboard: they type no text), the grave key the console
    // and the Pause key pauses or resumes the world (both only while not captured). Returns now with F1..F10 cleared
    // (for the input map); disabled: now unchanged
    InputState handleKeys(const InputState& now, const InputState& previous, bool captured);
    static constexpr int kKeyF1 = 290, kKeyGrave = 96, kKeyPause = 284;
    // Called on a press of the Pause key (not while captured; the app pauses or resumes the world)
    void setPauseKey(std::function<void()> fn) { pauseKey = std::move(fn); }

    // ---- Drawing (in an ImGui frame)
    void draw();
    // The bar's height in pixels (game windows go below it)
    float barHeight() const;

    DebugConsole& console() { return *consolePtr; }

    // ---- State
    // Reads the state file (no file: defaults); writes it (also called by the destructor and, at most once per
    // second when something changed, by draw())
    void load();
    void save();
    const std::string& stateFile() const { return file; }
    // The state as JSON (tests)
    json11::Json state() const;
    void setState(const json11::Json& j);

   private:
    struct Tab
    {
        std::string name;
        std::function<void()> draw;
        struct Section
        {
            std::string title;
            std::function<void()> draw;
            bool defaultOpen = false;
        };
        std::vector<Section> sections;
    };
    void drawBar(float em);
    void drawPanel(float em);
    void drawSplitter(float em, ImVec2 pos, ImVec2 size);
    void saveIfChanged();
    float panelSize(float em) const;

    std::vector<Tab> tabs;
    std::unique_ptr<DebugConsole> consolePtr;
    std::function<void(float)> summary;
    std::function<void()> pauseKey;
    std::map<std::string, bool> sections;
    std::string file;
    bool enabledFlag      = true;
    int selectedTab       = 0;
    float panelWidth      = 0;  // pixels; 0: the default
    int focusConsoleDelay = 0;
    // A selected game tab that does not exist yet (the state and SAIGA_DEBUG_TAB are applied before the game adds its
    // tabs): selected when it is added
    std::string pendingTab;
    int pendingTabIndex = -1;  // frames until the console's command line gets the focus
    std::string savedState;
    double lastSaveCheck = 0;
};

}  // namespace Engine
}  // namespace Saiga

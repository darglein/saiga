# S19c: Debug UI

Status: implemented (PR pending).

Sections follow the stage plan template of S05.

## Goal

Every engine game gets one developer UI instead of loose ImGui windows:

- **The top bar**: a thin bar at the top of the screen, with tabs on the left (`F1 Game`, `F2 Performance`, `F3 Engine`,
  `F4 Console`) and a compact performance summary on the right (fps, frame, CPU, GPU, step).
- **Real tabs**: exactly one tab is selected, and an F-key selects its tab (never a toggle). **F1 Game** is the game
  alone: no debug panel, only the top bar and the game's own GUI over the view. **F2–F4** share one docked side panel
  below the bar.
- **No floating windows** in the engine tabs: the panel is a fixed side bar (left by default; right or bottom
  selectable), and F2–F4 share one visual language.
- **The console** merges the engine log with a command line. The commands are the MCP tools, run locally.
- **The MCP call log**: every MCP tool call is logged in the console, whether it came from an agent or from the console,
  so a developer sees what an agent is doing to the game while it happens.

The entity inspector and editor, the camera tools and the profile "freeze" follow in S19d (Out of scope).

## Depends on

- S13: `EngineApp`, `RenderScene` with ImGui, the frame and GPU profiles.
- S14: the app tools, `answerDeadline`.
- S10: `WorldApi`, `Profiler`, `LogRing`/`SAIGA_LOG`.
- PR #66: the engine statistics (`FrameStatistics`, the "Engine statistics" window and its overlay; F1 today).
- PR #67: the swap chain change. This stage is stacked on it.

## Rules that apply

- R-MCP-1: the console adds no tools; it calls the registered ones.
- R-DET-1: the UI changes the world only through `WorldApi`, between steps. Wall-clock time is used only for display.
- §5.2: `saiga_vulkan` never includes engine headers. The UI lives in `saiga_engine`; `McpServer` stays engine-free in
  `saiga_core`; the world prototype's GUI is reached through game hooks.
- R-TEST-1, R-DOC-1/2, R-FILE-1.
- AGENTS.md conventions: export macros (no exported variables, only functions), MSVC, clang-format.

## Scope

### 1. The top bar, the panels and the keys (`engine/app/DebugUi.{h,cpp}`, `EngineApp`)

`DebugUi` is owned by `EngineApp` (`EngineApp::debugUi()`). `renderFrame`'s gui callback calls `DebugUi::draw()` in
place of `statisticsWindow()`/`statisticsOverlay()`. `DebugUi` is a member declared before `server`, so it is
destroyed after it (the destructor body stops the server, and members are destroyed after the body), because the
server's pending local calls refer to the console.

- **Always there**: `DebugUi` exists in every mode (also with `debugUi = false`, `render.imgui = false`, `--benchmark`
  and `noRender`), so games can call `app.debugUi().addTab(...)` in `renderHooks.init` unconditionally and the queued
  actions work in `EngineAppNoRender` tests. When off it draws nothing and lets the keys through.
- **Engine content stays in `EngineApp`**: the Performance tab and the bar's summary need `EngineApp`'s statistics
  state (`shownStatistics`, `refreshShownStatistics()`, the verdict, the graph helpers). `EngineApp` registers them as
  the Performance tab's draw callback and as the bar's summary callback (`DebugUi::setSummary(fn)`); the Engine tab is
  registered by `EngineApp` the same way. `DebugUi` itself only knows tabs, keys, the panel, the persistence and the
  console. The PR #66 drawing code moves from `EngineApp.cpp` into `engine/app/StatisticsPanel.{h,cpp}` (functions
  taking the statistics state), so `EngineApp.cpp` does not grow.

- **Top bar**: a window across the top, one line high (font size + 2 × 4 px padding), background alpha 0.6. From the
  left: the tab buttons with their key (`F1 Game` …; the selected tab highlighted), and on the right the performance
  summary of §2.
  Clicking a tab button does the same as pressing its key.
- **Real tabs**: there is always exactly one selected tab (F1 at the first start). F1 Game shows no panel: the view is
  the game's, with its own GUI (§5). F2–F4 and the game's extra tabs (F5–F10) show their content in the one panel
  (§1.1); switching between them never moves the panel.
- **Keys** (`InputState DebugUi::handleKeys(const InputState& now, const InputState& previous, bool captured)`, a pure
  function, testable without a window): `previous` is the raw, unfiltered state of the last frame (`previousKeys`), so
  a held F-key is one press; the return value is `now` with F1–F10 cleared, which the app passes on:
  - F1–F4 select their tab on the press edge, and F5–F10 the game's own tabs. Pressing the key of the selected tab
    changes nothing (tabs, not toggles); F1 is the way back to the plain game view. The F-keys also work while ImGui has the keyboard (they never type text), so a focused
    console input does not trap the user.
  - Grave (`` ` ``) selects the console and focuses its input, only while ImGui does not capture the keyboard. The opening
    press is not typed into the field: the focus request takes effect in the next frame.
  - `EngineApp::applyPlayerInput` calls `handleKeys` with the raw states and gives the returned (filtered) state to
    `updateInputMap` and `lastDeviceState`; `previousKeys` keeps the raw keys. The input map never sees F1–F10 while
    the debug UI is on.
  - ESC first leaves a focused input (ImGui's own handling). It quits only when nothing is captured, as today.
  - G still hides all ImGui, including the bar. While it is hidden the F-keys still switch the tab, and the UI shows
    again only with G.
  - F12 stays the screenshot.
  - `EngineAppSetup::debugUi = false` turns the UI off (keys go to the input map, PR #66's window and overlay are not
    drawn either).
- **State**: `debug_ui.ini` in the working directory, like ImGui's `imgui.ini` (the path is a `DebugUi` constructor
  argument, so tests use a temporary directory). It is not in the data directory,
  because the data directory is the source tree's `data/<name>`. It stores:
  - the selected tab, the panel's edge and its size,
  - the collapsed sections,
  - the console filters and the command history (the last 100 commands).

  It is read at start and written at quit, and at most once per second when something changed. Persistence is off
  (nothing read or written) when headless, with a screenshot option (`AppOptions::screenshot`: `SAIGA_SCREENSHOT` or
  `--screenshot`), with `--benchmark` and when
  `EngineAppSetup::debugUiPersistence` is false (tests). This keeps screenshots and tests independent of the last
  session.
- **Start tab**: F1 Game without an ini (and always when persistence is off, unless overridden): screenshots and tests
  show the game. `--stats` opens the Performance tab. `SAIGA_DEBUG_TAB=<n>` opens tab n (screenshots and checks). Both
  override the ini. `EngineApp::showStatistics(bool)` is removed, and so are its calls in the games (platformer, flight,
  racing, world): the constructor already applies `--stats`. This intentionally changes a default: racing and world no
  longer open the statistics at start (the top bar shows the summary; §17.1 is updated).
- **Registration**:
  - `DebugUi::addTab(name, draw)` adds a game tab on the next free key (F5…F10), shown in the same panel (for game
    debug content that wants the docked layout, e.g. AI or physics state).
  - `DebugUi::addSection(tab, title, draw)` appends a collapsible section to an engine tab. Sections are drawn in
    registration order after the engine's own content.
  - The engine tabs are `DebugTab::Game`, `Performance`, `Engine` and `Console`.
- **Docking**: every debug UI window sets `ImGuiWindowFlags_NoDocking` (docking is enabled in `imgui_saiga.cpp`), so
  game windows cannot dock into the bars.
- **Text**: ASCII only (`->`, `<-`, `v`), because ImGui's default font has only Basic Latin and Latin-1, and MSVC
  builds without `/utf-8`.

### 1.1 The panel and the look of the tabs (`engine/app/DebugUiStyle.{h,cpp}`)

The panel is one fixed side bar, not a floating window, shared by all tabs except F1. It is an ImGui window with the flags
`NoTitleBar`, `NoMove`, `NoResize` (the splitter is our own), `NoCollapse`, `NoSavedSettings`, `NoBringToFrontOnFocus`
and `NoDocking`, positioned and sized every frame (`ImGuiCond_Always`) from the display size, so it cannot get lost and
follows window resizes.

- **Edge**: left side bar by default, from below the top bar to the screen bottom. A small button in the panel's
  toolbar switches it between left, right and bottom (stored in `debug_ui.ini`); all tabs use the chosen edge. The
  default width is 32 × font size (about 520 px at the default font), the bottom height 35% of the screen.
- **Sizing**: the inner edge is a splitter: a 4 px drag handle, highlighted on hover, with the resize cursor. One size
  for all tabs (switching tabs never changes the layout), clamped to 15–70% of the screen dimension. Double-clicking the
  splitter resets the default. The panel never covers the top bar.
- **Content fits the panel's shape**: tabs lay out for a narrow, tall panel (sections stacked vertically). On the bottom
  edge, tabs with wide content use columns (the Performance tab: summary, graph and tables side by side, §3).
- **Over the game**: the 3D view keeps its size. The panels lie over it with background alpha 0.92 and a 1 px line on
  the inner edge, so the game stays visible next to them.
- **One visual language** (`DebugUiStyle`, used by every engine tab and offered to games):
  - **Sections**: a full-width, slightly lighter header strip (`DebugUiStyle::section(title)`), collapsible, with the
    state kept in `debug_ui.ini`.
  - **Properties**: two-column tables, a dimmed label on the left and the value or widget on the right at full width
    (`beginProperties()` / `property(label)` / `endProperties()`). No free-floating text lines.
  - **Numbers**: right-aligned, with fixed decimals per unit (ms 2, fps 0, % 0, m 2). The unit is in the column header
    or in dimmed text after the value. A value over its threshold is drawn in the warn color.
  - **Colors**: PR #66's frame, CPU, GPU and warn colors as named constants, plus `mcp` (cyan) and `api` (yellow) for
    the console. The same color means the same quantity in every tab, the graphs and the top bar.
  - **Style**: set once for the debug UI and pushed and popped around it, so games keep their own: window padding 8,
    item spacing 6 × 4, rounding 0 for bars and 3 for buttons, a frame border on inputs.
  - **Toolbar**: each tab's actions (copy JSON, pause, step, clear) are small buttons in one row at the top of the
    panel, next to the tab's title, never in between content.
  - **Empty states**: a dimmed hint in the middle of the panel ("No game GUI").
  - **Scaling**: sizes are in units of `ImGui::GetFontSize()`, not raw pixels, so the layout scales with the font.

### 2. The performance summary in the top bar

It replaces PR #66's overlay line and reads the same `shownStatistics` (refreshed every 0.5 s, so the numbers can be
read). It shows, in this order:

1. **fps**: the presented frames per second over the statistics window.
2. **frame ms**: the median frame interval; with vsync, of the shown frames.
3. **CPU** and **GPU**: the median busy time in ms.
4. **step**: the median of the world's `step` scope in ms.

A warning in the warn color is added when there are missed refreshes or long frames per second > 0, or dropped steps in
the window. A call running for more than 0.5 s adds `MCP: <tool>`. `screenshot` itself is never shown, so that
`screenshot gui=true` images do not show their own call.

The colors are PR #66's. Hovering shows the vsync rate, the 1% low and the max frame, and clicking opens the
Performance tab. When the bar is too narrow, items are dropped from the right (step, GPU, CPU); fps always stays.

### 3. The tabs

| Tab | Content |
|---|---|
| **F1 Game** | No panel: the game view with the game's own GUI, `GameRenderHooks::gui`, drawn as the game wants (§5). |
| **F2 Performance** | Stacked sections in the side bar (three columns on the bottom edge):<br>• Summary: PR #66's verdict, busy shares, fps / 1% low / max / missed, and the 3/10/30 s window selector.<br>• Frame graph at the panel's full width.<br>• Tables: PR #66's frame table, the step profile scopes (avg/p95/share) and the GPU sections.<br>Toolbar: "copy JSON" (the `get_profile` answer to the clipboard). The world prototype adds "World passes" as a section (§5). |
| **F3 Engine** | Four sections:<br>• **Simulation**: pause/resume, step 1, step 10, time scale (0.1–4), step index, dropped steps, entities, game objects, with `--coop` "client world".<br>• **Systems**: per phase the systems with avg/p95 ms from the step profile.<br>• **Rendering**: tone map and exposure (moved from the statistics window); vsync and MSAA shown read-only; the game's render settings as sections (the world prototype's "World", §5).<br>• **Camera and input**: view camera position, yaw/pitch, fovy, held by a tool; the input map's actions and held buttons; the local player. |
| **F4 Console** | §4. |

**UI actions run between steps, not in the gui callback.** The gui callback runs inside `RenderScene::renderFrame`
after the frame was latched and before recording. That is where `World::step` must not run: the frame statistics
already counted the frame's steps, and the extract has already caught up. So the UI only queues its actions (pause,
step, time scale, exposure). `EngineApp::frame` applies them right after `server->poll()`, through `WorldApi::call`
(`api` log lines as for the tools) or the scene's setters. Reads stay in the gui callback.

### 4. The console (`engine/app/DebugConsole.{h,cpp}`)

The console logic is separate from its drawing. `DebugConsole` parses, completes, runs commands through `McpServer`
and keeps the view buffer, all without ImGui. `DebugConsole::draw()` is the ImGui part.

- **Log view**:
  - The lines of `engineLog()` since the console's start, newest at the bottom. The view auto-scrolls unless the user
    scrolled up, in which case a "v N new" button appears.
  - Each line shows its time (s since start), `[category]` and message.
  - Filters: a category multi-select (every category seen so far), a case-insensitive text search and an "MCP only"
    toggle.
  - Colors by category: `mcp` cyan, `console` white, `api` yellow. A message that starts with `error` or `warning` is
    red or orange.
  - Right-click copies a line.
  - The view reads `LogRing::since(last)` once per frame into its own buffer (at most 4096 lines). Command results go
    into that buffer too (see Execution).
- **LogRing**: `Line` gains `double time`, steady-clock seconds since the process' first log line, filled in by
  `LogRing::add` (so every ring's lines carry it). The `logs` tool returns it as an added key. There is no step field, because worker threads and `--coop`
  host/client steps make it ambiguous.
- **Command line**: at the bottom, with a history (Up/Down) and Tab completion of tool names (unique prefix: complete;
  ambiguous: list the candidates). The syntax:
  - `tool`: no arguments.
  - `tool {json}`: the JSON object as arguments.
  - `tool key=value key=value`: arguments by name. A value that parses as JSON (`3`, `true`, `[1,2,3]`, `"a b"`) is
    taken as that JSON, anything else as a string (`class=car`).
  - Built-ins (not tools): `help [tool]` (the description and the arguments from the input schema), `tools` (the list)
    and `clear` (clears the view, not the ring).
- **Execution**:
  - The console calls tools through `McpServer::callLocal` (§4.1), so a command behaves exactly like an agent's call:
    the same argument checks and the same deferred answers (`screenshot`, `measure_performance(frames)`).
  - Only the command is logged (as an `mcp` line, §4.1). The answer goes into the console's view buffer only, never
    into the ring or stdout: an error in red, text as text, JSON pretty-printed and folded after 40 lines ("show all"
    expands it), an image as `[image WxH]`.
  - While a call is pending the input stays usable, and the command's line shows `... running`.

### 4.1 MCP call log (`core/util/mcp/McpServer.{h,cpp}`)

- **`McpServer::callLocal(name, arguments, Respond respond)`**:
  - Queues the call in the same pending queue as HTTP `tools/call`; it runs in the next `poll()`, in arrival order
    with the agents' calls.
  - A local call has no waiting thread, so `poll()` enforces its deadline: past `toolTimeoutSeconds` it claims the
    call and answers "Timeout" once (the same once-flag as a late deferred answer). An unknown tool is answered with an
    error.
  - `respond` runs on the polling thread. A deferred answer from another thread is stored, and `poll()` delivers it.
    The only exception is `stop()` (and the destructor): it answers pending local calls with an error on the calling
    thread, even when the server never listened (`mcpPort = 0`); in the app that is the main thread.
  - It does not block, unlike `handleMessage`.
- **The call observer**: `McpServer::setCallObserver(std::function<void(const CallRecord&)>)`, one observer (the
  app's).
  - `CallRecord` has: `id` (a counter assigned by the server; the JSON-RPC id never reaches the tool call), `tool`,
    `arguments`, `source` (`Remote` or `Local`), `phase` (`Started` or `Answered`), `isError`, the duration in ms and
    the first text part of the answer, cut at 200 characters.
  - `Started` is recorded when `poll()` starts the handler (not at queue time). `Answered` records come from several
    threads: a timeout is detected on the HTTP connection thread, and deferred answers can come from anywhere. So they
    go into a record queue, and `poll()` delivers the queue to the observer on the polling thread; the observer is
    never called from another thread (`stop()` drops undelivered records).
  - The queue and its mutex live in a `shared_ptr` state that the answer callbacks capture (as they capture the
    `shared_ptr<PendingCall>` today), never `this`: a stored callback called after the server is destroyed stays
    harmless.
  - `McpServer::toolNames()` and `toolInfo(name)` (description, input schema) for the console's `tools`, `help` and
    completion.
- **Log lines** (`engine/app/McpCallLog.{h,cpp}`: the observer that `EngineApp` installs; category `mcp`): one line per call when it is answered:
  `<tool> <arguments, compact JSON, cut at 200 chars> -> ok|error <ms> ms: <answer, cut at 200 chars>`, with
  `(console)` for local calls. A call that has not been answered after 0.5 s also gets a `<tool> ... running` line, so
  long calls are visible while they run.
- **Flooding**: agents poll the read-only tools `logs`, `app_status`, `world_status` and `get_profile` in loops. Calls
  of those tools are logged at most once per 2 s per tool: in between they are counted, and the next line says
  `(+N calls)`; a pending count is flushed as its own line once 2 s have passed (checked in each `poll`), so the
  last calls of a polling loop are not lost. The clock is injectable for the test.
- **No echo**: `mcp` lines are not echoed to stdout by default (headless agent runs would print every call).
  `setLogEcho` keeps its meaning for the other categories, and `SAIGA_MCP_ECHO=1` echoes them too.
- **The `logs` tool** leaves out the category `mcp` unless `category: "mcp"` is asked for. Otherwise an agent reading
  its logs reads its own calls. This is an added default, written into the tool's description.

### 4.2 The `set_ui_tab` app tool (`engine/app/AppTools.cpp`)

An agent can switch the tabs, e.g. to take a `screenshot gui=true` of the Performance tab or to show the user something:

| Tool | Contract |
|---|---|
| `set_ui_tab(tab?)` | Selects a tab by name (`game`, `performance`, `engine`, `console`, or a game tab's name, case-insensitive), the same as pressing its key. Without `tab` it only reads. Returns the selected tab, the list of tabs with their keys, and whether the UI is visible (G) and enabled. An unknown name is an error that lists the names; with `debugUi = false` it is an error. Without ImGui (`noRender`, `--benchmark`) the selection is still stored and answered (`visible: false`). Changes only the UI (not the world), and the change is persisted like a key press. It does not toggle G: a hidden UI stays hidden (the answer says so). |

`app_status` adds `ui_tab` (the selected tab, or null when the UI is off). The tool has a `handleMessage` test
(R-MCP-1): select by name, read only, unknown name, game tab, UI off.

### 5. Games and the world prototype

- `GameRenderHooks::gui` keeps its signature and runs every frame in every tab (the world game toggles its walk mode
  inside it, and game windows must not vanish while F2–F4 are open). The games keep their own small windows; they are
  moved below the top bar (`ImGuiCond_FirstUseEver` positions under the bar's height, `DebugUi::barHeight()`), and the
  key help names the new keys. On F2–F4 a game window on the panel's side is covered by the panel (accepted; F1 shows
  the game alone). A game may add tabs or sections in `renderHooks.init` through `app.debugUi()`.
- **Between-steps queue for games**: `EngineApp::queueBetweenSteps(std::function<void()>)` runs a function after the
  next `poll()` (the same queue as the UI's actions, §3). The games' buttons that change the world (platformer
  restart, racing and flight respawn) use it.
- The world prototype (`game_world`, racing) draws only widgets, no windows of its own:
  - `WorldRenderer::imgui()` becomes the Engine tab section "World" (under Rendering).
  - `imguiStatistics()` (the "World Statistics" window, `WorldStatistics.cpp`) becomes the Performance tab section
    "World passes".

  `WorldRenderer` stays in `saiga_vulkan`, with no engine headers (§5.2). Its "Engine statistics (F1)" hint
  (`WorldRenderer.cpp`) becomes "Performance (F2)". Flight and platformer also use `WorldRenderer`: their calls of
  `imgui()` (if any) move the same way.
- The key help in `world/main.cpp` and §17.1 name the new keys.

## Out of scope

- **S19d**: the entity inspector and editor in the Engine tab (`list_entities` with filters and paging, `inspect` as a
  tree, leaf edits as `set_component`, where vectors are sent whole), the camera tools ("copy camera" as a
  `set_camera` command), profile "freeze" and "freeze on spike", sortable profile tables, console argument completion
  from the input schema.
- The game UI (S46) and ImGui multi-viewport.
- Changing vsync, MSAA or the resolution at runtime (needs a swap chain rebuild).
- Debug views (wireframe, LOD colors, normals, overdraw): they need shader variants in `RenderScene`.
- Console scripting (loops, variables), console commands that are not tools, remote consoles.
- Benchmark runs from the UI: `--benchmark` stays, because it needs a fresh app at a fixed size.
- The host world with `--coop` (the UI shows the client world, the tools' world).

## Design notes

- **Commands are the MCP tools.** Agents and the console share one command surface. A new tool is a new console
  command for free, `help` shows the same descriptions the agent reads, and the call log shows both sources in one
  stream. The cost is a JSON-like syntax, which the `key=value` form keeps short (`spawn class=car position=[0,1,0]`).
- **Everything that changes the world runs between steps**: tool calls in `poll()` (local ones too) and the UI's
  actions right after it. Nothing in the engine UI changes the world in the gui callback, which runs during the frame;
  games get the same queue (`queueBetweenSteps`, §5). The ordering
  against agent calls is their arrival order.
- **The observer only fires on the polling thread.** The bar, the console and the log need no locks of their own, and
  `McpServer` keeps its threading in one place.
- **Tabs in one docked panel, not windows.** One panel at one edge, shared by all tabs, uses the screen best: the game
  view always has the same free area, and switching tabs never moves anything. The panel is positioned every frame and
  never saved by ImGui, so it cannot get lost off-screen after a resize (an annoyance of today's loose windows).
  `DebugUiStyle` gives every tab the same sections, property tables, number formats and colors.
- **Tabs, not toggles.** An F-key always selects, so the key you press always shows that tab, whatever was open before.
  The Game tab is the "no debug panel" state, so there is no separate collapse button; G still hides all ImGui (also
  the bar and the game's GUI) for clean captures.
- **The F-keys bypass ImGui's keyboard capture** only on the press edge and only for F1–F10. They never produce text,
  so text input is not affected.

## Tests

- `test_engine_debug_ui.cpp` (new, CI, no device, no ImGui context):
  - `DebugConsole.ParsesCommands`: empty input, `tool`, `tool {json}`, `key=value` with numbers, bools, arrays, quoted
    and unquoted strings, an invalid JSON object (an error message), the built-ins.
  - `DebugConsole.CompletesToolNames`: unique prefix, ambiguous prefix (the candidates), no match.
  - `DebugConsole.RunsToolsThroughThePoll`: a command queues a call that runs only in `poll()`; the answer is in the
    view buffer and not in the ring; an error is marked; a deferred tool answers in a later poll.
  - `DebugUi.KeysSelectTabs`: F-key edges select tabs (the selected tab's key changes nothing; F1 has no panel); keys work while captured; grave works only when
    not captured; F1–F10 are cleared from the state; `debugUi = false` passes them through.
  - `DebugUi.KeysFilterAndEdges`: a held F-key is one press; the returned state has F1–F10 cleared.
  - `DebugUi.StatePersists`: tab, edge, size, sections, filters and history are written and read back (temporary
    directory); with persistence off nothing is read or written.
  - `DebugUi.ActionsRunAfterThePoll`: a queued step or pause from the UI is applied by `EngineApp::frame` after
    `poll()`, not during the gui callback (`EngineAppNoRender`, the step index before and after).
  - `McpCallLog.LinesAndThrottling` (the class alone, a test server): an answered call gives one `mcp` line with tool, arguments, result and duration; a
    `(console)` mark for local calls; polling tools are throttled to one line per 2 s with `(+N calls)` (injected
    clock); a pending count is flushed after 2 s; `mcp` lines are not echoed.
- `test_engine_app.cpp` (extended, no device): `EngineAppNoRender.SetUiTab` (§4.2; the UI exists without a device, so
  the selection works and is read back; with `debugUi = false` an error).
- `test_core_mcp_server.cpp` (extended):
  - `McpServer.CallLocal`: runs in `poll()`; unknown tool; a timeout enforced by `poll()`; a deferred answer from
    another thread is delivered by `poll()`; `stop()` answers pending local calls without `start()`.
  - `McpServer.CallObserver`: started/answered records for remote and local calls, with an error and a duration ≥ 0. A
    remote call that times out without `poll()` (the HTTP thread gives up) delivers its record at the next `poll()`,
    on the polling thread (thread id checked). Ids are unique.
- `test_engine_core.cpp` (extended): lines added directly to a `LogRing` carry a non-decreasing `time` > 0.
- `test_engine_world_api.cpp` (extended): `logs` leaves out `mcp` lines by default and returns them with
  `category: "mcp"`; the `time` key.
- `test_engine_app.cpp` (extended, Vulkan, skips without a driver): `EngineAppRender.DebugUiTabsRender`. A headless app
  renders 3 frames with each tab selected (no validation error). It clicks a tab button through ImGui's input queue, as
  `test_vulkan_standalone_window` does. An MCP call made with `handleMessage` on a `std::thread`, while the main thread
  runs `app.frame()` until it is answered (not `Poller`: the handler and the observer must run on the frame thread),
  shows up as an `mcp` line.

## Done when

- The four tabs and the top bar work in racing, flight, platformer and world. F2–F4 share the one docked panel, F1 shows the
  game without it, and no engine window floats.
- MCP calls from an agent (through the `saiga-world` server) and from the console appear in the console, and
  `set_ui_tab` switches the tabs. Polling tools
  are throttled.
- `logs` leaves out `mcp` lines by default.
- `docs/engine_design.md` §17.1 and §16.2 describe the debug UI, the console and the call log, and its stage table has
  S19c (and S19d as planned). AGENTS.md's engine line mentions `DebugUi`.
- All tests pass in the CI configuration, and the new Vulkan test also passes on lavapipe.

## Verification

```shell
cmake --build build --parallel -- -k 0
(cd build && ctest --output-on-failure --parallel 4)
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ./build/tests/test_engine_app --gtest_filter='EngineAppRender.*'
./build/tests/test_engine_debug_ui && ./build/tests/test_core_mcp_server
```

Locally, with a GPU: take a screenshot of each tab and review them.

```shell
for t in 1 2 3 4; do SAIGA_DEBUG_TAB=$t SAIGA_SCREENSHOT=/tmp/racing_tab$t.png ./build/samples/games/game_racing --demo; done
```

Then run `game_world` with an agent connected to `saiga-world`, and check that its calls show up in the console while
the game runs.

## Files

- **New**:
  - `src/saiga/engine/app/DebugUi.{h,cpp}`, `DebugUiStyle.{h,cpp}`, `DebugConsole.{h,cpp}`, `McpCallLog.{h,cpp}`,
    `StatisticsPanel.{h,cpp}`
  - `tests/test_engine_debug_ui.cpp`, registered in `tests/CMakeLists.txt` under `MODULE_ENGINE`
- **Changed**:
  - `src/saiga/engine/app/EngineApp.{h,cpp}`: owns `DebugUi`, the call observer, the key handling and the queued UI
    actions. The statistics window and overlay become the Performance tab and the top bar.
  - `src/saiga/engine/app/EngineAppSetup.h`: `debugUi`, `debugUiPersistence`.
  - `src/saiga/engine/app/EngineMain.cpp`: `--stats`, `SAIGA_DEBUG_TAB`.
  - `src/saiga/core/util/mcp/McpServer.{h,cpp}`: `callLocal`, the observer, the record queue, `toolNames`/`toolInfo`.
  - `src/saiga/engine/core/{LogRing,Log}.{h,cpp}`: the time, the `mcp` echo.
  - `src/saiga/engine/api/WorldApi.cpp`, `src/saiga/engine/tools/WorldTools.cpp`: `logs`.
  - `src/saiga/engine/app/AppTools.{h,cpp}`: `set_ui_tab`, `app_status.ui_tab`.
  - `src/saiga/vulkan/world/WorldRenderer.{h,cpp}`, `WorldStatistics.cpp`: no windows.
  - `samples/games/{racing,flight,platformer,world}/*`: the GUI without windows; `showStatistics` removed.
  - The tests above.
  - `docs/engine_design.md`: §17.1, §16.2, the stage table (S19c, S19d).
  - `AGENTS.md`.

## Design changes

Decided with the user during the implementation, or found in the three UI rating rounds (below):

- **The panel**: one panel at the right edge for F2-F4, not movable (no edge choice); its width can be dragged at its
  left edge (default 28 x font size). F1 Game shows no panel; the game's own GUI is drawn in every tab.
- **The tabs**: F1 Game, F2 Performance, F3 Engine (simulation, rendering, camera and input; the world prototype's
  settings), F4 Console. The panel has no title (the bar shows the tab).
- **The Performance tab**: what matters at a glance first, without expanding anything: a short tag of what limits the
  frame rate (VSYNC-BOUND, GPU-BOUND, CPU-BOUND, WAIT-BOUND, STUTTER; the sentence in its tooltip), fps / 1% low / max /
  missed, the CPU and GPU busy bars, the frame graph taking the free height (scales and legend in the gutter). The window
  selector, reset and copy JSON in one line under the graph. Collapsed at first: frame timings, the world step (the
  systems sorted by their mean time; moved here from the Engine tab), the engine's GPU sections (only when no game
  section breaks the GPU down) and the game's sections (`addSection` is collapsed by default).
- **The bar**: fps, CPU, GPU, step (with units, fixed widths so they do not move), warnings (stutter above
  `kStutterWarnPerSecond`, dropped steps, a long MCP call) and a PAUSED button. Its numbers refresh with the statistics.
- **The Pause key** pauses and resumes the world (`DebugUi::setPauseKey`).
- **The state file** is `debug_ui.json` (JSON instead of Ini: the command history holds arbitrary text).
- **The code**: the statistics and tab drawing moved from `EngineApp.cpp` into `EngineAppDebugUi.cpp` (members of
  `EngineApp`), not into a `StatisticsPanel` class (they need the app's statistics state).
- **The MCP lines**: `tool -> ok|error <ms> ms: <answer>  args <arguments>` (result first); the tool name in the MCP
  color, failed calls red; console lines cut with "..." (the whole line in a tooltip), the time right-aligned.
- **The world prototype**: its sub-sections are tree nodes without indent (`subSection`), not nested headers; the
  overview sentence and the duplicate MSAA are gone.
- **No step field** in the log lines (only the time), as planned after the review.

## Results

- New tests: `test_engine_debug_ui` (9: parser, completion, tools through the poll, built-ins and filters, an answer
  after the console is gone, tab keys, state, the call log's lines/throttling/running lines, no echo),
  `McpServer.CallLocal`, `McpServer.CallObserver`, `Log.LinesCarryTheTime`, `WorldApi.LogsLeaveOutTheMcpCallLog`,
  `EngineAppNoRender.SetUiTab`, `SetUiTabWithTheUiOff`, `ActionsRunBetweenSteps`, `McpCallsInTheLog`,
  `EngineAppRender.DebugUiTabsRender` (each tab rendered headless, a click on a tab button; also on lavapipe).
- `ctest` passes locally except `test_opengl_window` (`ReturnsNullWithoutADisplay` fails on a machine with a display; it
  fails the same way without this change). The engine and Vulkan tests also pass on lavapipe.
- UI ratings (10 metrics: glanceability, progressive disclosure, hierarchy and density, consistency, readability,
  navigation, game view intrusion, task efficiency, console, layout robustness; 1-10 each), each round by a new reviewer
  from screenshots of every tab taken through `set_ui_tab` + `screenshot` and the code: round 1 6.3, round 2 7.2, round
  3 6.8 (a stricter reviewer), after the round 3 changes 8.0. Screenshots were checked on an RTX 5080 (racing --demo).

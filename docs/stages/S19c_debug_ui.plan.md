# S19c: Debug UI

Status: planned.

Sections follow the stage plan template of S05.

## Goal

Every engine game gets one developer UI instead of loose ImGui windows:

- **The top bar**: a thin bar at the top of the screen, with tabs on the left (`F1 Game`, `F2 Performance`, `F3 Engine`,
  `F4 Console`) and a compact performance summary on the right (fps, frame, CPU, GPU, step).
- **Opening a tab**: an F-key opens its tab, and the same key closes it again.
- **Engine tabs are docked**: F2–F4 are fixed bars at a screen edge (bottom, side), never floating windows, and they
  share one visual language. The Game tab belongs to the game.
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
place of `statisticsWindow()`/`statisticsOverlay()`. `DebugUi` is destroyed after the MCP server (`EngineApp`'s
destructor resets `server` first), because the server's pending local calls refer to the console.

- **Top bar**: a window across the top, one line high (font size + 2 × 4 px padding), background alpha 0.6. On the
  left are the tab buttons with their key (`F1 Game` …), and the open tab is highlighted. The performance summary of §2
  is on the right. Clicking a tab button does the same as pressing its key.
- **Panels**: one tab is open at a time. Its panel is docked to a screen edge below the top bar (§1.1).
- **Keys** (`DebugUi::handleKeys(const InputState& now, const InputState& previous, bool captured)`, a pure function,
  testable without a window):
  - F1–F4 toggle their tab on the press edge, and F5–F10 toggle the game's own tabs. They also work while ImGui has the
    keyboard (they never type text), so a focused console input does not trap the user.
  - Grave (`` ` ``) opens the console and focuses its input, only while ImGui does not capture the keyboard. The opening
    press is not typed into the field: the focus request takes effect in the next frame.
  - `EngineApp::applyPlayerInput` calls `handleKeys` and clears F1–F10 from the state before `updateInputMap` and
    `lastDeviceState`, so the input map never sees them while the debug UI is on.
  - ESC first leaves a focused input (ImGui's own handling). It quits only when nothing is captured, as today.
  - G still hides all ImGui, including the bar. While it is hidden the F-keys still switch the tab, and the UI shows
    again only with G.
  - F12 stays the screenshot.
  - `EngineAppSetup::debugUi = false` turns the UI off (keys go to the input map, PR #66's window and overlay are not
    drawn either).
- **State**: `debug_ui.ini` in the working directory, like ImGui's `imgui.ini`. It is not in the data directory,
  because the data directory is the source tree's `data/<name>`. It stores:
  - the open tab and each tab's panel size,
  - the collapsed sections,
  - the console filters and the command history (the last 100 commands).

  It is read at start and written at quit, and at most once per second when something changed. Persistence is off
  (nothing read or written) when headless, with `SAIGA_SCREENSHOT`, with `--benchmark` and when
  `EngineAppSetup::debugUiPersistence` is false (tests). This keeps screenshots and tests independent of the last
  session.
- **Start tab**: `--stats` opens the Performance tab. `SAIGA_DEBUG_TAB=<n>` opens tab n (screenshots and checks). Both
  override the ini. `EngineApp::showStatistics(bool)` is removed, and so are its calls in the games (platformer, flight,
  racing, world): the constructor already applies `--stats`.
- **Registration**:
  - `DebugUi::addTab(name, draw, edge)` adds a game tab on the next free key (F5…F10).
  - `DebugUi::addSection(tab, title, draw)` appends a collapsible section to an engine tab. Sections are drawn in
    registration order after the engine's own content.
  - The engine tabs are `DebugTab::Game`, `Performance`, `Engine` and `Console`.
- **Docking**: every debug UI window sets `ImGuiWindowFlags_NoDocking` (docking is enabled in `imgui_saiga.cpp`), so
  game windows cannot dock into the bars.
- **Text**: ASCII only (`->`, `<-`, `v`), because ImGui's default font has only Basic Latin and Latin-1, and MSVC
  builds without `/utf-8`.

### 1.1 Layout and look of the engine tabs (`engine/app/DebugUiStyle.{h,cpp}`)

The engine tabs are fixed bars at the screen edges, not floating windows. Each panel is an ImGui window with the flags
`NoTitleBar`, `NoMove`, `NoResize` (the splitter is our own), `NoCollapse`, `NoSavedSettings`, `NoBringToFrontOnFocus`
and `NoDocking`. It is positioned and sized every frame (`ImGuiCond_Always`) from the display size, so it cannot get
lost and follows window resizes. Each tab has the edge that suits its content:

| Tab | Edge | Default size | Why |
|---|---|---|---|
| F1 Game (default container) | left side bar | 26 × font size | short game state and settings; the view stays free |
| F2 Performance | bottom bar | 38% of the height | graphs need width: three columns (summary · frame graph · tables) |
| F3 Engine | right side bar | 30 × font size | a property list: simulation, render settings, camera, input |
| F4 Console | bottom bar | 40% of the height | log lines are wide, like an IDE terminal |

- **Sizing**: the inner edge of a panel is a splitter: a 4 px drag handle, highlighted on hover, with the resize cursor.
  The size is clamped to 15–70% of the screen dimension and stored per tab. Double-clicking the splitter resets the
  default. No panel covers the top bar.
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
| **F1 Game** | The game's `GameRenderHooks::gui`, drawn inside the left side bar (§5). |
| **F2 Performance** | Bottom bar in three columns.<br>• Left: PR #66's verdict, busy shares, fps / 1% low / max / missed, and the 3/10/30 s window selector.<br>• Middle: the frame graph at the column's full width and height.<br>• Right: PR #66's frame table, the step profile scopes (avg/p95/share) and the GPU sections.<br>Toolbar: "copy JSON" (the `get_profile` answer to the clipboard). The world prototype adds "World passes" as a section (§5). |
| **F3 Engine** | Right side bar, four sections:<br>• **Simulation**: pause/resume, step 1, step 10, time scale (0.1–4), step index, dropped steps, entities, game objects, with `--coop` "client world".<br>• **Systems**: per phase the systems with avg/p95 ms from the step profile.<br>• **Rendering**: tone map and exposure (moved from the statistics window); vsync and MSAA shown read-only; the game's render settings as sections (the world prototype's "World", §5).<br>• **Camera and input**: view camera position, yaw/pitch, fovy, held by a tool; the input map's actions and held buttons; the local player. |
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
  `logPrintf`. The `logs` tool returns it as an added key. There is no step field, because worker threads and `--coop`
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
  - `respond` always runs on the polling thread. A deferred answer from another thread is stored, and `poll()`
    delivers it.
  - `stop()` and the destructor answer pending local calls with an error even when the server never listened
    (`mcpPort = 0`).
  - It does not block, unlike `handleMessage`.
- **The call observer**: `McpServer::setCallObserver(std::function<void(const CallRecord&)>)`, one observer (the
  app's).
  - `CallRecord` has: `id` (a counter assigned by the server; the JSON-RPC id never reaches the tool call), `tool`,
    `arguments`, `source` (`Remote` or `Local`), `phase` (`Started` or `Answered`), `isError`, the duration in ms and
    the first text part of the answer, cut at 200 characters.
  - Records come from several threads: a timeout is detected on the HTTP connection thread, `stop()` runs on any
    thread, and deferred answers can come from anywhere. So the server queues them under `pendingMutex`, and `poll()`
    delivers them to the observer on the polling thread. The observer is never called from another thread.
- **Log lines** (`EngineApp`'s observer, category `mcp`): one line per call when it is answered:
  `<tool> <arguments, compact JSON, cut at 200 chars> -> ok|error <ms> ms: <answer, cut at 200 chars>`, with
  `(console)` for local calls. A call that has not been answered after 0.5 s also gets a `<tool> ... running` line, so
  long calls are visible while they run.
- **Flooding**: agents poll the read-only tools `logs`, `app_status`, `world_status` and `get_profile` in loops. Calls
  of those tools are logged at most once per 2 s per tool: in between they are counted, and the next line says
  `(+N calls)`. The clock is injectable for the test.
- **No echo**: `mcp` lines are not echoed to stdout by default (headless agent runs would print every call).
  `setLogEcho` keeps its meaning for the other categories, and `SAIGA_MCP_ECHO=1` echoes them too.
- **The `logs` tool** leaves out the category `mcp` unless `category: "mcp"` is asked for. Otherwise an agent reading
  its logs reads its own calls. This is an added default, written into the tool's description.

### 5. Games and the world prototype

- `GameRenderHooks::gui` keeps its signature and is drawn inside the Game tab's side bar. The games (racing, flight,
  platformer, world) remove their `ImGui::Begin/End`, and their key help moves to the top of the tab. A game may still
  draw its own windows (the Game tab is the game's), and it may add tabs or sections in `renderHooks.init` through
  `app.debugUi()`.
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
  actions right after it. Nothing changes the world in the gui callback, which runs during the frame. The ordering
  against agent calls is their arrival order.
- **The observer only fires on the polling thread.** The bar, the console and the log need no locks of their own, and
  `McpServer` keeps its threading in one place.
- **Engine tabs are docked bars, not windows.** They are positioned every frame and never saved by ImGui, so they cannot
  get lost off-screen after a resize (an annoyance of today's loose windows). Each edge fits its content: wide graphs
  and logs at the bottom, property lists at the side. `DebugUiStyle` gives every tab the same sections, property tables,
  number formats and colors.
- **The F-keys bypass ImGui's keyboard capture** only on the press edge and only for F1–F10. They never produce text,
  so text input is not affected.

## Tests

- `test_engine_debug_ui.cpp` (new, CI, no device, no ImGui context):
  - `DebugConsole.ParsesCommands`: empty input, `tool`, `tool {json}`, `key=value` with numbers, bools, arrays, quoted
    and unquoted strings, an invalid JSON object (an error message), the built-ins.
  - `DebugConsole.CompletesToolNames`: unique prefix, ambiguous prefix (the candidates), no match.
  - `DebugConsole.RunsToolsThroughThePoll`: a command queues a call that runs only in `poll()`; the answer is in the
    view buffer and not in the ring; an error is marked; a deferred tool answers in a later poll.
  - `DebugUi.KeysToggleTabs`: F-key edges open, close and switch tabs; keys work while captured; grave works only when
    not captured; F1–F10 are cleared from the state; `debugUi = false` passes them through.
  - `DebugUi.StatePersists`: tab, sizes, sections, filters and history are written and read back (temporary
    directory); with persistence off nothing is read or written.
  - `DebugUi.ActionsRunAfterThePoll`: a queued step or pause from the UI is applied by `EngineApp::frame` after
    `poll()`, not during the gui callback (`EngineAppNoRender`, the step index before and after).
  - `McpCallLog.LinesAndThrottling`: an answered call gives one `mcp` line with tool, arguments, result and duration; a
    `(console)` mark for local calls; polling tools are throttled to one line per 2 s with `(+N calls)` (injected
    clock); `mcp` lines are not echoed.
- `test_core_mcp_server.cpp` (extended):
  - `McpServer.CallLocal`: runs in `poll()`; unknown tool; a timeout enforced by `poll()`; a deferred answer from
    another thread is delivered by `poll()`; `stop()` answers pending local calls without `start()`.
  - `McpServer.CallObserver`: started/answered records for remote and local calls, with an error and a duration ≥ 0. A
    remote call that times out without `poll()` (the HTTP thread gives up) delivers its record at the next `poll()`,
    on the polling thread (thread id checked). Ids are unique.
- `test_engine_core.cpp` (extended): `LogRing` lines carry a non-decreasing `time`.
- `test_engine_world_api.cpp` (extended): `logs` leaves out `mcp` lines by default and returns them with
  `category: "mcp"`; the `time` key.
- `test_engine_app.cpp` (extended, Vulkan, skips without a driver): `EngineAppRender.DebugUiTabsRender`. A headless app
  renders 3 frames with each tab open (no validation error). It clicks a tab button through ImGui's input queue, as
  `test_vulkan_standalone_window` does. An MCP call made with `handleMessage` shows up as an `mcp` line.

## Done when

- The four tabs and the top bar work in racing, flight, platformer and world. The engine tabs are docked bars, and no
  engine window floats.
- MCP calls from an agent (through the `saiga-world` server) and from the console appear in the console. Polling tools
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
  - `src/saiga/engine/app/DebugUi.{h,cpp}`, `DebugUiStyle.{h,cpp}`, `DebugConsole.{h,cpp}`
  - `tests/test_engine_debug_ui.cpp`, registered in `tests/CMakeLists.txt` under `MODULE_ENGINE`
- **Changed**:
  - `src/saiga/engine/app/EngineApp.{h,cpp}`: owns `DebugUi`, the call observer, the key handling and the queued UI
    actions. The statistics window and overlay become the Performance tab and the top bar.
  - `src/saiga/engine/app/EngineAppSetup.h`: `debugUi`, `debugUiPersistence`.
  - `src/saiga/engine/app/EngineMain.cpp`: `--stats`, `SAIGA_DEBUG_TAB`.
  - `src/saiga/core/util/mcp/McpServer.{h,cpp}`: `callLocal`, the observer, the record queue.
  - `src/saiga/engine/core/{LogRing,Log}.{h,cpp}`: the time, the `mcp` echo.
  - `src/saiga/engine/api/WorldApi.cpp`, `src/saiga/engine/tools/WorldTools.cpp`: `logs`.
  - `src/saiga/vulkan/world/WorldRenderer.{h,cpp}`, `WorldStatistics.cpp`: no windows.
  - `samples/games/{racing,flight,platformer,world}/*`: the GUI without windows; `showStatistics` removed.
  - The tests above.
  - `docs/engine_design.md`: §17.1, §16.2, the stage table (S19c, S19d).
  - `AGENTS.md`.

## Design changes

(Filled in when the stage is done.)

## Results

(Filled in when the stage is done.)

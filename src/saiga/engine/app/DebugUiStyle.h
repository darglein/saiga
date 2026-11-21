/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/export.h"

#include <string>

#include <initializer_list>

namespace Saiga
{
namespace Engine
{
/**
 * The look of the debug UI (docs/engine_design.md §17.1; S19c), used by the engine tabs and offered to games: one color
 * per quantity (the same in every tab, the graphs and the top bar), section headers, two-column property tables,
 * right-aligned numbers. push()/pop() set the debug UI's ImGui style around its windows (games keep their own).
 */
namespace DebugUiStyle
{
// Quantities
constexpr ImU32 kCpu   = IM_COL32(90, 200, 255, 255);
constexpr ImU32 kGpu   = IM_COL32(255, 170, 60, 255);
constexpr ImU32 kFrame = IM_COL32(120, 230, 120, 255);
constexpr ImU32 kRef   = IM_COL32(210, 210, 210, 170);
constexpr ImU32 kWarn  = IM_COL32(255, 110, 90, 255);
constexpr ImU32 kWhite = IM_COL32(255, 255, 255, 255);
// Console
constexpr ImU32 kMcp     = IM_COL32(190, 150, 255, 255);
constexpr ImU32 kApi     = IM_COL32(200, 200, 140, 255);
constexpr ImU32 kConsole = IM_COL32(255, 255, 255, 255);
constexpr ImU32 kOutput  = IM_COL32(200, 200, 200, 255);
constexpr ImU32 kError   = IM_COL32(255, 95, 85, 255);
constexpr ImU32 kWarning = IM_COL32(240, 210, 80, 255);

// The color of a log line: by category (mcp, api), red/orange for a message starting with error/warning
SAIGA_ENGINE_API ImU32 categoryColor(const std::string& category, const std::string& message);

// The debug UI's style (padding, spacing, rounding, borders); every push() needs a pop()
SAIGA_ENGINE_API void push();
SAIGA_ENGINE_API void pop();

// A full-width section header; open is the state (kept by the caller, e.g. in debug_ui.ini); returns open
SAIGA_ENGINE_API bool section(const char* title, bool& open);

// Two-column property tables: a dimmed label left, the value or widget right at full width
SAIGA_ENGINE_API bool beginProperties(const char* id);
// Starts a row; the next widget or text is the value (widgets: use an ID-only label, e.g. "##exposure")
SAIGA_ENGINE_API void property(const char* label);
SAIGA_ENGINE_API void endProperties();

SAIGA_ENGINE_API void coloredText(ImU32 color, const char* text);
// Right-aligned text in the current table cell (moves to the next column first)
SAIGA_ENGINE_API void cellRight(const char* text);
// A table's header row: the first column left-aligned, the others (numbers) right-aligned
SAIGA_ENGINE_API void headerRow(std::initializer_list<const char*> names);
// A dimmed hint in the middle of the remaining region
SAIGA_ENGINE_API void emptyState(const char* text);
}  // namespace DebugUiStyle

}  // namespace Engine
}  // namespace Saiga

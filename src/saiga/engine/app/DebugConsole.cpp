/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/app/DebugConsole.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_stdlib.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/DebugUiStyle.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

namespace Saiga
{
namespace Engine
{
namespace
{
const char* const kBuiltins[] = {"clear", "help", "tools"};

std::string trim(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

bool isBuiltin(const std::string& name)
{
    return std::find(std::begin(kBuiltins), std::end(kBuiltins), name) != std::end(kBuiltins);
}

// The same clock as the log lines' time
double secondsSinceStart()
{
    return LogRing::clockSeconds();
}

// key=value tokens separated by whitespace outside quotes and brackets
bool splitArguments(const std::string& s, std::vector<std::string>& tokens, std::string& error)
{
    std::string cur;
    int depth     = 0;
    bool inString = false, escape = false;
    for (char c : s)
    {
        if (inString)
        {
            cur += c;
            if (escape)
                escape = false;
            else if (c == '\\')
                escape = true;
            else if (c == '"')
                inString = false;
            continue;
        }
        if (c == '"')
            inString = true;
        else if (c == '[' || c == '{')
            depth++;
        else if (c == ']' || c == '}')
            depth--;
        if (depth == 0 && std::isspace((unsigned char)c))
        {
            if (!cur.empty()) tokens.push_back(cur);
            cur.clear();
            continue;
        }
        cur += c;
    }
    if (inString || depth != 0)
    {
        error = inString ? "unterminated string" : "unbalanced brackets";
        return false;
    }
    if (!cur.empty()) tokens.push_back(cur);
    return true;
}
}  // namespace

bool DebugConsole::parse(const std::string& line, Command& out, std::string& error)
{
    out                   = Command();
    const std::string all = trim(line);
    if (all.empty())
    {
        error = "empty command";
        return false;
    }
    size_t space = 0;
    while (space < all.size() && !std::isspace((unsigned char)all[space])) space++;
    out.name               = all.substr(0, space);
    const std::string rest = trim(all.substr(space));
    if (isBuiltin(out.name))
    {
        out.builtin         = true;
        out.builtinArgument = rest;
        return true;
    }
    if (rest.empty()) return true;
    if (rest[0] == '{')
    {
        std::string err;
        json11::Json j = json11::Json::parse(rest, err);
        if (!err.empty() || !j.is_object())
        {
            error = "the arguments are not a JSON object" + (err.empty() ? std::string() : ": " + err);
            return false;
        }
        out.arguments = j;
        return true;
    }
    std::vector<std::string> tokens;
    if (!splitArguments(rest, tokens, error)) return false;
    json11::Json::object args;
    for (const std::string& t : tokens)
    {
        const size_t eq = t.find('=');
        if (eq == std::string::npos || eq == 0)
        {
            error = "expected key=value, got \"" + t + "\"";
            return false;
        }
        const std::string key = t.substr(0, eq), value = t.substr(eq + 1);
        std::string err;
        json11::Json j = json11::Json::parse(value, err);
        args[key]      = err.empty() && !value.empty() ? j : json11::Json(value);
    }
    out.arguments = args;
    return true;
}

DebugConsole::DebugConsole(McpServer* server, LogRing& ring)
    : server(server), ring(ring), self(std::make_shared<DebugConsole*>(this))
{
    update();
}

DebugConsole::~DebugConsole()
{
    // Answers that still arrive find no console
    *self = nullptr;
}

void DebugConsole::add(Entry e)
{
    if (e.kind == Entry::Kind::Log) seenCategories.insert(e.category);
    view.push_back(std::move(e));
    addedEntries++;
    if (view.size() > kMaxEntries)
    {
        view.pop_front();
        if (seenEntries > 0) seenEntries--;
    }
}

void DebugConsole::output(Entry::Kind kind, const std::string& text)
{
    Entry e;
    e.kind     = kind;
    e.time     = secondsSinceStart();
    e.category = "console";
    e.text     = text;
    add(std::move(e));
}

void DebugConsole::update()
{
    for (LogRing::Line& l : ring.since(lastSequence))
    {
        lastSequence = l.sequence;
        Entry e;
        e.kind     = Entry::Kind::Log;
        e.time     = l.time;
        e.category = std::move(l.category);
        e.text     = std::move(l.message);
        add(std::move(e));
    }
}

void DebugConsole::clear()
{
    view.clear();
    seenEntries = 0;
}

std::vector<std::string> DebugConsole::complete(const std::string& prefix) const
{
    std::vector<std::string> names;
    if (server) names = server->toolNames();
    for (const char* b : kBuiltins) names.push_back(b);
    std::vector<std::string> out;
    for (const std::string& n : names)
        if (n.compare(0, prefix.size(), prefix) == 0) out.push_back(n);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

bool DebugConsole::visible(const Entry& e) const
{
    if (e.kind == Entry::Kind::Log)
    {
        if (mcpOnly && e.category != "mcp") return false;
        if (hiddenCategories.count(e.category)) return false;
    }
    else if (mcpOnly)
        return false;
    if (search.empty()) return true;
    auto lower = [](std::string s)
    {
        for (char& c : s) c = char(std::tolower((unsigned char)c));
        return s;
    };
    // The lowercase forms are kept (the filter runs over every entry each frame)
    if (searchLower.first != search) searchLower = {search, lower(search)};
    if (e.lower.empty()) e.lower = lower("[" + e.category + "] " + e.text);
    return e.lower.find(searchLower.second) != std::string::npos;
}

void DebugConsole::builtin(const Command& c)
{
    if (c.name == "clear")
    {
        clear();
        return;
    }
    if (c.name == "tools")
    {
        std::string text;
        if (server)
            for (const std::string& n : server->toolNames()) text += (text.empty() ? "" : "  ") + n;
        output(Entry::Kind::Output, text.empty() ? "no tools" : text);
        return;
    }
    // help [tool]
    if (c.builtinArgument.empty())
    {
        output(Entry::Kind::Output,
               "Commands are the MCP tools: tool, tool {json}, tool key=value ... (values parse as JSON, else a "
               "string).\nBuilt-ins: help [tool], tools, clear. Tab completes tool names, Up/Down the history.");
        return;
    }
    std::string description;
    json11::Json schema;
    if (!server || !server->toolInfo(c.builtinArgument, description, schema))
    {
        output(Entry::Kind::Error, "unknown tool: " + c.builtinArgument);
        return;
    }
    std::ostringstream text;
    text << c.builtinArgument << ": " << description;
    std::set<std::string> required;
    for (const auto& r : schema["required"].array_items()) required.insert(r.string_value());
    for (const auto& [name, p] : schema["properties"].object_items())
    {
        std::string type = p["type"].is_string() ? p["type"].string_value() : p["type"].dump();
        text << "\n  " << name << (required.count(name) ? "" : "?") << " (" << type
             << "): " << p["description"].string_value();
    }
    output(Entry::Kind::Output, text.str());
}

void DebugConsole::submit(const std::string& line)
{
    const std::string cmd = trim(line);
    if (cmd.empty()) return;
    if (history.empty() || history.back() != cmd) history.push_back(cmd);
    if (history.size() > kHistorySize) history.erase(history.begin());
    Entry input;
    input.kind     = Entry::Kind::Input;
    input.time     = secondsSinceStart();
    input.category = "console";
    input.text     = cmd;
    Command c;
    std::string error;
    if (!parse(cmd, c, error))
    {
        add(std::move(input));
        output(Entry::Kind::Error, error);
        return;
    }
    if (c.builtin)
    {
        add(std::move(input));
        builtin(c);
        return;
    }
    if (!server)
    {
        add(std::move(input));
        output(Entry::Kind::Error, "no MCP server");
        return;
    }
    input.pending = true;
    add(std::move(input));
    pending++;
    // The entry is found again by its position from the end (later entries may be added before the answer)
    const uint64_t inputIndex          = addedEntries - 1;
    std::shared_ptr<DebugConsole*> ref = self;
    server->callLocal(c.name, c.arguments,
                      [ref, inputIndex](McpServer::ToolResult result)
                      {
                          DebugConsole* con = *ref;
                          if (!con) return;
                          con->pending--;
                          // Mark the input line answered (if it is still in the view)
                          const uint64_t back = con->addedEntries - 1 - inputIndex;
                          if (back < con->view.size()) con->view[con->view.size() - 1 - size_t(back)].pending = false;
                          std::string text;
                          for (const auto& part : result.content)
                          {
                              if (!text.empty()) text += "\n";
                              if (part["type"].string_value() == "image")
                                  text += "[image, " + part["mimeType"].string_value() + ", " +
                                          std::to_string(part["data"].string_value().size() * 3 / 4 / 1024) + " KB]";
                              else
                                  text += part["text"].string_value();
                          }
                          con->output(result.isError ? Entry::Kind::Error : Entry::Kind::Output,
                                      text.empty() ? "(no content)" : text);
                      });
}

// ----------------------------------------------------------------------------------------------------- ImGui

namespace
{
struct InputCallbackData
{
    DebugConsole* console;
    int* historyPos;
};
}  // namespace

void DebugConsole::draw(bool focusInput)
{
    update();
    // The toolbar: search, MCP only, categories, clear
    const float em = ImGui::GetFontSize();
    ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - 19 * em, 6 * em));
    ImGui::InputTextWithHint("##search", "search", &search);
    ImGui::SameLine();
    ImGui::Checkbox("MCP only", &mcpOnly);
    ImGui::SameLine();
    if (ImGui::Button("categories")) ImGui::OpenPopup("##categories");
    if (ImGui::BeginPopup("##categories"))
    {
        for (const std::string& c : seenCategories)
        {
            bool shown = hiddenCategories.count(c) == 0;
            if (ImGui::Checkbox(c.c_str(), &shown))
            {
                if (shown)
                    hiddenCategories.erase(c);
                else
                    hiddenCategories.insert(c);
            }
        }
        if (seenCategories.empty()) ImGui::TextDisabled("no lines yet");
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("clear")) clear();

    // The log view
    const float inputHeight = ImGui::GetFrameHeightWithSpacing();
    if (ImGui::BeginChild("##log", ImVec2(0, -inputHeight), ImGuiChildFlags_Borders))
    {
        std::vector<int> shown;
        shown.reserve(view.size());
        for (int i = 0; i < int(view.size()); ++i)
            if (visible(view[size_t(i)])) shown.push_back(i);
        const bool atBottom   = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1;
        const float timeWidth = ImGui::CalcTextSize("0000.00").x;
        if (shown.empty())
            DebugUiStyle::emptyState(view.empty() ? "No log lines yet. Type help below; MCP tool calls appear here."
                                                  : "No line matches the filters");
        char prefix[64];
        ImGuiListClipper clipper;
        clipper.Begin(int(shown.size()));
        while (clipper.Step())
        {
            for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r)
            {
                Entry& e = view[size_t(shown[size_t(r)])];
                ImGui::PushID(shown[size_t(r)]);
                // The time right-aligned in a fixed column (the font is proportional)
                std::snprintf(prefix, sizeof(prefix), "%.2f", e.time);
                const float startX = ImGui::GetCursorPosX();
                ImGui::SetCursorPosX(startX + timeWidth - ImGui::CalcTextSize(prefix).x);
                ImGui::TextDisabled("%s", prefix);
                ImGui::SameLine(startX + timeWidth + 0.6f * em);
                ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
                std::string text;
                switch (e.kind)
                {
                    case Entry::Kind::Log:
                    {
                        color = DebugUiStyle::categoryColor(e.category, e.text);
                        text  = "[" + e.category + "] " + e.text;
                        break;
                    }
                    case Entry::Kind::Input:
                        color = DebugUiStyle::kConsole;
                        text  = "> " + e.text + (e.pending ? "   ... running" : "");
                        break;
                    case Entry::Kind::Output:
                    case Entry::Kind::Error:
                    {
                        color = e.kind == Entry::Kind::Error ? DebugUiStyle::kError : DebugUiStyle::kOutput;
                        text  = e.text;
                        // Long answers folded after kFoldLines lines
                        size_t pos = 0;
                        int lines  = 0;
                        while (!e.expanded && lines < kFoldLines && (pos = text.find('\n', pos)) != std::string::npos)
                        {
                            lines++;
                            pos++;
                        }
                        if (!e.expanded && lines == kFoldLines && pos != std::string::npos)
                        {
                            text = text.substr(0, pos);
                            ImGui::PushStyleColor(ImGuiCol_Text, color);
                            ImGui::TextUnformatted(text.c_str());
                            ImGui::PopStyleColor();
                            if (ImGui::SmallButton("show all")) e.expanded = true;
                            ImGui::PopID();
                            continue;
                        }
                        break;
                    }
                }
                // A line wider than the view is cut with "..." (the whole line in the tooltip); multi-line answers
                // keep their lines
                const float avail  = ImGui::GetContentRegionAvail().x;
                const bool oneLine = text.find('\n') == std::string::npos;
                // The cut line is kept per entry for its width (not searched again every frame)
                if (e.clipWidth != avail || e.clipSource != text)
                {
                    e.clipWidth  = avail;
                    e.clipSource = text;
                    e.clipped    = text;
                    if (oneLine && ImGui::CalcTextSize(text.c_str()).x > avail)
                    {
                        size_t lo = 0, hi = text.size();
                        while (lo + 1 < hi)
                        {
                            const size_t mid = (lo + hi) / 2;
                            if (ImGui::CalcTextSize((text.substr(0, mid) + "...").c_str()).x <= avail)
                                lo = mid;
                            else
                                hi = mid;
                        }
                        e.clipped = text.substr(0, lo) + "...";
                    }
                }
                const std::string& shown = e.clipped;
                const bool clipped       = shown.size() != text.size() || shown != text;
                // An MCP line: the tool name in the MCP color, the rest in the text color (a failed call all red)
                const size_t arrow = e.kind == Entry::Kind::Log && e.category == "mcp" && color == DebugUiStyle::kMcp
                                         ? shown.find(" -> ")
                                         : std::string::npos;
                ImGui::BeginGroup();
                if (arrow != std::string::npos)
                {
                    DebugUiStyle::coloredText(color, shown.substr(0, arrow).c_str());
                    ImGui::SameLine(0, 0);
                    ImGui::TextUnformatted(shown.c_str() + arrow);
                }
                else
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, color);
                    ImGui::TextUnformatted(shown.c_str());
                    ImGui::PopStyleColor();
                }
                ImGui::EndGroup();
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) ImGui::SetClipboardText(text.c_str());
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    if (clipped)
                    {
                        ImGui::PushTextWrapPos(40 * em);
                        ImGui::TextUnformatted(text.c_str());
                        ImGui::PopTextWrapPos();
                        ImGui::Separator();
                    }
                    ImGui::TextDisabled("right-click: copy");
                    ImGui::EndTooltip();
                }
                ImGui::PopID();
            }
        }
        if (scrollToBottom || (atBottom && view.size() != seenEntries))
        {
            ImGui::SetScrollHereY(1.0f);
            scrollToBottom = false;
        }
        if (atBottom || scrollToBottom) seenEntries = view.size();
        // New lines below while scrolled up
        if (!atBottom && view.size() > seenEntries)
        {
            const ImVec2 keep = ImGui::GetCursorPos();
            char label[48];
            std::snprintf(label, sizeof(label), "v %d new", int(view.size() - seenEntries));
            ImGui::SetCursorPos(ImVec2(
                ImGui::GetScrollX() + ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize(label).x - 2 * em,
                ImGui::GetScrollY() + ImGui::GetWindowContentRegionMax().y - 1.6f * em));
            if (ImGui::SmallButton(label)) scrollToBottom = true;
            ImGui::SetCursorPos(keep);
        }
    }
    ImGui::EndChild();

    // The command line
    InputCallbackData data{this, &historyPos};
    auto callback = [](ImGuiInputTextCallbackData* d) -> int
    {
        auto* cd           = static_cast<InputCallbackData*>(d->UserData);
        DebugConsole& con  = *cd->console;
        int& pos           = *cd->historyPos;
        const auto replace = [&](const std::string& s)
        {
            d->DeleteChars(0, d->BufTextLen);
            d->InsertChars(0, s.c_str());
        };
        if (d->EventFlag == ImGuiInputTextFlags_CallbackHistory && !con.history.empty())
        {
            const int n = int(con.history.size());
            if (d->EventKey == ImGuiKey_UpArrow)
                pos = pos < 0 ? n - 1 : std::max(pos - 1, 0);
            else if (d->EventKey == ImGuiKey_DownArrow && pos >= 0)
                pos = pos + 1 < n ? pos + 1 : -1;
            replace(pos >= 0 ? con.history[size_t(pos)] : std::string());
        }
        if (d->EventFlag == ImGuiInputTextFlags_CallbackCompletion)
        {
            const std::string text(d->Buf, size_t(d->BufTextLen));
            if (text.find(' ') != std::string::npos) return 0;  // tool names only
            const auto names = con.complete(text);
            if (names.empty()) return 0;
            // The common prefix of the candidates
            std::string common = names.front();
            for (const std::string& n : names)
            {
                size_t k = 0;
                while (k < common.size() && k < n.size() && common[k] == n[k]) k++;
                common.resize(k);
            }
            if (names.size() == 1) common += " ";
            replace(common);
            if (names.size() > 1)
            {
                std::string list;
                for (const std::string& n : names) list += (list.empty() ? "" : "  ") + n;
                con.output(Entry::Kind::Output, list);
                con.scrollToBottom = true;
            }
        }
        return 0;
    };
    ImGui::SetNextItemWidth(-1);
    if (focusInput) ImGui::SetKeyboardFocusHere();
    const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory |
                                      ImGuiInputTextFlags_CallbackCompletion;
    if (ImGui::InputTextWithHint("##command", "command - help, tools; Tab completes, Up/Down history", &inputLine,
                                 flags, callback, &data))
    {
        submit(inputLine);
        inputLine.clear();
        historyPos     = -1;
        scrollToBottom = true;
        // Keep typing after Enter
        ImGui::SetKeyboardFocusHere(-1);
    }
}

}  // namespace Engine
}  // namespace Saiga

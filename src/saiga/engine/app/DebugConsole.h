/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/core/LogRing.h"
#include "saiga/export.h"

#include <deque>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace Saiga
{
class McpServer;

namespace Engine
{
/**
 * The console of the debug UI (docs/engine_design.md §17.1; S19c): the engine log with filters, and a command line
 * whose commands are the MCP tools, run through McpServer::callLocal (the same argument checks and answers as an
 * agent's call; the call shows up in the "mcp" log lines). Syntax:
 *
 *   tool                      no arguments
 *   tool {"key": value}       a JSON object
 *   tool key=value key=value  by name: a value that parses as JSON (3, true, [1,2,3], "a b") is that JSON, else a
 * string help [tool], tools, clear built-ins
 *
 * The answers go into the console's own view, never into the log ring or stdout. The logic (parse, complete, submit,
 * the view buffer) needs no ImGui; draw() is the ImGui part. Main thread.
 */
class SAIGA_ENGINE_API DebugConsole
{
   public:
    struct Command
    {
        std::string name;  // a tool or a built-in
        json11::Json arguments = json11::Json::object{};
        std::string builtinArgument;  // help <tool>
        bool builtin = false;
    };
    // false and an error for an empty line, invalid JSON or a malformed key=value
    static bool parse(const std::string& line, Command& out, std::string& error);

    struct Entry
    {
        enum class Kind
        {
            Log,     // a line of the log ring
            Input,   // a command typed into the console
            Output,  // an answer or a built-in's output
            Error    // an error answer or a parse error
        };
        Kind kind   = Kind::Log;
        double time = 0;  // s since the start (log lines: LogRing::Line::time)
        std::string category;
        std::string text;
        bool pending  = false;  // Input: the call is not answered yet
        bool expanded = false;  // Output: show more than kFoldLines lines
        // Kept by draw() and visible(): the line cut to the view's width, the lowercase text for the search
        mutable std::string lower;
        std::string clipped, clipSource;
        float clipWidth = -1;
    };

    // server may be null (no tools; the log view works)
    DebugConsole(McpServer* server, LogRing& ring);
    ~DebugConsole();
    DebugConsole(const DebugConsole&)            = delete;
    DebugConsole& operator=(const DebugConsole&) = delete;

    // The new log lines into the view (every frame)
    void update();
    // Runs a command line (a tool call is answered by a later McpServer::poll)
    void submit(const std::string& line);
    // The tool and built-in names starting with prefix (sorted)
    std::vector<std::string> complete(const std::string& prefix) const;
    void clear();

    const std::deque<Entry>& entries() const { return view; }
    int pendingCalls() const { return pending; }
    std::vector<std::string> history;  // oldest first, at most kHistorySize
    // The log view's filters
    std::set<std::string> hiddenCategories;
    std::string search;
    bool mcpOnly = false;
    // Whether an entry passes the filters
    bool visible(const Entry& e) const;
    // All categories seen so far
    const std::set<std::string>& categories() const { return seenCategories; }

    // ImGui: the toolbar (filters), the log view and the command line, filling the available region. focusInput: give
    // the command line the keyboard focus
    void draw(bool focusInput);

    static constexpr size_t kMaxEntries  = 4096;
    static constexpr size_t kHistorySize = 100;
    static constexpr int kFoldLines      = 40;

   private:
    void add(Entry e);
    void output(Entry::Kind kind, const std::string& text);
    void builtin(const Command& c);

    McpServer* server;
    LogRing& ring;
    uint64_t lastSequence = 0;
    std::deque<Entry> view;
    std::set<std::string> seenCategories;
    int pending = 0;
    // The tool answers come later: they check that the console still exists
    std::shared_ptr<DebugConsole*> self;
    mutable std::pair<std::string, std::string> searchLower;  // the search and its lowercase form
    // draw() state
    std::string inputLine;
    int historyPos        = -1;
    size_t seenEntries    = 0;  // entries when the view was last at the bottom
    bool scrollToBottom   = true;
    uint64_t addedEntries = 0;
};

}  // namespace Engine
}  // namespace Saiga

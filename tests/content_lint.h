/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The integer lint of the content generators (docs/engine_design.md §12.1, R-CONTENT-1; S23a), used by the
// ContentLint tests of test_engine_content: a small tokenizer (CRLF, backslash-newline continuations, comments,
// string / character / raw string literals, pp-numbers) and token rules over the stripped text. Every finding names
// file, line and token.

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace ContentLint
{
struct Finding
{
    int line = 0;
    std::string token;
    bool operator==(const Finding& o) const { return line == o.line && token == o.token; }
    bool operator<(const Finding& o) const { return line != o.line ? line < o.line : token < o.token; }
};

struct Include
{
    int line = 0;
    std::string header;  // without the quotes or angle brackets
    bool angled = false;
};

// The stripped text: comments and literals replaced by spaces, one entry per logical line with the number of its
// first physical line; the include directives and the floating literals found on the way
struct Stripped
{
    std::vector<std::pair<int, std::string>> lines;
    std::vector<Include> includes;
    std::vector<Finding> floats;
};

inline bool identStart(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
inline bool identChar(char c)
{
    return identStart(c) || (c >= '0' && c <= '9');
}
inline bool digit(char c)
{
    return c >= '0' && c <= '9';
}

inline Stripped strip(const std::string& raw)
{
    // 1. CRLF -> LF, then the logical lines: a backslash-newline joins (the joined text keeps the first line's number)
    std::string text;
    text.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        if (!(raw[i] == '\r' && i + 1 < raw.size() && raw[i + 1] == '\n')) text += raw[i];
    std::vector<char> chars;
    std::vector<int> lineOf;
    int physical = 1, logical = 1;
    for (size_t i = 0; i < text.size(); ++i)
    {
        if (text[i] == '\\' && i + 1 < text.size() && text[i + 1] == '\n')
        {
            ++i;
            ++physical;
            continue;
        }
        chars.push_back(text[i]);
        lineOf.push_back(logical);
        if (text[i] == '\n') logical = ++physical;
    }

    // 2. Comments, literals, directives and pp-numbers
    Stripped out;
    std::string current;
    int currentLine     = 1;
    bool lineHasContent = false;
    auto put            = [&](char c, int line)
    {
        if (current.empty() && out.lines.empty() && currentLine == 1) currentLine = line;
        if (c == '\n')
        {
            out.lines.push_back({currentLine, current});
            current.clear();
            currentLine    = line + 1;
            lineHasContent = false;
            return;
        }
        if (current.empty()) currentLine = line;
        current += c;
        if (c != ' ' && c != '\t') lineHasContent = true;
    };
    const size_t n = chars.size();
    size_t i       = 0;
    auto at        = [&](size_t k) { return k < n ? chars[k] : '\0'; };
    while (i < n)
    {
        const char c   = chars[i];
        const int line = lineOf[i];
        if (c == '/' && at(i + 1) == '/')
        {
            while (i < n && chars[i] != '\n') ++i;
            continue;
        }
        if (c == '/' && at(i + 1) == '*')
        {
            i += 2;
            while (i < n && !(chars[i] == '*' && at(i + 1) == '/'))
            {
                if (chars[i] == '\n') put('\n', lineOf[i]);
                ++i;
            }
            i += 2;
            put(' ', line);
            continue;
        }
        if (c == '#' && !lineHasContent)
        {
            // A directive: an include records its header; the rest of the line is kept (other directives)
            size_t k = i + 1;
            while (k < n && (chars[k] == ' ' || chars[k] == '\t')) ++k;
            std::string word;
            while (k < n && identChar(chars[k])) word += chars[k++];
            if (word == "include")
            {
                while (k < n && (chars[k] == ' ' || chars[k] == '\t')) ++k;
                const char open = at(k);
                if (open == '"' || open == '<')
                {
                    const char close = open == '"' ? '"' : '>';
                    std::string header;
                    ++k;
                    while (k < n && chars[k] != close && chars[k] != '\n') header += chars[k++];
                    out.includes.push_back({line, header, open == '<'});
                }
                while (k < n && chars[k] != '\n') ++k;
                put('#', line);
                i = k;
                continue;
            }
        }
        if (identStart(c))
        {
            std::string ident;
            size_t k = i;
            while (k < n && identChar(chars[k])) ident += chars[k++];
            const bool rawPrefix = ident == "R" || ident == "u8R" || ident == "uR" || ident == "UR" || ident == "LR";
            if (rawPrefix && at(k) == '"')
            {
                // R"delim( ... )delim"
                size_t m = k + 1;
                std::string delim;
                while (m < n && chars[m] != '(') delim += chars[m++];
                const std::string end = ")" + delim + "\"";
                ++m;
                while (m < n)
                {
                    bool match = true;
                    for (size_t q = 0; q < end.size(); ++q)
                        if (at(m + q) != end[q]) match = false;
                    if (match) break;
                    if (chars[m] == '\n') put('\n', lineOf[m]);
                    ++m;
                }
                i = m + end.size();
                put(' ', line);
                continue;
            }
            const bool prefix = ident == "u8" || ident == "u" || ident == "U" || ident == "L";
            if (prefix && (at(k) == '"' || at(k) == '\''))
            {
                i = k;  // the literal follows
                continue;
            }
            for (char ch : ident) put(ch, line);
            i = k;
            continue;
        }
        if (c == '"' || c == '\'')
        {
            size_t k = i + 1;
            while (k < n && chars[k] != c && chars[k] != '\n')
            {
                if (chars[k] == '\\') ++k;
                ++k;
            }
            i = k + 1;
            put(' ', line);
            continue;
        }
        if (digit(c) || (c == '.' && digit(at(i + 1))))
        {
            // A pp-number: [0-9A-Za-z_.'] and the signs after e E p P
            std::string num;
            size_t k = i;
            while (k < n)
            {
                const char d = chars[k];
                if ((d == '+' || d == '-') && !num.empty() &&
                    (num.back() == 'e' || num.back() == 'E' || num.back() == 'p' || num.back() == 'P'))
                    num += chars[k++];
                else if (identChar(d) || d == '.' || d == '\'')
                    num += chars[k++];
                else
                    break;
            }
            const bool hex = num.size() > 1 && num[0] == '0' && (num[1] == 'x' || num[1] == 'X');
            bool floating  = false;
            if (hex)
                floating = num.find_first_of("pP") != std::string::npos;
            else
                floating = num.find_first_of(".eE") != std::string::npos;
            if (floating) out.floats.push_back({line, num});
            for (char ch : num) put(ch, line);
            i = k;
            continue;
        }
        put(c, line);
        ++i;
    }
    if (!current.empty()) out.lines.push_back({currentLine, current});
    return out;
}

// What a scanned file may include
struct IncludeRules
{
    bool gameContent   = false;  // a file of a game's content/ directory: its own content/ headers
    bool testGenerator = false;  // tests/content_test_*.h
};

inline bool includeAllowed(const Include& inc, const IncludeRules& rules)
{
    static const std::vector<std::string> system = {"cstdint",   "cstddef",     "vector",  "array",
                                                    "string",    "string_view", "map",     "optional",
                                                    "algorithm", "memory",      "utility", "limits"};
    static const std::vector<std::string> engine = {"saiga/engine/content/Generator.h",
                                                    "saiga/engine/content/WorldPlan.h",
                                                    "saiga/engine/content/ChunkContent.h",
                                                    "saiga/engine/world/SpaceKind.h",
                                                    "saiga/engine/ecs/Ids.h",
                                                    "saiga/engine/core/Hash.h",
                                                    "saiga/core/math/ChunkId.h",
                                                    "saiga/core/math/TerrainSurface.h",
                                                    "saiga/core/util/assert.h",
                                                    "saiga/export.h"};
    const std::string& h                         = inc.header;
    if (inc.angled) return std::find(system.begin(), system.end(), h) != system.end();
    if (h.rfind("saiga/engine/content/gen/", 0) == 0 && h.find("..") == std::string::npos) return true;
    if (std::find(engine.begin(), engine.end(), h) != engine.end()) return true;
    if (rules.testGenerator && h.rfind("content_test_", 0) == 0 && h.find('/') == std::string::npos) return true;
    if (rules.gameContent && h.rfind("saiga/", 0) != 0 && h.find("..") == std::string::npos &&
        (h.find('/') == std::string::npos || h.rfind("content/", 0) == 0))
        return true;
    return false;
}

// Every finding of one file's text, sorted by line
inline std::vector<Finding> scan(const std::string& text, const IncludeRules& rules)
{
    static const std::vector<std::regex> rules0 = {
        std::regex(R"(\b(float|double|long|size_t|__int128)\b)"),
        std::regex(R"(\bunordered_\w+)"),
        std::regex(R"(\b\w+_distribution\b)"),
        std::regex(R"(\brand\s*\()"),
        std::regex(R"(\bRng\b)"),
        std::regex(
            R"(\b(sin|cos|tan|asin|acos|atan2?|sinh|cosh|tanh|exp2?|expm1|log(2|10|1p)?|pow|sqrt|cbrt|hypot|fma|)"
            R"(fmod|remainder|round|lround|llround|floor|ceil|trunc|rint|nearbyint|ldexp|frexp|modf|fabs)\s*\()"),
        std::regex(R"(\bstd::hash\b)"),
        std::regex(R"(\b(kHeightUnit|terrainSurfaceDouble|chunkSize|chunkOrigin)\b)"),
        std::regex(R"(\b(Vec[234]|vec[234]|Quat|quat|Mat[34]|mat[34])\b)"),
        std::regex(R"(\bSpaceTemplate\b)"),
    };
    const Stripped s           = strip(text);
    std::vector<Finding> found = s.floats;
    for (const Include& inc : s.includes)
        if (!includeAllowed(inc, rules)) found.push_back({inc.line, inc.header});
    for (const auto& [line, code] : s.lines)
        for (const std::regex& r : rules0)
            for (auto it = std::sregex_iterator(code.begin(), code.end(), r); it != std::sregex_iterator(); ++it)
            {
                std::string token = it->str();
                token.erase(std::remove_if(token.begin(), token.end(), [](char c) { return c == ' ' || c == '\t'; }),
                            token.end());
                found.push_back({line, token});
            }
    std::sort(found.begin(), found.end());
    return found;
}

inline std::string readFile(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace ContentLint

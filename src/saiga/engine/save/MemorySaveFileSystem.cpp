/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/MemorySaveFileSystem.h"

#include "saiga/engine/core/Hash.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
namespace
{
bool fail(std::string* error, const std::string& message)
{
    if (error) *error = message;
    return false;
}

std::vector<std::string> split(const std::string& path)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : path)
    {
        if (c == '/')
        {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        }
        else
            cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
}  // namespace

std::string MemorySaveFileSystem::Op::describe() const
{
    static const char* const names[] = {"makeDirectory", "writeNew",      "append", "truncate",
                                        "syncFile",      "syncDirectory", "rename", "remove"};
    std::string s                    = std::string(names[int(kind)]) + " " + path;
    if (kind == OpKind::Rename) s += " -> " + to;
    if (kind == OpKind::WriteNew || kind == OpKind::Append) s += " (" + std::to_string(data.size()) + " bytes)";
    if (kind == OpKind::Truncate) s += " to " + std::to_string(size);
    return s;
}

MemorySaveFileSystem::MemorySaveFileSystem()
{
    nodes.emplace_back();
    nodes[0].dir = true;
}

MemorySaveFileSystem::~MemorySaveFileSystem() = default;

int MemorySaveFileSystem::resolve(const std::string& path) const
{
    int n = 0;
    for (const std::string& part : split(path))
    {
        if (!nodes[size_t(n)].dir) return -1;
        auto it = nodes[size_t(n)].entries.find(part);
        if (it == nodes[size_t(n)].entries.end()) return -1;
        n = it->second;
    }
    return n;
}

bool MemorySaveFileSystem::parentOf(const std::string& path, int& parent, std::string& name) const
{
    std::vector<std::string> parts = split(path);
    if (parts.empty()) return false;
    name = parts.back();
    parts.pop_back();
    std::string dir;
    for (const std::string& p : parts) dir += (dir.empty() ? "" : "/") + p;
    parent = resolve(dir);
    return parent >= 0 && nodes[size_t(parent)].dir;
}

bool MemorySaveFileSystem::exists(const std::string& path)
{
    std::lock_guard<std::mutex> l(mutex);
    return resolve(path) >= 0;
}

bool MemorySaveFileSystem::isDirectory(const std::string& path)
{
    std::lock_guard<std::mutex> l(mutex);
    const int n = resolve(path);
    return n >= 0 && nodes[size_t(n)].dir;
}

bool MemorySaveFileSystem::list(const std::string& dir, std::vector<std::string>& names, std::string* error)
{
    std::lock_guard<std::mutex> l(mutex);
    names.clear();
    const int n = resolve(dir);
    if (n < 0 || !nodes[size_t(n)].dir) return fail(error, "list " + dir + ": no such directory");
    for (const auto& kv : nodes[size_t(n)].entries) names.push_back(kv.first);
    return true;
}

bool MemorySaveFileSystem::fileSize(const std::string& path, uint64_t& size, std::string* error)
{
    std::lock_guard<std::mutex> l(mutex);
    const int n = resolve(path);
    if (n < 0 || nodes[size_t(n)].dir) return fail(error, "size of " + path + ": no such file");
    size = nodes[size_t(n)].data.size();
    return true;
}

bool MemorySaveFileSystem::readAt(const std::string& path, uint64_t offset, uint64_t size, uint8_t* out,
                                  std::string* error)
{
    std::lock_guard<std::mutex> l(mutex);
    const int n = resolve(path);
    if (n < 0 || nodes[size_t(n)].dir) return fail(error, "read " + path + ": no such file");
    const std::vector<uint8_t>& d = nodes[size_t(n)].data;
    if (offset > d.size() || size > d.size() - offset) return fail(error, "read " + path + ": unexpected end of file");
    std::copy(d.begin() + std::ptrdiff_t(offset), d.begin() + std::ptrdiff_t(offset + size), out);
    return true;
}

bool MemorySaveFileSystem::count(const Op& op, std::string* error, bool* tornNow)
{
    *tornNow = false;
    if (failLimit >= 0 && int64_t(ops) >= failLimit)
    {
        ops++;
        if (torn && !tornUsed && (op.kind == OpKind::Append || op.kind == OpKind::WriteNew))
        {
            tornUsed = true;
            *tornNow = true;
            return true;
        }
        return fail(error, "simulated crash (" + op.describe() + ")");
    }
    ops++;
    log.push_back(op);
    return true;
}

bool MemorySaveFileSystem::apply(const Op& op, std::string* error, bool tornHalf)
{
    switch (op.kind)
    {
        case OpKind::MakeDirectory:
        {
            const int existing = resolve(op.path);
            if (existing >= 0)
                return nodes[size_t(existing)].dir || fail(error, "create directory " + op.path + ": a file exists");
            int parent;
            std::string name;
            if (!parentOf(op.path, parent, name)) return fail(error, "create directory " + op.path + ": no parent");
            nodes.emplace_back();
            nodes.back().dir                    = true;
            nodes[size_t(parent)].entries[name] = int(nodes.size() - 1);
            return true;
        }
        case OpKind::WriteNew:
        case OpKind::Append:
        {
            const size_t n = tornHalf ? op.data.size() / 2 : op.data.size();
            int node       = resolve(op.path);
            if (node >= 0 && nodes[size_t(node)].dir) return fail(error, "write " + op.path + ": a directory");
            if (node < 0)
            {
                int parent;
                std::string name;
                if (!parentOf(op.path, parent, name)) return fail(error, "write " + op.path + ": no such directory");
                nodes.emplace_back();
                node                                = int(nodes.size() - 1);
                nodes[size_t(parent)].entries[name] = node;
            }
            std::vector<uint8_t>& d = nodes[size_t(node)].data;
            if (op.kind == OpKind::WriteNew) d.clear();
            d.insert(d.end(), op.data.begin(), op.data.begin() + std::ptrdiff_t(n));
            if (tornHalf) return fail(error, "simulated crash (torn " + op.describe() + ")");
            return true;
        }
        case OpKind::Truncate:
        {
            const int node = resolve(op.path);
            if (node < 0 || nodes[size_t(node)].dir) return fail(error, "truncate " + op.path + ": no such file");
            nodes[size_t(node)].data.resize(size_t(op.size), 0);
            return true;
        }
        case OpKind::SyncFile:
        {
            const int node = resolve(op.path);
            if (node < 0 || nodes[size_t(node)].dir) return fail(error, "sync " + op.path + ": no such file");
            nodes[size_t(node)].durable = nodes[size_t(node)].data;
            return true;
        }
        case OpKind::SyncDirectory:
        {
            const int node = resolve(op.path);
            if (node < 0 || !nodes[size_t(node)].dir) return fail(error, "sync " + op.path + ": no such directory");
            nodes[size_t(node)].durableEntries = nodes[size_t(node)].entries;
            return true;
        }
        case OpKind::Rename:
        {
            int fromParent, toParent;
            std::string fromName, toName;
            if (!parentOf(op.path, fromParent, fromName) || !nodes[size_t(fromParent)].entries.count(fromName))
                return fail(error, "rename " + op.path + ": no such file");
            if (!parentOf(op.to, toParent, toName)) return fail(error, "rename to " + op.to + ": no such directory");
            const int node = nodes[size_t(fromParent)].entries[fromName];
            nodes[size_t(fromParent)].entries.erase(fromName);
            nodes[size_t(toParent)].entries[toName] = node;
            return true;
        }
        case OpKind::Remove:
        {
            int parent;
            std::string name;
            if (!parentOf(op.path, parent, name) || !nodes[size_t(parent)].entries.count(name))
                return fail(error, "remove " + op.path + ": no such file");
            const int node = nodes[size_t(parent)].entries[name];
            if (nodes[size_t(node)].dir && !nodes[size_t(node)].entries.empty())
                return fail(error, "remove " + op.path + ": the directory is not empty");
            nodes[size_t(parent)].entries.erase(name);
            return true;
        }
    }
    return false;
}

#define SAIGA_MEMFS_OP(OPDEF)                      \
    std::lock_guard<std::mutex> l(mutex);          \
    Op op = OPDEF;                                 \
    bool tornNow;                                  \
    if (!count(op, error, &tornNow)) return false; \
    return apply(op, error, tornNow)

namespace
{
MemorySaveFileSystem::Op makeOp(MemorySaveFileSystem::OpKind kind, const std::string& path, const std::string& to = "",
                                const uint8_t* data = nullptr, size_t size = 0, uint64_t length = 0)
{
    MemorySaveFileSystem::Op op;
    op.kind = kind;
    op.path = path;
    op.to   = to;
    if (data) op.data.assign(data, data + size);
    op.size = length;
    return op;
}
}  // namespace

bool MemorySaveFileSystem::makeDirectory(const std::string& dir, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::MakeDirectory, dir));
}
bool MemorySaveFileSystem::writeNew(const std::string& path, const uint8_t* data, size_t size, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::WriteNew, path, "", data, size));
}
bool MemorySaveFileSystem::append(const std::string& path, const uint8_t* data, size_t size, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::Append, path, "", data, size));
}
bool MemorySaveFileSystem::truncate(const std::string& path, uint64_t size, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::Truncate, path, "", nullptr, 0, size));
}
bool MemorySaveFileSystem::syncFile(const std::string& path, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::SyncFile, path));
}
bool MemorySaveFileSystem::syncDirectory(const std::string& dir, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::SyncDirectory, dir));
}
bool MemorySaveFileSystem::rename(const std::string& from, const std::string& to, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::Rename, from, to));
}
bool MemorySaveFileSystem::remove(const std::string& path, std::string* error)
{
    SAIGA_MEMFS_OP(makeOp(OpKind::Remove, path));
}
#undef SAIGA_MEMFS_OP

bool MemorySaveFileSystem::lock(std::string* error)
{
    std::lock_guard<std::mutex> l(mutex);
    if (locked) return fail(error, "the save directory (memory) is in use (locked by another world)");
    locked = true;
    // The lock file exists while the directory is used (not counted, durable at once)
    if (!nodes[0].entries.count("lock"))
    {
        nodes.emplace_back();
        nodes[0].entries["lock"]        = int(nodes.size() - 1);
        nodes[0].durableEntries["lock"] = int(nodes.size() - 1);
    }
    return true;
}

void MemorySaveFileSystem::unlock()
{
    std::lock_guard<std::mutex> l(mutex);
    locked = false;
}

uint64_t MemorySaveFileSystem::operations() const
{
    std::lock_guard<std::mutex> l(mutex);
    return ops;
}

std::vector<MemorySaveFileSystem::Op> MemorySaveFileSystem::operationLog() const
{
    std::lock_guard<std::mutex> l(mutex);
    return log;
}

void MemorySaveFileSystem::clearLog()
{
    std::lock_guard<std::mutex> l(mutex);
    log.clear();
    ops = 0;
}

void MemorySaveFileSystem::failAfter(int64_t n, bool tornWrite)
{
    std::lock_guard<std::mutex> l(mutex);
    failLimit = n < 0 ? -1 : int64_t(ops) + n;
    torn      = tornWrite;
    tornUsed  = false;
}

std::shared_ptr<MemorySaveFileSystem> MemorySaveFileSystem::clone() const
{
    std::lock_guard<std::mutex> l(mutex);
    auto c   = std::make_shared<MemorySaveFileSystem>();
    c->nodes = nodes;
    return c;
}

std::shared_ptr<MemorySaveFileSystem> MemorySaveFileSystem::replay(const std::vector<Op>& ops, size_t n,
                                                                   bool tornLast) const
{
    auto c = clone();
    for (size_t i = 0; i < n && i < ops.size(); ++i) c->apply(ops[i], nullptr, false);
    if (tornLast && n < ops.size() && (ops[n].kind == OpKind::Append || ops[n].kind == OpKind::WriteNew))
        c->apply(ops[n], nullptr, true);
    return c;
}

void MemorySaveFileSystem::copyTree(const MemorySaveFileSystem& from, int fromNode, int toNode, bool durableEntries,
                                    bool durableBytes)
{
    const Node& src = from.nodes[size_t(fromNode)];
    const auto& es  = durableEntries ? src.durableEntries : src.entries;
    for (const auto& [name, child] : es)
    {
        const Node& c = from.nodes[size_t(child)];
        nodes.emplace_back();
        const int id          = int(nodes.size() - 1);
        nodes[size_t(id)].dir = c.dir;
        if (!c.dir)
        {
            nodes[size_t(id)].data    = durableBytes ? c.durable : c.data;
            nodes[size_t(id)].durable = nodes[size_t(id)].data;
        }
        nodes[size_t(toNode)].entries[name]        = id;
        nodes[size_t(toNode)].durableEntries[name] = id;
        if (c.dir) copyTree(from, child, id, durableEntries, durableBytes);
    }
}

std::shared_ptr<MemorySaveFileSystem> MemorySaveFileSystem::afterPowerLoss() const
{
    std::lock_guard<std::mutex> l(mutex);
    auto c = std::make_shared<MemorySaveFileSystem>();
    c->copyTree(*this, 0, 0, true, true);
    return c;
}

std::shared_ptr<MemorySaveFileSystem> MemorySaveFileSystem::afterEntriesOnly() const
{
    std::lock_guard<std::mutex> l(mutex);
    auto c = std::make_shared<MemorySaveFileSystem>();
    c->copyTree(*this, 0, 0, false, true);
    return c;
}

uint64_t MemorySaveFileSystem::stateHash() const
{
    std::lock_guard<std::mutex> l(mutex);
    Hasher h;
    std::vector<std::pair<int, std::string>> stack{{0, ""}};
    while (!stack.empty())
    {
        auto [n, path] = stack.back();
        stack.pop_back();
        const Node& node = nodes[size_t(n)];
        h.addString(path).addU8(node.dir ? 1 : 0);
        if (!node.dir) h.addU64(fnv1a64Bytes(node.data.data(), node.data.size())).addU64(node.data.size());
        for (auto it = node.entries.rbegin(); it != node.entries.rend(); ++it)
            stack.push_back({it->second, path + "/" + it->first});
    }
    return h.value();
}

std::vector<std::string> MemorySaveFileSystem::files() const
{
    std::lock_guard<std::mutex> l(mutex);
    std::vector<std::string> out;
    std::vector<std::pair<int, std::string>> stack{{0, ""}};
    while (!stack.empty())
    {
        auto [n, path] = stack.back();
        stack.pop_back();
        for (const auto& [name, child] : nodes[size_t(n)].entries)
        {
            const std::string p = path.empty() ? name : path + "/" + name;
            if (nodes[size_t(child)].dir)
                stack.push_back({child, p});
            else
                out.push_back(p);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

void MemorySaveFileSystem::putFile(const std::string& path, const std::vector<uint8_t>& data)
{
    std::lock_guard<std::mutex> l(mutex);
    // Parents created as needed
    std::vector<std::string> parts = split(path);
    int n                          = 0;
    for (size_t i = 0; i + 1 < parts.size(); ++i)
    {
        auto it = nodes[size_t(n)].entries.find(parts[i]);
        if (it == nodes[size_t(n)].entries.end())
        {
            nodes.emplace_back();
            nodes.back().dir                          = true;
            const int id                              = int(nodes.size() - 1);
            nodes[size_t(n)].entries[parts[i]]        = id;
            nodes[size_t(n)].durableEntries[parts[i]] = id;
            n                                         = id;
        }
        else
            n = it->second;
    }
    int f   = -1;
    auto it = nodes[size_t(n)].entries.find(parts.back());
    if (it != nodes[size_t(n)].entries.end())
        f = it->second;
    else
    {
        nodes.emplace_back();
        f                                             = int(nodes.size() - 1);
        nodes[size_t(n)].entries[parts.back()]        = f;
        nodes[size_t(n)].durableEntries[parts.back()] = f;
    }
    nodes[size_t(f)].data    = data;
    nodes[size_t(f)].durable = data;
}

std::vector<uint8_t> MemorySaveFileSystem::fileBytes(const std::string& path) const
{
    std::lock_guard<std::mutex> l(mutex);
    const int n = resolve(path);
    return n < 0 || nodes[size_t(n)].dir ? std::vector<uint8_t>() : nodes[size_t(n)].data;
}

}  // namespace Engine
}  // namespace Saiga

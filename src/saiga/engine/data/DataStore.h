/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/data/DataFile.h"
#include "saiga/engine/ecs/FieldList.h"
#include "saiga/engine/ecs/TypeCheck.h"
#include "saiga/export.h"

#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace Saiga
{
namespace Engine
{
// The tables of one data type (type-erased base)
class SAIGA_ENGINE_API DataTableBase
{
   public:
    DataTableBase(const char* name, const std::type_info& type) : name(name), type(&type) {}
    virtual ~DataTableBase() = default;
    // Reads the file into a pending table (nothing visible changes); false with an error naming file and key
    virtual bool read(const std::string& path, std::string& error) = 0;
    // Pending entries that replace existing ones must keep every existing name (entries are never removed while
    // running: components refer to them)
    virtual bool checkPending(std::string& error) const = 0;
    // Applies the pending table: existing entries are updated in place (references stay valid), new ones added
    virtual void applyPending()                                          = 0;
    virtual std::vector<std::string> names() const                       = 0;
    virtual const json11::Json entryJson(const std::string& entry) const = 0;  // null if missing
    virtual std::vector<FieldInfo> fields() const                        = 0;

    const char* const name;
    const std::type_info* const type;
    std::string file;  // the file name (snake_case(name).json)
};

template <class T>
class DataTable final : public DataTableBase
{
   public:
    DataTable() : DataTableBase(T::name, typeid(T)) {}

    bool read(const std::string& path, std::string& error) override
    {
        pending.clear();
        return readDataFile(path, pending, error);
    }
    bool checkPending(std::string& error) const override
    {
        for (const auto& kv : entries)
            if (!pending.count(kv.first))
            {
                error = file + ": the entry " + kv.first + " was removed (entries can only be added while running)";
                return false;
            }
        return true;
    }
    void applyPending() override
    {
        for (auto& kv : pending)
        {
            auto it = entries.find(kv.first);
            if (it == entries.end())
                entries.emplace(kv.first, std::make_unique<T>(std::move(kv.second)));
            else
                *it->second = std::move(kv.second);
        }
        pending.clear();
    }
    std::vector<std::string> names() const override
    {
        std::vector<std::string> out;
        for (const auto& kv : entries) out.push_back(kv.first);
        return out;
    }
    const json11::Json entryJson(const std::string& entry) const override
    {
        const T* v = find(entry);
        return v ? toJson(*v) : json11::Json();
    }
    std::vector<FieldInfo> fields() const override { return describeFields<T>(); }

    const T* find(const std::string& entry) const
    {
        auto it = entries.find(entry);
        return it == entries.end() ? nullptr : it->second.get();
    }

   private:
    std::map<std::string, std::unique_ptr<T>> entries;  // stable addresses (DataRef caches them)
    std::map<std::string, T> pending;
};

// The registered data types (WorldSetup::dataTypes)
class SAIGA_ENGINE_API DataTypeList
{
   public:
    // T: name, version and visit; its file is <dataDirectory>/<snake_case(T::name)>.json
    template <class T>
    void add()
    {
        checkVisitFields<T>();
        for (auto& t : tables)
            SAIGA_ASSERT(std::string(t->name) != T::name,
                         std::string("DataTypeList::add: ") + T::name + " is already registered");
        auto table  = std::make_unique<DataTable<T>>();
        table->file = snakeCase(T::name) + ".json";
        tables.push_back(std::move(table));
    }
    std::vector<std::unique_ptr<DataTableBase>> tables;
};

/**
 * The data of a world (docs/engine_design.md §14): one table per registered type, loaded from the data directory.
 * Shared by worlds of one setup (WorldSetup::sharedData), so scenario worlds do not read the files again.
 */
class SAIGA_ENGINE_API DataStore
{
   public:
    DataStore(DataTypeList types, std::string directory);
    // Reads every file; false with the first error (nothing applied)
    bool load(std::string& error);
    // Reads every file again and applies them only if all are valid and no entry was removed
    bool reload(std::string& error);

    template <class T>
    const DataTable<T>* table() const
    {
        for (auto& t : types.tables)
            if (isRegisteredType(*t->type, typeid(T))) return static_cast<const DataTable<T>*>(t.get());
        return nullptr;
    }
    const DataTableBase* find(const std::string& name) const;
    const std::vector<std::unique_ptr<DataTableBase>>& tables() const { return types.tables; }
    const std::string& directory() const { return dir; }

   private:
    bool readAll(std::string& error);
    DataTypeList types;
    std::string dir;
};

}  // namespace Engine
}  // namespace Saiga

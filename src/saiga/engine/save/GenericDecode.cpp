/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SaveDecoder: the binary form of saved values by their saved field lists, into the JSON form of VisitJson.h.

#include "saiga/engine/save/GenericDecode.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json = json11::Json;

// Like JsonWriter::write for integers: exact in a double below 2^53, larger values as strings
Json integerJson(int64_t v)
{
    constexpr int64_t kExact = int64_t(1) << 53;
    if (v >= kExact || v <= -kExact) return Json(std::to_string(v));
    return Json(double(v));
}
Json unsignedJson(uint64_t v)
{
    constexpr uint64_t kExact = uint64_t(1) << 53;
    if (v >= kExact) return Json(std::to_string(v));
    return Json(double(v));
}
Json doubleJson(double v)
{
    if (std::isnan(v)) return Json("nan");
    if (std::isinf(v)) return Json(v > 0 ? "inf" : "-inf");
    return Json(v);
}

template <class T>
bool readAs(BinaryReader& r, T& v)
{
    r.read(v);
    return r.ok();
}

// "vec3d" -> (3, 1, double), "mat4f" -> (4, 4, float)
bool mathShape(const std::string& t, int& rows, int& cols, bool& isDouble)
{
    if (t.size() != 5 || (t.compare(0, 3, "vec") != 0 && t.compare(0, 3, "mat") != 0)) return false;
    if (t[3] < '1' || t[3] > '9' || (t[4] != 'd' && t[4] != 'f')) return false;
    rows     = t[3] - '0';
    cols     = t[0] == 'v' ? 1 : rows;
    isDouble = t[4] == 'd';
    return true;
}

constexpr int kMaxDepth = 64;
}  // namespace

bool SaveDecoder::decodeValue(const std::string& type, const FieldInfo& field, BinaryReader& r, Json& out,
                              std::string& error)
{
    auto fail = [&](const std::string& m)
    {
        if (error.empty()) error = m;
        return false;
    };
    if (type == "bool")
    {
        uint8_t b = 0;
        if (!readAs(r, b)) return false;
        if (b > 1) return fail("bool byte " + std::to_string(b) + " is neither 0 nor 1");
        out = Json(b != 0);
        return true;
    }
    if (type == "int8" || type == "int16" || type == "int32" || type == "int64")
    {
        if (type == "int8")
        {
            int8_t v;
            if (!readAs(r, v)) return false;
            out = integerJson(v);
        }
        else if (type == "int16")
        {
            int16_t v;
            if (!readAs(r, v)) return false;
            out = integerJson(v);
        }
        else if (type == "int32")
        {
            int32_t v;
            if (!readAs(r, v)) return false;
            out = integerJson(v);
        }
        else
        {
            int64_t v;
            if (!readAs(r, v)) return false;
            out = integerJson(v);
        }
        return true;
    }
    if (type == "uint8" || type == "uint16" || type == "uint32" || type == "uint64" || type == "PersistentId")
    {
        uint64_t v = 0;
        if (type == "uint8")
        {
            uint8_t x;
            if (!readAs(r, x)) return false;
            v = x;
        }
        else if (type == "uint16")
        {
            uint16_t x;
            if (!readAs(r, x)) return false;
            v = x;
        }
        else if (type == "uint32")
        {
            uint32_t x;
            if (!readAs(r, x)) return false;
            v = x;
        }
        else if (!readAs(r, v))
            return false;
        out = unsignedJson(v);
        return true;
    }
    if (type == "double")
    {
        double v;
        if (!readAs(r, v)) return false;
        out = doubleJson(v);
        return true;
    }
    if (type == "float")
    {
        float v;
        if (!readAs(r, v)) return false;
        out = doubleJson(double(v));
        return true;
    }
    if (type == "string")
    {
        std::string v;
        if (!readAs(r, v)) return false;
        out = Json(v);
        return true;
    }
    if (type == "Entity")
    {
        uint32_t index = 0, generation = 0;
        if (!readAs(r, index) || !readAs(r, generation)) return false;
        out = Json(Json::object{{"index", double(index)}, {"generation", double(generation)}});
        return true;
    }
    if (type == "enum")
    {
        int64_t v = 0;
        if (!readAs(r, v)) return false;
        if (v < 0 || uint64_t(v) >= field.enumValues.size())
            return fail(field.name + ": enum value " + std::to_string(v) + " out of range");
        out = Json(field.enumValues[size_t(v)]);
        return true;
    }
    if (type == "quatd" || type == "quatf")
    {
        Json::array a;
        for (int k = 0; k < 4; ++k)
        {
            Json c;
            if (!decodeValue(type == "quatd" ? "double" : "float", field, r, c, error)) return false;
            a.push_back(c);
        }
        out = a;
        return true;
    }
    int rows = 0, cols = 0;
    bool isDouble = true;
    if (mathShape(type, rows, cols, isDouble))
    {
        Json::array a;
        for (int k = 0; k < rows * cols; ++k)
        {
            Json c;
            if (!decodeValue(isDouble ? "double" : "float", field, r, c, error)) return false;
            a.push_back(c);
        }
        out = a;
        return true;
    }
    if (type.compare(0, 7, "vector<") == 0 && type.back() == '>')
    {
        const std::string element = type.substr(7, type.size() - 8);
        uint64_t n                = 0;
        if (!readAs(r, n)) return false;
        // Bounded by the input (elements of zero bytes are capped like the binary reader does)
        const uint64_t left = uint64_t(r.remaining());
        if (n > left && n > kMaxEmptyElements) return fail(field.name + ": a vector longer than the input");
        Json::array a;
        for (uint64_t k = 0; k < n; ++k)
        {
            Json e;
            const size_t before = r.position();
            if (!decodeValue(element, field, r, e, error) || !spendZeroByte(before, r, error)) return false;
            a.push_back(e);
        }
        out = a;
        return true;
    }
    if (type.compare(0, 6, "array<") == 0 && type.back() == '>')
    {
        const size_t comma = type.rfind(',');
        if (comma == std::string::npos || comma < 6) return fail("malformed field type " + type);
        const std::string element = type.substr(6, comma - 6);
        const std::string count   = type.substr(comma + 1, type.size() - comma - 2);
        uint64_t n                = 0;
        if (count.empty() || count.size() > 10) return fail("malformed field type " + type);
        for (char c : count)
        {
            if (c < '0' || c > '9') return fail("malformed field type " + type);
            n = n * 10 + uint64_t(c - '0');
        }
        if (n > uint64_t(r.remaining()) && n > kMaxEmptyElements)
            return fail(field.name + ": an array longer than the input");
        Json::array a;
        for (uint64_t k = 0; k < n; ++k)
        {
            Json e;
            const size_t before = r.position();
            if (!decodeValue(element, field, r, e, error) || !spendZeroByte(before, r, error)) return false;
            a.push_back(e);
        }
        out = a;
        return true;
    }
    // A nested struct, by its saved schema; migrated to the registered struct of the same name if there is one
    const SchemaType* nested = saved.findStruct(type);
    if (!nested) return fail(field.name + ": the saved schema has no type " + type);
    const SchemaType* target = current ? current->findStruct(type) : nullptr;
    return decode(*nested, r, target, out, error);
}

bool SaveDecoder::spendZeroByte(size_t before, const BinaryReader& r, std::string& error)
{
    if (r.position() != before) return true;
    if (zeroByteBudget == 0)
    {
        if (error.empty()) error = "too many elements without bytes (a malformed save)";
        return false;
    }
    zeroByteBudget--;
    return true;
}

bool SaveDecoder::skipValue(const std::string& type, const FieldInfo& field, BinaryReader& r, std::string& error)
{
    auto fail = [&](const std::string& m)
    {
        if (error.empty()) error = m;
        return false;
    };
    auto bytes = [&](uint64_t n) { return r.skip(n) || fail(field.name + ": " + r.error()); };
    if (type == "bool")
    {
        uint8_t b = 0;
        if (!readAs(r, b)) return fail(field.name + ": " + r.error());
        return b <= 1 || fail("bool byte " + std::to_string(b) + " is neither 0 nor 1");
    }
    if (type == "int8" || type == "uint8") return bytes(1);
    if (type == "int16" || type == "uint16") return bytes(2);
    if (type == "int32" || type == "uint32" || type == "float") return bytes(4);
    if (type == "int64" || type == "uint64" || type == "double" || type == "PersistentId" || type == "Entity")
        return bytes(8);
    if (type == "string")
    {
        uint64_t n = 0;
        if (!readAs(r, n)) return fail(field.name + ": " + r.error());
        return bytes(n);
    }
    if (type == "enum")
    {
        int64_t v = 0;
        if (!readAs(r, v)) return fail(field.name + ": " + r.error());
        return (v >= 0 && uint64_t(v) < field.enumValues.size()) ||
               fail(field.name + ": enum value " + std::to_string(v) + " out of range");
    }
    if (type == "quatd") return bytes(32);
    if (type == "quatf") return bytes(16);
    int rows = 0, cols = 0;
    bool isDouble = true;
    if (mathShape(type, rows, cols, isDouble)) return bytes(uint64_t(rows * cols) * (isDouble ? 8 : 4));
    if (type.compare(0, 7, "vector<") == 0 && type.back() == '>')
    {
        const std::string element = type.substr(7, type.size() - 8);
        uint64_t n                = 0;
        if (!readAs(r, n)) return fail(field.name + ": " + r.error());
        if (n > uint64_t(r.remaining()) && n > kMaxEmptyElements)
            return fail(field.name + ": a vector longer than the input");
        for (uint64_t k = 0; k < n; ++k)
        {
            const size_t before = r.position();
            if (!skipValue(element, field, r, error) || !spendZeroByte(before, r, error)) return false;
        }
        return true;
    }
    if (type.compare(0, 6, "array<") == 0)
    {
        // Rare: decoded (it parses the count)
        json11::Json ignored;
        return decodeValue(type, field, r, ignored, error);
    }
    const SchemaType* nested = saved.findStruct(type);
    if (!nested) return fail(field.name + ": the saved schema has no type " + type);
    return skip(*nested, r, error);
}

bool SaveDecoder::skip(const SchemaType& savedType, BinaryReader& r, std::string& error)
{
    if (depth >= kMaxDepth)
    {
        error = savedType.name + ": nested deeper than " + std::to_string(kMaxDepth) + " structs";
        return false;
    }
    depth++;
    const size_t start = r.position();
    for (const FieldInfo& f : savedType.fields)
    {
        const size_t before = r.position();
        if (!skipValue(f.type, f, r, error) || !spendZeroByte(before, r, error))
        {
            depth--;
            if (error.find(savedType.name) != 0) error = savedType.name + "." + error;
            return false;
        }
    }
    depth--;
    // A value without bytes (an empty struct) counts too
    return spendZeroByte(start, r, error);
}

bool SaveDecoder::decode(const SchemaType& savedType, BinaryReader& r, const SchemaType* target, Json& out,
                         std::string& error)
{
    // R-SAVE-1 for nested structs too: at the same version, a field must keep its type
    if (target && target->version == savedType.version)
        for (const FieldInfo& f : savedType.fields)
            for (const FieldInfo& g : target->fields)
                if (f.name == g.name && f.type != g.type)
                {
                    error = savedType.name + "." + f.name + " changed its type from " + f.type + " to " + g.type +
                            " without a version increment (R-SAVE-1)";
                    return false;
                }
    if (depth >= kMaxDepth)
    {
        error = savedType.name + ": nested deeper than " + std::to_string(kMaxDepth) + " structs";
        return false;
    }
    depth++;
    FieldMap fields;
    const size_t start = r.position();
    for (const FieldInfo& f : savedType.fields)
    {
        Json v;
        const size_t before = r.position();
        if (!decodeValue(f.type, f, r, v, error) || !spendZeroByte(before, r, error))
        {
            depth--;
            if (error.empty())
                error = savedType.name + "." + f.name + ": " + r.error();
            else if (error.find(savedType.name) != 0)
                error = savedType.name + "." + error;
            return false;
        }
        fields.fields[f.name] = v;
    }
    depth--;
    if (!spendZeroByte(start, r, error)) return false;
    if (current && !migrate(savedType.name, savedType.version, target, fields, error)) return false;
    out = Json(fields.fields);
    return true;
}

// A game migration runs on untrusted values: an exception is a load error (the world is unchanged)
bool SaveDecoder::runMigration(MigrationList::Fn fn, const std::string& typeName, uint32_t version, FieldMap& fields,
                               std::string& error)
{
    try
    {
        fn(fields);
        return true;
    }
    catch (const std::exception& e)
    {
        error = "the migration of " + typeName + " from version " + std::to_string(version) + " threw: " + e.what();
    }
    catch (...)
    {
        error = "the migration of " + typeName + " from version " + std::to_string(version) + " threw";
    }
    return false;
}

bool SaveDecoder::migrate(const std::string& typeName, uint32_t fromVersion, const SchemaType* target, FieldMap& fields,
                          std::string& error) const
{
    if (target)
    {
        if (fromVersion > target->version)
        {
            error = typeName + " was saved with version " + std::to_string(fromVersion) +
                    ", newer than the registered version " + std::to_string(target->version);
            return false;
        }
        for (uint32_t v = fromVersion; v < target->version; ++v)
        {
            MigrationList::Fn fn = migrations ? migrations->find(typeName, v) : nullptr;
            if (!fn)
            {
                error = typeName + ": no migration from version " + std::to_string(v) + " (saved " +
                        std::to_string(fromVersion) + ", registered " + std::to_string(target->version) + ")";
                return false;
            }
            if (!runMigration(fn, typeName, v, fields, error)) return false;
        }
        // Fields that no longer exist are dropped (new fields keep their defaults when the result is merged)
        for (auto it = fields.fields.begin(); it != fields.fields.end();)
        {
            bool known = false;
            for (const FieldInfo& f : target->fields) known = known || f.name == it->first;
            it = known ? std::next(it) : fields.fields.erase(it);
        }
        return true;
    }
    // A type without a registered counterpart (removed, or a nested struct a parent migration converts): its own
    // migrations as far as they go
    for (uint32_t v = fromVersion; migrations; ++v)
    {
        MigrationList::Fn fn = migrations->find(typeName, v);
        if (!fn) break;
        if (!runMigration(fn, typeName, v, fields, error)) return false;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga

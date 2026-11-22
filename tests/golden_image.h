/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

// True on lavapipe, the device the goldens are rendered on: only there do the golden images and the committed counts
// gate.
inline bool goldenGating(const std::string& deviceName)
{
    return deviceName.find("llvmpipe") != std::string::npos;
}

// SAIGA_UPDATE_GOLDEN=1: compareGolden writes the goldens instead of comparing (lavapipe only)
inline bool goldenUpdateMode()
{
    const char* update = std::getenv("SAIGA_UPDATE_GOLDEN");
    return update && std::string(update) == "1";
}

// The golden metrics of two images of the same size (§17.6): the mean absolute difference per channel (/255), the
// fraction of the pixels with a channel differing by more than 16, the largest difference; pass: mean < 1 and at most
// 0.5 % large differences
struct GoldenMetrics
{
    double meanAbs = 0, largeFraction = 0;
    int maxDelta = 0;
    bool pass    = false;
};
inline GoldenMetrics goldenMetrics(const Saiga::TemplatedImage<Saiga::ucvec4>& a,
                                   const Saiga::TemplatedImage<Saiga::ucvec4>& b)
{
    GoldenMetrics m;
    if (a.w != b.w || a.h != b.h || a.w == 0) return m;
    double sum        = 0;
    long large        = 0;
    const long pixels = long(a.w) * a.h;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x)
        {
            int pixelMax = 0;
            for (int c = 0; c < 3; ++c)
            {
                const int d = std::abs(int(a(y, x)(c)) - int(b(y, x)(c)));
                sum += d;
                pixelMax = std::max(pixelMax, d);
            }
            if (pixelMax > 16) large++;
            m.maxDelta = std::max(m.maxDelta, pixelMax);
        }
    m.meanAbs       = sum / (double(pixels) * 3);
    m.largeFraction = double(large) / double(pixels);
    m.pass          = m.meanAbs < 1.0 && m.largeFraction <= 0.005;
    return m;
}

// Saves an RGBA image as 8-bit grayscale if it is gray and opaque, else as RGBA
inline bool saveGolden(const Saiga::TemplatedImage<Saiga::ucvec4>& image, const std::string& file)
{
    using namespace Saiga;
    bool gray = true;
    for (int y = 0; y < image.h && gray; ++y)
        for (int x = 0; x < image.w && gray; ++x)
        {
            auto p = image(y, x);
            gray   = p(0) == p(1) && p(1) == p(2) && p(3) == 255;
        }
    if (!gray) return image.save(file);
    TemplatedImage<unsigned char> g(image.h, image.w);
    for (int y = 0; y < image.h; ++y)
        for (int x = 0; x < image.w; ++x) g(y, x) = image(y, x)(0);
    return g.save(file);
}

enum class GoldenLoad
{
    Ok,
    Missing,
    Unreadable,
    UnsupportedFormat
};

// Loads a golden as RGBA: 8-bit gray, gray + alpha, RGB or RGBA PNGs (e.g. after a PNG optimizer changed the type)
inline GoldenLoad loadGolden(const std::string& file, Saiga::TemplatedImage<Saiga::ucvec4>& out)
{
    using namespace Saiga;
    Image raw;
    if (!std::filesystem::exists(file)) return GoldenLoad::Missing;
    // Exists but does not load (truncated, a Git LFS pointer, ...): reported separately, it must not be overwritten
    // blindly
    if (!raw.load(file)) return GoldenLoad::Unreadable;
    out.create(raw.h, raw.w);
    switch (raw.type)
    {
        case UC1:
        {
            auto v = raw.getConstImageView<unsigned char>();
            for (int y = 0; y < raw.h; ++y)
                for (int x = 0; x < raw.w; ++x) out(y, x) = ucvec4(v(y, x), v(y, x), v(y, x), 255);
            return GoldenLoad::Ok;
        }
        case UC2:
        {
            auto v = raw.getConstImageView<ucvec2>();
            for (int y = 0; y < raw.h; ++y)
                for (int x = 0; x < raw.w; ++x) out(y, x) = ucvec4(v(y, x)(0), v(y, x)(0), v(y, x)(0), v(y, x)(1));
            return GoldenLoad::Ok;
        }
        case UC3:
        {
            auto v = raw.getConstImageView<ucvec3>();
            for (int y = 0; y < raw.h; ++y)
                for (int x = 0; x < raw.w; ++x) out(y, x) = ucvec4(v(y, x)(0), v(y, x)(1), v(y, x)(2), 255);
            return GoldenLoad::Ok;
        }
        case UC4:
            raw.getConstImageView<ucvec4>().copyTo(out.getImageView());
            return GoldenLoad::Ok;
        default:
            return GoldenLoad::UnsupportedFormat;
    }
}

/**
 * Golden image comparison (docs/engine_design.md §17.6).
 *
 * The golden of `name` is data/tests/golden/<name>.png. The image passes if the mean absolute difference (over all
 * pixels and the RGB channels) is below 1/255 and at most 0.5% of the pixels differ by more than 16/255 in a channel.
 *
 * The comparison gates (adds a gtest failure) only on lavapipe, whose device name contains "llvmpipe"; the goldens are
 * rendered there, other drivers (GPUs) differ slightly and only print the metrics. On a gating failure
 * golden_failed/<name>_actual.png and golden_failed/<name>_diff.png are written to the working directory (build/tests/
 * under ctest; CI uploads that directory as the artifact "golden-failed-<job>"); a passing gating comparison removes
 * stale ones. With SAIGA_UPDATE_GOLDEN=1 the image is written as the new golden instead (only on lavapipe). Clay
 * goldens are gray: they are stored as 8-bit grayscale PNGs when every pixel has R = G = B and alpha 255 (about half
 * the size), else as RGBA.
 *
 * Returns false only on a gating failure (a mismatch, a missing golden or a failed update on lavapipe); on other
 * devices it returns true and only prints.
 */
inline bool compareGolden(const Saiga::TemplatedImage<Saiga::ucvec4>& image, const std::string& name,
                          const std::string& deviceName)
{
    using namespace Saiga;
    namespace fs = std::filesystem;

    const bool gating   = goldenGating(deviceName);
    const fs::path file = fs::path(SAIGA_PROJECT_SOURCE_DIR) / "data/tests/golden" / (name + ".png");

    if (goldenUpdateMode())
    {
        if (!gating)
        {
            ADD_FAILURE() << "SAIGA_UPDATE_GOLDEN=1 is only allowed on lavapipe (device: " << deviceName << ")";
            return false;
        }
        std::error_code ec;
        fs::create_directories(file.parent_path(), ec);
        bool saved = saveGolden(image, file.string());
        EXPECT_TRUE(saved) << "could not write " << file;
        std::cout << "golden " << name << ": written to " << file << std::endl;
        return saved;
    }

    // Failure images go to golden_failed/ in the working directory (CI uploads them). No exceptions here: the caller
    // must still finish its frames in flight (e.g. a read-only working directory).
    const fs::path actualFile = fs::path("golden_failed") / (name + "_actual.png");
    const fs::path diffFile   = fs::path("golden_failed") / (name + "_diff.png");
    // Saved like a golden (gray if possible), so it can be copied into data/tests/golden/ as it is
    auto writeActual = [&]()
    {
        std::error_code ec;
        fs::create_directories("golden_failed", ec);
        if (ec) ADD_FAILURE() << "could not create golden_failed/: " << ec.message();
        if (!saveGolden(image, actualFile.string())) ADD_FAILURE() << "could not write " << actualFile;
        // A diff of an earlier failure would not belong to this image
        fs::remove(diffFile, ec);
    };

    TemplatedImage<ucvec4> golden;
    GoldenLoad loaded = loadGolden(file.string(), golden);
    if (loaded != GoldenLoad::Ok)
    {
        std::string problem = loaded == GoldenLoad::Missing ? "is missing (create it with SAIGA_UPDATE_GOLDEN=1)"
                              : loaded == GoldenLoad::Unreadable
                                  ? "exists but cannot be read (corrupt or a Git LFS pointer?)"
                                  : "has an unsupported pixel format (expected 8-bit gray, gray + alpha, "
                                    "RGB or RGBA)";
        if (!gating)
        {
            std::cout << "golden " << file << " " << problem << " (not gating on this device)" << std::endl;
            return true;
        }
        // Regression: these failures returned before the actual image was written, so CI had nothing to upload
        writeActual();
        ADD_FAILURE() << "golden " << file << " " << problem;
        return false;
    }
    if (golden.w != image.w || golden.h != image.h)
    {
        if (!gating)
        {
            std::cout << "golden " << name << ": size " << image.w << "x" << image.h << ", golden " << golden.w << "x"
                      << golden.h << " (not gating on this device)" << std::endl;
            return true;
        }
        writeActual();
        ADD_FAILURE() << "golden " << name << ": size " << image.w << "x" << image.h << ", golden " << golden.w << "x"
                      << golden.h;
        return false;
    }

    TemplatedImage<ucvec4> diff(image.h, image.w);
    double sum   = 0;
    long large   = 0;
    long pixels  = long(image.w) * image.h;
    int maxDelta = 0;
    for (int y = 0; y < image.h; ++y)
    {
        for (int x = 0; x < image.w; ++x)
        {
            int pixelMax = 0;
            for (int c = 0; c < 3; ++c)
            {
                int d = std::abs(int(image(y, x)(c)) - int(golden(y, x)(c)));
                sum += d;
                pixelMax = std::max(pixelMax, d);
            }
            if (pixelMax > 16) large++;
            maxDelta = std::max(maxDelta, pixelMax);
            // Black where equal, amplified difference otherwise
            unsigned char v = (unsigned char)std::min(255, pixelMax * 8);
            diff(y, x)      = ucvec4(v, v, v, 255);
        }
    }
    double meanAbs       = sum / (double(pixels) * 3);
    double largeFraction = double(large) / double(pixels);
    bool pass            = meanAbs < 1.0 && largeFraction <= 0.005;

    std::cout << "golden " << name << " (" << deviceName << "): mean abs diff " << meanAbs << "/255, "
              << largeFraction * 100 << "% of the pixels > 16/255, max " << maxDelta << "/255"
              << (pass ? "" : " -> differs") << (gating ? "" : " (not gating on this device)") << std::endl;

    // Failure images only where the comparison gates (other devices always differ a little and would litter the working
    // directory)
    std::error_code ec;
    if (pass && gating)
    {
        // Stale images of an earlier failure would look current
        fs::remove(actualFile, ec);
        fs::remove(diffFile, ec);
    }
    if (!pass && gating)
    {
        writeActual();
        if (!diff.save(diffFile.string())) ADD_FAILURE() << "could not write " << diffFile;
        ADD_FAILURE() << "golden " << name << " differs: mean abs diff " << meanAbs << "/255 (limit 1), "
                      << largeFraction * 100 << "% of the pixels > 16/255 (limit 0.5%); see "
                      << fs::absolute("golden_failed", ec).string();
    }
    return pass || !gating;
}

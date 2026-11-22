/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// S36 (no device): the celestial model, the exposure curve, WorldTime / Weather with worldTimeSystem, the environment
// file and the validated writes.

#include "saiga/engine/components/Weather.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/lighting/Celestial.h"
#include "saiga/engine/lighting/EnvironmentExtract.h"
#include "saiga/engine/lighting/EnvironmentFile.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/lighting/Systems.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include "gtest/gtest.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
// The environment without worldTimeSystem: the stored values stay, the extract extrapolates from validAt (as on a
// client between snapshots)
WorldSetup frozenEnvironmentSetup()
{
    WorldSetup s;
    s.environment.enabled = true;
    return s;
}

EnvironmentSample sample(const World& w, double alpha, SpaceId space = 0,
                         const Vulkan::EnvironmentSettings& settings = Vulkan::EnvironmentSettings())
{
    return sampleEnvironment(w, space, alpha, settings, w.environmentSetup());
}

WorldSetup environmentSetup()
{
    WorldSetup s;
    s.environment.enabled = true;
    s.systems             = addEnvironmentSystems;
    return s;
}

std::string writeTemp(const std::string& name, const std::string& content)
{
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p);
    f << content;
    return p.string();
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// The celestial model

TEST(Celestial, NoonEquinoxAtLatitude)
{
    CelestialConfig c;
    for (double lat : {0.0, 30.0, 45.0, 60.0})
    {
        c.latitudeDeg          = lat;
        const CelestialState s = celestialState(c, 12, 0);
        EXPECT_NEAR(s.sunElevationDeg, 90 - lat, 0.1) << lat;
        if (lat > 0)
        {
            // South (+z)
            EXPECT_NEAR(s.sunAzimuthDeg, 180, 0.1) << lat;
            EXPECT_GT(s.sunDir(2), 0);
        }
        EXPECT_NEAR(s.sunDir.norm(), 1, 1e-12);
    }
}

TEST(Celestial, SunriseSunsetAtEquinox)
{
    CelestialConfig c;
    // The declination moves by the time of the day: 0.5 degrees of tolerance
    EXPECT_NEAR(celestialState(c, 6, 0).sunElevationDeg, 0, 0.5);
    EXPECT_NEAR(celestialState(c, 18, 0).sunElevationDeg, 0, 0.5);
    // Rises in the east (+x), sets in the west
    EXPECT_GT(celestialState(c, 6.5, 0).sunDir(0), 0.5);
    EXPECT_LT(celestialState(c, 17.5, 0).sunDir(0), -0.5);
    EXPECT_LT(celestialState(c, 0, 0).sunElevationDeg, -40);
}

TEST(Celestial, SeasonsMoveTheSun)
{
    CelestialConfig c;
    c.latitudeDeg = 45;
    // Summer solstice: a quarter year after the equinox; the noon elevation is 90 - lat + tilt
    const int64_t summer = int64_t(std::lround(c.yearLengthDays / 4));
    const int64_t winter = int64_t(std::lround(3 * c.yearLengthDays / 4));
    EXPECT_NEAR(celestialState(c, 12, summer).sunElevationDeg, 45 + c.axialTiltDeg, 0.1);
    EXPECT_NEAR(celestialState(c, 12, winter).sunElevationDeg, 45 - c.axialTiltDeg, 0.1);
}

TEST(Celestial, MoonPhaseLitFraction)
{
    CelestialConfig c;
    c.moonPhase = 0.5;
    EXPECT_NEAR(celestialState(c, 0, 0).moonIlluminance, c.moonIlluminance, 1e-12);
    c.moonPhase = 0;
    EXPECT_NEAR(celestialState(c, 0, 0).moonIlluminance, 0, 1e-12);
    c.moonPhase = 0.25;
    EXPECT_NEAR(celestialState(c, 0, 0).moonIlluminance, 0.5 * c.moonIlluminance, 1e-12);
}

TEST(Celestial, StarsRotateWithHourAngle)
{
    CelestialConfig c;
    const double lat = c.latitudeDeg * 3.14159265358979323846 / 180;
    const Vec3 pole(0, std::sin(lat), -std::cos(lat));
    const Mat3 r0 = starRotation(c, 0);
    const Mat3 r6 = starRotation(c, 3.14159265358979323846 / 2);
    // The pole stays, other stars move; rotations are orthonormal
    EXPECT_NEAR((r6 * pole - pole).norm(), 0, 1e-12);
    EXPECT_NEAR((r0 * Vec3(1, 0, 0) - Vec3(1, 0, 0)).norm(), 0, 1e-12);
    EXPECT_GT((r6 * Vec3(1, 0, 0) - Vec3(1, 0, 0)).norm(), 0.5);
    const Mat3 i = r6 * r6.transpose();
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) EXPECT_NEAR(i(a, b), a == b ? 1 : 0, 1e-12);
}

TEST(Celestial, FullMoonRisesAtSunset)
{
    CelestialConfig c;
    c.moonPhase = 0.5;
    for (double t : {6.0, 18.0})
    {
        const CelestialState s = celestialState(c, t, 0);
        EXPECT_NEAR(s.moonElevationDeg, s.sunElevationDeg, 1.0) << t;
        EXPECT_NEAR(s.moonElevationDeg, 0, 1.0) << t;
    }
    // At midnight the full moon is high, opposite the noon sun
    EXPECT_GT(celestialState(c, 0, 0).moonElevationDeg, 40);
    EXPECT_EQ(selectShadowLight(celestialState(c, 0, 0)), ShadowLight::Moon);
    EXPECT_EQ(selectShadowLight(celestialState(c, 12, 0)), ShadowLight::Sun);
}

TEST(Celestial, NewMoonNightHasNoShadowLight)
{
    CelestialConfig c;
    c.moonPhase            = 0;
    const CelestialState s = celestialState(c, 0, 0);
    EXPECT_EQ(s.moonIlluminance, 0);
    EXPECT_EQ(selectShadowLight(s), ShadowLight::None);
}

TEST(Celestial, ShadowLightHysteresis)
{
    CelestialState s;
    s.sunIlluminance   = 1;
    s.moonIlluminance  = 1.5;
    s.sunElevationDeg  = 30;
    s.moonElevationDeg = 30;
    EXPECT_EQ(selectShadowLight(s, ShadowLight::Sun), ShadowLight::Sun);
    EXPECT_EQ(selectShadowLight(s, ShadowLight::None), ShadowLight::Moon);
    s.moonIlluminance = 2.5;
    EXPECT_EQ(selectShadowLight(s, ShadowLight::Sun), ShadowLight::Moon);
}

TEST(Celestial, AnglesRoundTrip)
{
    for (double az : {0.0, 45.0, 90.0, 180.0, 300.0})
        for (double el : {-30.0, 0.0, 20.0, 80.0})
        {
            const Vec3 d = directionFromAngles(az, el);
            EXPECT_NEAR(elevationDeg(d), el, 1e-9);
            EXPECT_NEAR(azimuthDeg(d), az, 1e-9);
        }
    EXPECT_NEAR((directionFromAngles(90, 0) - Vec3(1, 0, 0)).norm(), 0, 1e-12);
    EXPECT_NEAR((directionFromAngles(0, 0) - Vec3(0, 0, -1)).norm(), 0, 1e-12);
}

// ---------------------------------------------------------------------------------------------------------------------
// Exposure

TEST(Exposure, CurveInterpolatesAndClamps)
{
    Vulkan::ExposureSettings e;
    EXPECT_FLOAT_EQ(e.ev100At(-90), -1);
    EXPECT_FLOAT_EQ(e.ev100At(-18), -1);
    EXPECT_FLOAT_EQ(e.ev100At(-12), 1);
    EXPECT_FLOAT_EQ(e.ev100At(0), 9);
    EXPECT_FLOAT_EQ(e.ev100At(5), 11);
    EXPECT_FLOAT_EQ(e.ev100At(90), 15);
    EXPECT_FLOAT_EQ(e.ev100At(120), 15);
    e.curve.clear();
    EXPECT_FLOAT_EQ(e.ev100At(10), 15);
}

TEST(Exposure, FixedMode)
{
    Vulkan::ExposureSettings e;
    e.mode  = Vulkan::ExposureMode::Fixed;
    e.ev100 = 3;
    EXPECT_FLOAT_EQ(e.ev100At(-50), 3);
    EXPECT_FLOAT_EQ(e.ev100At(60), 3);
}

TEST(Exposure, CompensationIsLog2OfMultiplier)
{
    // --exposure 2 (the multiplier) = +1 EV of compensation
    Vulkan::ExposureSettings e;
    e.compensation = 1;
    EXPECT_FLOAT_EQ(e.ev100At(90), 14);
    EXPECT_FLOAT_EQ(Vulkan::exposureFromEv100(e.ev100At(90), 1), Vulkan::exposureFromEv100(15, 2));
    EXPECT_NEAR(Vulkan::exposureFromEv100(15), 1 / (1.2 * 32768), 1e-12);
    // The clamp
    EXPECT_FLOAT_EQ(Vulkan::exposureFromEv100(100), std::exp2(-24.0f));
    EXPECT_FLOAT_EQ(Vulkan::exposureFromEv100(-100), std::exp2(10.0f));
}

// ---------------------------------------------------------------------------------------------------------------------
// WorldTime, Weather and worldTimeSystem

TEST(WorldTime, SystemAdvancesWrapsAndCountsDays)
{
    World w(environmentSetup());
    const double dt = w.dt();
    {
        std::string error;
        ASSERT_TRUE(setWorldTime(w, WorldTimeChange{23.5, 4, 3600 * 0.5 / (30 * dt)}, error)) << error;
    }
    // 0.5 game hours in 30 steps: wraps into the next day
    w.step(60);
    EXPECT_NEAR(w.readGlobal<WorldTime>().timeOfDay, 0.5, 1e-9);
    EXPECT_EQ(w.readGlobal<WorldTime>().day, 5);
    EXPECT_NEAR(w.readGlobal<WorldTime>().validAt, w.time(), 1e-12);
    // Backwards
    {
        std::string error;
        ASSERT_TRUE(setWorldTime(w, WorldTimeChange{std::nullopt, std::nullopt, -3600 * 1.0 / (60 * dt)}, error));
    }
    w.step(60);
    EXPECT_NEAR(w.readGlobal<WorldTime>().timeOfDay, 23.5, 1e-9);
    EXPECT_EQ(w.readGlobal<WorldTime>().day, 4);
    // rate 0: frozen
    {
        std::string error;
        ASSERT_TRUE(setWorldTime(w, WorldTimeChange{10.0, std::nullopt, 0.0}, error));
    }
    w.step(100);
    EXPECT_EQ(w.readGlobal<WorldTime>().timeOfDay, 10);
}

TEST(WorldTime, SystemNormalisesInvalidValues)
{
    World w(environmentSetup());
    // As a raw set_global could write them (Visit has no limits)
    w.global<WorldTime>().timeOfDay     = 30;
    w.global<WorldTime>().rate          = 0;
    w.spaceState<Weather>(0).cloudiness = 2;
    w.spaceState<Weather>(0).fogDensity = std::numeric_limits<double>::quiet_NaN();
    w.spaceState<Weather>(0).wind       = Vec2(std::numeric_limits<double>::infinity(), 2);
    w.step();
    EXPECT_NEAR(w.readGlobal<WorldTime>().timeOfDay, 6, 1e-12);
    EXPECT_EQ(w.readGlobal<WorldTime>().day, 1);
    EXPECT_EQ(w.readSpaceState<Weather>(0).cloudiness, 1);
    EXPECT_EQ(w.readSpaceState<Weather>(0).fogDensity, Weather().fogDensity);
    EXPECT_EQ(w.readSpaceState<Weather>(0).wind(0), Weather().wind(0));
    EXPECT_EQ(w.readSpaceState<Weather>(0).wind(1), 2);
    w.global<WorldTime>().timeOfDay = std::numeric_limits<double>::quiet_NaN();
    w.global<WorldTime>().rate      = 1e9;
    w.step();
    EXPECT_TRUE(std::isfinite(w.readGlobal<WorldTime>().timeOfDay));
    EXPECT_EQ(w.readGlobal<WorldTime>().rate, kMaxTimeRate);
}

TEST(WorldTime, CloudOffsetWrapsSeamless)
{
    WorldSetup s                 = environmentSetup();
    s.environment.cloudWindScale = 2;
    World w(s);
    {
        std::string error;
        WeatherChange c;
        c.wind        = Vec2(150, -90);
        c.cloudOffset = Vec2(16000, 100);
        ASSERT_TRUE(setWeather(w, 0, c, error)) << error;
    }
    const int n = 120;
    w.step(n);
    const Weather& weather = w.readSpaceState<Weather>(0);
    const double t         = n * w.dt() * 2;
    for (int i = 0; i < 2; ++i)
    {
        const double unwrapped = (i == 0 ? 16000 : 100) + (i == 0 ? 150 : -90) * t;
        EXPECT_GE(weather.cloudOffset(i), 0);
        EXPECT_LT(weather.cloudOffset(i), kCloudTile);
        // The wrapped offset is the unwrapped one modulo the tile (what the noise of period kCloudTile sees)
        EXPECT_NEAR(wrapPeriod(unwrapped, kCloudTile), weather.cloudOffset(i), 1e-6);
    }
}

// D4: a plain world registers both types (the engine IDs stay a prefix) but has no world-state entity for them
TEST(WorldTime, RegisteredButNoEntityWithoutEnvironment)
{
    World plain{WorldSetup{}};
    EXPECT_TRUE(plain.registry().registered(WorldTime::id));
    EXPECT_TRUE(plain.registry().registered(Weather::id));
    EXPECT_FALSE(plain.registry().isGlobal(WorldTime::id));
    EXPECT_FALSE(plain.registry().isSpaceState(Weather::id));
    EXPECT_FALSE(plain.worldStateEntity().valid());
    EXPECT_FALSE(plain.spaceStateEntity(0).valid());
    EXPECT_EQ(plain.registry().aliveCount(), 0u);
    EXPECT_FALSE(hasWorldTime(plain));
    std::string error;
    EXPECT_FALSE(setWorldTime(plain, WorldTimeChange{6.0, std::nullopt, std::nullopt}, error));
    EXPECT_NE(error.find("EnvironmentSetup::enabled"), std::string::npos) << error;
    // The registration validates with the environment (both kinds)
    World env(environmentSetup());
    EXPECT_TRUE(env.registry().isGlobal(WorldTime::id));
    EXPECT_TRUE(env.registry().isSpaceState(Weather::id));
    EXPECT_TRUE(env.worldStateEntity().valid());
    EXPECT_TRUE(env.spaceStateEntity(0).valid());
    // The engine systems step the plain world unchanged: the same hash as a world without the environment system
    WorldSetup withSystem = WorldSetup{};
    withSystem.systems    = addEnvironmentSystems;
    World a{WorldSetup{}}, b(withSystem);
    a.step(10);
    b.step(10);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(WorldTime, SaveLoadRoundTrip)
{
    World a(environmentSetup());
    {
        std::string error;
        ASSERT_TRUE(setWorldTime(a, WorldTimeChange{17.25, 3, 120.0}, error));
        WeatherChange c;
        c.cloudiness = 0.8;
        c.rain       = 0.4;
        c.wind       = Vec2(-2, 7);
        ASSERT_TRUE(setWeather(a, 0, c, error));
    }
    a.step(7);
    const std::vector<uint8_t> save = a.save();
    World b(environmentSetup());
    const LoadResult r = b.tryLoad(save);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.readGlobal<WorldTime>().timeOfDay, a.readGlobal<WorldTime>().timeOfDay);
    EXPECT_EQ(b.readGlobal<WorldTime>().day, a.readGlobal<WorldTime>().day);
    EXPECT_EQ(b.readGlobal<WorldTime>().rate, 120);
    EXPECT_EQ(b.readGlobal<WorldTime>().validAt, a.readGlobal<WorldTime>().validAt);
    EXPECT_EQ(b.readSpaceState<Weather>(0).cloudiness, 0.8);
    EXPECT_EQ(b.readSpaceState<Weather>(0).cloudOffset, a.readSpaceState<Weather>(0).cloudOffset);
    a.step(5);
    b.step(5);
    EXPECT_EQ(a.hash(), b.hash());
    // A plain save loads into a world with the environment: the state entities get their defaults
    World plain{WorldSetup{}};
    plain.step(2);
    World c(environmentSetup());
    const LoadResult rc = c.tryLoad(plain.save());
    ASSERT_TRUE(rc.ok()) << rc.error;
    EXPECT_EQ(c.readGlobal<WorldTime>().timeOfDay, WorldTime().timeOfDay);
    EXPECT_EQ(c.readSpaceState<Weather>(0).cloudiness, Weather().cloudiness);
}

// ---------------------------------------------------------------------------------------------------------------------
// The validated writes

TEST(EnvironmentWrites, RangeErrorsNameTheField)
{
    World w(environmentSetup());
    std::string error;
    const WorldTime before = w.readGlobal<WorldTime>();
    EXPECT_FALSE(setWorldTime(w, WorldTimeChange{24.0, std::nullopt, std::nullopt}, error));
    EXPECT_NE(error.find("hours"), std::string::npos) << error;
    EXPECT_FALSE(setWorldTime(w, WorldTimeChange{-1.0, std::nullopt, std::nullopt}, error));
    EXPECT_FALSE(setWorldTime(w, WorldTimeChange{std::numeric_limits<double>::quiet_NaN(), std::nullopt, 0.0}, error));
    EXPECT_FALSE(setWorldTime(w, WorldTimeChange{6.0, std::nullopt, 4000.0}, error));
    EXPECT_NE(error.find("rate"), std::string::npos) << error;
    // Nothing changed (also not the valid hours of a rejected call)
    EXPECT_EQ(w.readGlobal<WorldTime>().timeOfDay, before.timeOfDay);
    EXPECT_EQ(w.readGlobal<WorldTime>().rate, before.rate);

    auto weather = [&](auto set, const char* field)
    {
        WeatherChange c;
        set(c);
        std::string e;
        EXPECT_FALSE(setWeather(w, 0, c, e)) << field;
        EXPECT_NE(e.find(field), std::string::npos) << e;
    };
    weather([](WeatherChange& c) { c.cloudiness = 1.5; }, "cloudiness");
    weather([](WeatherChange& c) { c.rain = -0.1; }, "rain");
    weather([](WeatherChange& c) { c.fogDensity = 0.2; }, "fog_density");
    weather([](WeatherChange& c) { c.snowCover = std::numeric_limits<double>::infinity(); }, "snow_cover");
    weather([](WeatherChange& c) { c.wind = Vec2(0, 1000); }, "wind[1]");
    weather([](WeatherChange& c) { c.cloudOffset = Vec2(std::numeric_limits<double>::quiet_NaN(), 0); },
            "cloud_offset[0]");
    {
        WeatherChange c;
        c.cloudiness = 0.5;
        std::string e;
        EXPECT_FALSE(setWeather(w, 7, c, e));
        EXPECT_NE(e.find("space"), std::string::npos) << e;
    }
    // previous holds the values before the write
    WorldTime previous;
    ASSERT_TRUE(setWorldTime(w, WorldTimeChange{6.0, 2, 1.0}, error, &previous)) << error;
    EXPECT_EQ(previous.timeOfDay, before.timeOfDay);
    EXPECT_EQ(w.readGlobal<WorldTime>().timeOfDay, 6);
    EXPECT_EQ(w.readGlobal<WorldTime>().day, 2);
}

// ---------------------------------------------------------------------------------------------------------------------
// The environment file (R-DATA-1)

TEST(EnvironmentFile, MissingKeysKeepDefaults)
{
    const std::string path  = writeTemp("saiga_environment_partial.json", R"({
        "celestial": {"latitudeDeg": 50},
        "exposure": {"mode": "fixed", "ev100": 12, "curve": [[0, 5], [90, 10]]},
        "clouds": {"altitude": 1800, "windScale": 0.5},
        "atmosphere": {"starIntensity": 2},
        "lighting": {"environmentResolution": 64}
    })");
    const EnvironmentFile f = readEnvironmentFile(path);
    EXPECT_EQ(f.setup.celestial.latitudeDeg, 50);
    EXPECT_EQ(f.setup.celestial.axialTiltDeg, CelestialConfig().axialTiltDeg);
    EXPECT_FALSE(f.setup.enabled);
    EXPECT_EQ(f.setup.cloudWindScale, 0.5);
    EXPECT_EQ(f.settings.exposure.mode, Vulkan::ExposureMode::Fixed);
    EXPECT_EQ(f.settings.exposure.ev100, 12);
    ASSERT_EQ(f.settings.exposure.curve.size(), 2u);
    EXPECT_EQ(f.settings.exposure.curve[1](1), 10);
    EXPECT_EQ(f.settings.clouds.altitude, 1800);
    EXPECT_EQ(f.settings.clouds.radius, Vulkan::CloudSettings().radius);
    EXPECT_EQ(f.settings.starIntensity, 2);
    EXPECT_EQ(f.settings.ambient.scale, 1);
    EXPECT_EQ(f.lighting.environmentResolution, 64);
    EXPECT_EQ(f.lighting.clouds, true);
    // An empty object: all defaults
    const EnvironmentFile e = readEnvironmentFile(writeTemp("saiga_environment_empty.json", "{}"));
    EXPECT_EQ(e.settings.exposure.ev100, 15);
    EXPECT_EQ(e.setup.celestial.latitudeDeg, 45);
}

TEST(EnvironmentFile, ErrorsNameTheKey)
{
    EnvironmentFile f;
    std::string error;
    EXPECT_FALSE(
        tryReadEnvironmentFile(writeTemp("saiga_environment_bad.json", R"({"clouds": {"altitud": 1}})"), f, error));
    EXPECT_NE(error.find("altitud"), std::string::npos) << error;
    EXPECT_FALSE(tryReadEnvironmentFile(writeTemp("saiga_environment_bad2.json", R"({"exposure": {"mode": "auto"}})"),
                                        f, error));
    EXPECT_NE(error.find("mode"), std::string::npos) << error;
    EXPECT_FALSE(tryReadEnvironmentFile(writeTemp("saiga_environment_bad3.json", R"({"weather": {}})"), f, error));
    EXPECT_NE(error.find("weather"), std::string::npos) << error;
    EXPECT_FALSE(tryReadEnvironmentFile("/nonexistent/saiga_environment.json", f, error));
}

TEST(EnvironmentFileDeathTest, UnknownKeyIsFatal)
{
    const std::string path = writeTemp("saiga_environment_unknown.json", R"({"celestial": {"latitude": 50}})");
    EXPECT_DEATH(readEnvironmentFile(path), "latitude");
}

// ---------------------------------------------------------------------------------------------------------------------
// The environment extract

TEST(EnvironmentExtract, InterpolatesBetweenSteps)
{
    World w(environmentSetup());
    std::string error;
    ASSERT_TRUE(setWorldTime(w, WorldTimeChange{10.0, std::nullopt, 3600.0}, error));
    w.step(3);
    const double current  = w.readGlobal<WorldTime>().timeOfDay;
    const double previous = current - 3600 * w.dt() / 3600;
    EXPECT_NEAR(sample(w, 1).time.timeOfDay, current, 1e-9);
    EXPECT_NEAR(sample(w, 0).time.timeOfDay, previous, 1e-9);
    EXPECT_NEAR(sample(w, 0.5).time.timeOfDay, 0.5 * (previous + current), 1e-9);
    // The sun follows the time; the state is the float of the sample
    const EnvironmentSample a = sample(w, 0.25);
    EXPECT_NEAR((a.celestial.sunDir - celestialState(CelestialConfig(), a.time.timeOfDay, 0).sunDir).norm(), 0, 1e-12);
    EXPECT_NEAR(a.state.sunDir(1), float(a.celestial.sunDir(1)), 1e-6);
    // The wind moves the clouds between the steps as well
    const Weather& weather = w.readSpaceState<Weather>(0);
    EXPECT_NEAR(sample(w, 0).weather.cloudOffset(0), weather.cloudOffset(0) - weather.wind(0) * w.dt(), 1e-9);
}

TEST(EnvironmentExtract, ClientExtrapolatesWithHostClock)
{
    // No system writes the time (as on a client between two snapshots): the render time runs ahead of validAt
    World w(frozenEnvironmentSetup());
    std::string error;
    ASSERT_TRUE(setWorldTime(w, WorldTimeChange{10.0, std::nullopt, 1800.0}, error));
    const double validAt = w.readGlobal<WorldTime>().validAt;
    w.step(30);
    EXPECT_EQ(w.readGlobal<WorldTime>().timeOfDay, 10);
    const double tr = w.time() + (0.5 - 1) * w.dt();
    EXPECT_NEAR(sample(w, 0.5).time.timeOfDay, 10 + 1800 * (tr - validAt) / 3600, 1e-9);
    EXPECT_NEAR(sample(w, 0.5).renderTime, tr, 1e-12);
    // Monotonic with the render time
    double last = 0;
    for (double alpha : {0.0, 0.25, 0.5, 0.75, 0.999})
    {
        const double t = sample(w, alpha).time.timeOfDay;
        EXPECT_GT(t, last);
        last = t;
    }
}

TEST(EnvironmentExtract, RateChangeDoesNotJump)
{
    World w(environmentSetup());
    std::string error;
    ASSERT_TRUE(setWorldTime(w, WorldTimeChange{10.0, std::nullopt, 1800.0}, error));
    w.step(10);
    const double before = sample(w, 1).time.timeOfDay;
    // A new rate between steps: the time continues from the same value (validAt is the write's time)
    ASSERT_TRUE(setWorldTime(w, WorldTimeChange{std::nullopt, std::nullopt, 3600.0}, error));
    EXPECT_NEAR(sample(w, 1).time.timeOfDay, before, 1e-12);
    w.step();
    EXPECT_NEAR(sample(w, 1).time.timeOfDay, before + 3600 * w.dt() / 3600, 1e-9);
    EXPECT_NEAR(sample(w, 0).time.timeOfDay, before, 1e-9);
}

TEST(EnvironmentExtract, SanitisesOutOfRangeWeather)
{
    World w(frozenEnvironmentSetup());
    w.spaceState<Weather>(0).cloudiness = 5;
    w.spaceState<Weather>(0).rain       = -1;
    w.spaceState<Weather>(0).fogDensity = std::numeric_limits<double>::quiet_NaN();
    w.global<WorldTime>().timeOfDay     = 36;
    const EnvironmentSample s           = sample(w, 1);
    EXPECT_EQ(s.weather.cloudiness, 1);
    EXPECT_EQ(s.state.cloudiness, 1.0f);
    EXPECT_EQ(s.state.rain, 0.0f);
    EXPECT_FLOAT_EQ(s.state.fogDensity, float(Weather().fogDensity));
    EXPECT_NEAR(s.time.timeOfDay, 12, 1e-12);
    EXPECT_EQ(s.time.day, 1);
}

TEST(EnvironmentExtract, EnclosedSpaceUsesTemplateAmbientAndEv100)
{
    for (bool enabled : {true, false})
    {
        WorldSetup s            = enabled ? frozenEnvironmentSetup() : WorldSetup{};
        SpaceTemplate cellar    = SpaceTemplate::enclosed("cellar");
        cellar.ambient          = Vec3(10, 20, 30);
        cellar.ev100            = 4;
        s.config.spaceTemplates = {SpaceTemplate(), cellar};
        World w(s);
        Vulkan::EnvironmentSettings settings;
        settings.exposure.compensation = 1;
        const EnvironmentSample in     = sample(w, 1, 1, settings);
        EXPECT_TRUE(in.enclosed);
        EXPECT_TRUE(in.state.enclosed);
        EXPECT_EQ(in.state.enclosedAmbient, vec3(10, 20, 30));
        EXPECT_EQ(in.state.ev100, 3.0f);
        EXPECT_EQ(in.state.exposureSource, Vulkan::ExposureSource::Enclosed);
        EXPECT_EQ(in.state.sunIlluminance, 0.0f);
        EXPECT_EQ(in.state.moonIlluminance, 0.0f);
        EXPECT_EQ(in.shadowLight, ShadowLight::None);
        const EnvironmentSample out = sample(w, 1, 0, settings);
        EXPECT_FALSE(out.state.enclosed);
        EXPECT_EQ(out.state.exposureSource, Vulkan::ExposureSource::Curve);
        EXPECT_GT(out.state.sunIlluminance, 0.0f);
    }
}

TEST(EnvironmentExtract, DefaultsWithoutEnvironmentAndSunOverride)
{
    World plain{WorldSetup{}};
    const EnvironmentSample s = sample(plain, 1);
    EXPECT_FALSE(s.hasTime);
    EXPECT_EQ(s.time.timeOfDay, 12);
    EXPECT_EQ(s.weather.cloudiness, 0.3);
    EXPECT_NEAR(s.celestial.sunElevationDeg, 45, 0.1);
    EXPECT_FLOAT_EQ(s.state.ev100, Vulkan::ExposureSettings().curveEv100(float(s.celestial.sunElevationDeg)));
    Vulkan::EnvironmentSettings settings;
    settings.sunOverride      = Vulkan::SunOverride{90, 10};
    const EnvironmentSample o = sample(plain, 1, 0, settings);
    EXPECT_NEAR(o.celestial.sunElevationDeg, 10, 1e-9);
    EXPECT_NEAR(o.celestial.sunAzimuthDeg, 90, 1e-9);
    EXPECT_GT(o.state.sunDir(0), 0.9f);
    EXPECT_FLOAT_EQ(o.state.ev100, Vulkan::ExposureSettings().curveEv100(10));
    // The phases are in [0, 2 pi)
    for (float ph : o.state.phases)
    {
        EXPECT_GE(ph, 0);
        EXPECT_LT(ph, 6.2832f);
    }
}

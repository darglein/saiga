# S05: Math additions

Status: done.

This plan is the template for all stage plans (`docs/stages/Sxx_<name>.plan.md`); the sections and when they are
written are listed in `docs/engine_design.md` §22.

## Goal

Engine and game code can do all double math it needs with tiny-eigen (the default and CI backend) through the API
subset of `docs/engine_design.md` §7.10 (R-MATH-2), and the same code compiles with Eigen.

## Depends on

Nothing (§22).

## Rules that apply

- R-MATH-1/2 (double simulation math, tiny-eigen subset that also compiles with Eigen).
- R-TEST-1 (a test per addition), R-DOC-1/2 (signatures from §7.10; design changes update the document).
- AGENTS.md: tiny-eigen is maintained in this repository (`tests/test_core_tiny_eigen.cpp`) and keeps its own
  formatting; new lines follow the house style.

## Scope

tiny-eigen (`submodules/tiny-eigen/include/tiny-eigen/`, maintained in this repository):

- `matrix.h`, `MatrixBase`: `SameMatrix& setIdentity()`, `bool isApprox(const MatrixBase<O>&, Scalar prec)` (Eigen
  semantics, default precision 1e-12 double / 1e-5 float), `cwiseProduct`, `cwiseMin`, `cwiseMax`. All four require the
  same scalar type and dimensions (`static_assert`, like Eigen).
- `matrix.h`, `Matrix` / `MatrixView` / `Array`: `RowsAtCompileTime`, `ColsAtCompileTime` (Eigen's names).
- `matrix.h`, `Matrix`: `static Unit(i)` (bounds asserted), `UnitX/Y/Z/W()` (column vectors),
  `operator*=(const Matrix<S, C, C>&)` (aliasing safe; the scalar `*=` of the base stays visible through a
  using-declaration).
- `quaternion.h`: `Quaternion::operator*=` (this * other), `angularDistance`, `isApprox`,
  `explicit Quaternion(const AngleAxis&)` ((cos(a/2), sin(a/2) * axis), axis used as given like Eigen); new class
  `AngleAxis<Scalar>` (`angle()`, `axis()`, `toRotationMatrix()`, `matrix()`, `explicit AngleAxis(const Quaternion&)`,
  only the exact identity has no axis, like Eigen).

saiga core math (`src/saiga/core/math/`):

- `Quaternion.h`: `Saiga::AngleAxis` = `Eigen::AngleAxisd` or `TinyEigen::AngleAxis<double>`.
- `EigenGLMInterface.h`: header-only double overloads, templates restricted to double scalars, argument shapes checked
  with `static_assert`: `Mat4 createTRSmatrix(Vec3/4 t, const Quat& r, Vec3/4 s)`,
  `Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up)` (asserts eye != center and up not parallel to the view direction),
  `Quat make_quat(Mat3 or Mat4)`, `Quat quat_cast(Mat3 or Mat4)` (normalized).

## Out of scope

- Scalar `make_quat(double x, y, z, w)`: it would make the existing calls with integer literals ambiguous; use
  `Quat(w, x, y, z)`.
- Double versions of `perspective`/`ortho` (projection is render code and stays float, R-MATH-1).
- A CI job with real Eigen (best effort, §7.10); the Eigen build is run locally (Verification).

## Design notes

- The double helpers are templates with `enable_if` on `Scalar == double`: with Eigen, a non-template `lookAt(Vec3...)`
  next to `lookAt(vec3...)` would be ambiguous for expression arguments (`eye + dir`), because Eigen matrices are
  constructible from any expression.
- With Eigen, the existing float overloads (`vec3`/`vec4`, `mat3`/`mat4`) are already ambiguous for expression
  arguments; tests pass concrete types.
- The default `isApprox` precision is a tiny-eigen internal (`TinyEigen::internal::dummyPrecision<Scalar>()`), not
  public API, because Eigen spells it differently (`NumTraits<Scalar>::dummy_precision()`).

## Tests

- `tests/test_core_tiny_eigen.cpp` (tiny-eigen builds): `UnitVectors`, `SetIdentity` (incl. 2x3), `IsApprox` (incl. a
  failing float case and `long double`), `Cwise`,
  `MatrixProductAssignment` (incl. `m *= m`), `QuaternionProductAssignmentAndAngularDistance` (incl. the order with
  non-commuting rotations), `AngleAxis` (incl. angles above pi, a non-unit axis, a 2e-13 rad rotation).
- `tests/test_core_math_functions.cpp` (both backends): `MathDouble.SubsetMatchesBothBackends` (incl. `*=` order,
  aliasing, angles above pi), `CreateTRSmatrixMatchesFloat`, `LookAtMatchesFloatAndIsExactFarFromTheOrigin`,
  `MakeQuatAndQuatCast` (incl. 180 degree rotations), `CreateTRSmatrixMatchesFloat` also with `Vec4` arguments;
  `MathDoubleDeathTest.LookAtRejectsDegenerateInput` (with `SAIGA_ASSERTS`).

## Done when

- Every addition has a test; all tests pass in the CI configuration.
- `tests/test_core_math_functions.cpp` builds and passes with Eigen (Verification).
- `docs/engine_design.md` R-MATH-2, §3, §7.10 and the S05 row are updated.

## Verification

- CI configuration (tiny-eigen): configure and build with the CI recipe of AGENTS.md ("Build (Linux)"), then
  `cd build && ctest --output-on-failure --parallel 4`.
- Eigen: the "Eigen backend check" of AGENTS.md (configure `build-eigen`, build and run `test_core_math_functions`).

## Design changes

- `RowsAtCompileTime`/`ColsAtCompileTime` added to tiny-eigen (not in the original §7.10 list): needed for the shape
  checks of the double helpers with both backends. §7.10 updated.
- `Quaternion(AngleAxis)` does not normalize the axis (tiny-eigen's older `FromAngleAxis` does) and
  `AngleAxis::toRotationMatrix()` uses Rodrigues' formula: Eigen semantics, so both backends agree also for a non-unit
  axis. `AngleAxis(q)` rescales a tiny vector part before the norm like Eigen's `stableNorm`. §7.10 updated.
- The double helpers accept exactly 3D or 4D vectors and 3x3 or 4x4 matrices (`static_assert`); `lookAt` asserts on
  degenerate input. §7.10 updated.
- The default `isApprox` precision follows Eigen per type (float 1e-5, double 1e-12, long double 1e-15) and is internal
  to tiny-eigen (`internal::dummyPrecision`).

## Results

- CI configuration (GCC 15, tiny-eigen): 39/39 tests pass; `test_core_tiny_eigen` 28 tests, `test_core_math_functions`
  27 tests (incl. the `lookAt` death test).
- Eigen configuration (GCC 15, Makefiles, AGENTS.md "Eigen backend check"): `test_core_math_functions` passes (26 tests;
  the tiny-eigen-only test is not built). Of the core tests with Eigen, `test_core_vectorization` and
  `test_core_geometry` fail; they fail the same way on `master` (not caused by S05).

## Files

`submodules/tiny-eigen/include/tiny-eigen/{matrix.h,quaternion.h}`, `src/saiga/core/math/{Quaternion.h,
EigenGLMInterface.h}`, `tests/test_core_tiny_eigen.cpp`, `tests/test_core_math_functions.cpp`,
`docs/engine_design.md`, `AGENTS.md`, this file.

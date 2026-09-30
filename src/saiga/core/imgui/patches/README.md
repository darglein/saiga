# Local Dear ImGui patches

The Dear ImGui sources in `src/saiga/core/imgui/` and the backends in `src/saiga/opengl/imgui/internal/` carry a few
local changes on top of upstream. An ImGui update replaces these files with the upstream versions, which silently drops
the changes. Every patched line is marked with a `SAIGA PATCH:` comment, and each patch is kept here as a diff against
the pristine upstream files.

| Patch | Change |
|-------|--------|
| `0001-crosshair-cursor.patch` | Adds `ImGuiMouseCursor_Crosshair` and maps it to `GLFW_CROSSHAIR_CURSOR` in the GLFW backend. |
| `0002-hand-cursor-on-clickable-widgets.patch` | Shows `ImGuiMouseCursor_Hand` while hovering buttons, image buttons, checkboxes, radio buttons, combos, tree nodes, collapsing headers, selectables, menu items and tabs. `InvisibleButton` is left unchanged because it is typically used for canvases. |
| `0003-hand-cursor-on-sliders.patch` | Shows `ImGuiMouseCursor_Hand` while hovering or dragging a slider (`SliderScalar`, which the `SliderFloat`/`SliderInt`/`SliderAngle` variants use). Drags keep the arrow. |

## Updating ImGui

Run all commands from the saiga root.

1. Copy the new upstream files over the vendored ones.
2. Reapply the patches:

   ```bash
   git apply --3way src/saiga/core/imgui/patches/*.patch
   ```

   If a patch no longer applies because upstream changed the surrounding code, add the marked lines by hand. Then
   regenerate that patch file as a diff of the pristine upstream file against the patched one, with `a/` and `b/`
   path prefixes relative to the saiga root.
3. Check that every patch is present. This succeeds only when all patches are applied:

   ```bash
   git apply --reverse --check src/saiga/core/imgui/patches/*.patch
   ```

4. `grep -rn "SAIGA PATCH" src/saiga` lists every patched line for review.
5. Build and run `test_core_imgui_cursors`. It checks the cursors these patches set, and the CI runs it on every pull
   request.

## Adding a patch

Mark every changed line with a `// SAIGA PATCH: <short description>` comment, add the diff here as the next numbered
`.patch` file, add a row to the table above, and cover the change in a test (see
`tests/test_core_imgui_cursors.cpp`).

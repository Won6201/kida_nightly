# Revisions

Manual revision log for notable build/toolchain/hardware-interface changes that
aren't obvious from the git history alone. Newest entries on top.

---

## 2026-07-29 — `eio/eio-dg5f.c`: target DG-5F-S (20 DOF) instead of DG-5F-M

**Change:** Switched the gripper model passed to `SetGripperOption()` from the
M model to the S model (20 DOF variant), in `eio-dg5f.c`:

```diff
-    if (htype == 0) gs.model = DG_MODEL_DG_5F_LEFT;    // 0x5F12
-    else gs.model = DG_MODEL_DG_5F_RIGHT;              // 0x5F22
+    if (htype == 0) gs.model = DG_MODEL_DG_5F_S_LEFT;  // 0x5F14
+    else gs.model = DG_MODEL_DG_5F_S_RIGHT;            // 0x5F24
```

**Why:** Moving from the DG-5F-M hands to DG-5F-S hands. Per the DG-5F-S SDK
manual (`eio/dg5fs_sdk/DG-5F-S_SDK_Manual.pdf`, §9.2 `enum DG_MODEL`), the
**DG-5F-S is 20 DOF** — same joint count as the M model. So this is the *only*
functional change needed: `jointCount=20`, `fingerCount=5`, the `u[20]`/`out[60]`
arrays, all `i<20` loops, the UDP wire sizes in `eio-kida.c`, and the Python
proprio layout (3×20 per hand in `kida.py`/`dg5f.py`) are all unchanged.

(The DG-5F-S15 variant would be 15 DOF and would instead require touching all of
the above — not our case.)

**Verification:** `g++ ... eio-dg5f.c -lDGSDK` compiles clean (exit 0). The new
`DG_MODEL_DG_5F_S_LEFT/RIGHT` enums exist in the installed DGSDK v2.0.1 header
(identical to the bundled `eio/dg5fs_sdk/` copy).

**NOT yet done / TODO (blocked on hardware, ~next week):**
- Confirm the S grippers' actual IPs vs the hardcoded `192.168.0.73` (left) /
  `192.168.0.72` (right) in `eio-dg5f.c:72-73`; fix there if they differ.
- Real-hardware bring-up check (`./kida-run -g 1 -x -v`).
- Current-limit note only: S model allows up to 360 mA (M was 150 mA). No code
  change today — `eio-dg5f.c` never calls the current-control API. Revisit only
  if/when current control is added.

## 2026-07-29 — `eio/build.sh`: dg5f compile `gcc` → `g++` (DGSDK 2.0.x)

**Change:** In `eio/build.sh`, the `eio-dg5f` build command switched from
`gcc` to `g++`:

```diff
-    gcc -W -Wall -o eio-dg5f eio-dg5f.c -I/usr/local/include/DGSDK -lDGSDK
+    g++ -W -Wall -o eio-dg5f eio-dg5f.c -I/usr/local/include/DGSDK -lDGSDK
```

**Why:** The installed Tesollo DGSDK header was upgraded (v1.7.2 → v2.0.x).
The new `/usr/local/include/DGSDK/DGSDK.h` declares functions with the C++-only
`noexcept` specifier, e.g.:

```c
DGSDK void GetLibraryVersion(int* version) noexcept;   // line 37
```

Compiling `eio-dg5f.c` with `gcc` (C mode) makes the C parser choke on
`noexcept`, which cascades into a wall of errors:

- `DGSDK.h:37: old-style parameter declarations in prototyped function definition`
- repeated `expected declaration specifiers before '__attribute__'`
  (the `DGSDK` = `__attribute__((visibility("default")))` macro on every prototype)
- finally our own code breaks: `storage class specified for parameter 'connected'`,
  `expected '{' at end of input`

The header is wrapped in `extern "C" { ... }`, so symbol linkage stays C and
`-lDGSDK` links fine — only the *compile* stage needs to be C++. `g++` builds it
cleanly (no warnings). `eio-dg5f.c` is plain enough to be valid C++ as-is; no
source change was needed.

**Verification:** `g++ -W -Wall -o eio-dg5f eio-dg5f.c -I/usr/local/include/DGSDK -lDGSDK`
compiles with exit 0 and no warnings.

**Notes / gotchas:**
- Old SDK v1.7.2 headers had **no** `noexcept` → `gcc` worked; that's why this
  built fine before the SDK bump.
- `gcc ... -lstdc++` is *not* a sufficient fix — the compile front-end itself
  must be C++, not just the link stage.
- The DGSDK header is CRLF + ISO-8859 (non-UTF-8 bytes), so plain `grep` treats
  it as binary and silently hides matches; use `grep -a` when inspecting it.
- On machines without `/usr/local/include/DGSDK`, `build.sh` still skips the
  dg5f build and keeps the tracked `eio/eio-dg5f` binary.

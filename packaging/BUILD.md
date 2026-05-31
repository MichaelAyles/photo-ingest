# Packaging & distribution (scaffolding)

> **Status: scaffolding, not yet wired.** The `[tool.briefcase]` block in
> `pyproject.toml` and the entitlements stub in this directory are a starting
> point. Code signing and notarization require the owner's Apple Developer ID
> (macOS) and an Authenticode certificate (Windows), neither of which is
> configured here. Nothing in this directory produces a shippable, signed build
> yet.

## Intended distribution path

Banger is a [pywebview](https://pywebview.flowrl.com/) desktop app (no Chromium
bundled), so the plan is to ship it as a native, self-contained installer per OS
using [BeeWare Briefcase](https://briefcase.readthedocs.io/):

- **macOS** → a `.app` bundle wrapped in a signed, notarized `.dmg`.
- **Windows** → an Authenticode-signed installer (MSI / app bundle).

Briefcase wraps the Python runtime + deps so the end user does not need a
Python install or to run `pip`.

## Prerequisites

```sh
pip install briefcase
```

Plus a working build toolchain for the native deps in the `requires` list
(torch, opencv-python, rawpy/libraw). On macOS a universal2 build needs both
arch slices of every wheel; some of these (notably rawpy) may need to be built
from source or pinned to a universal2-capable release.

## Build commands (once signing is wired)

```sh
# Create the platform scaffold + collect deps
briefcase create

# Compile the app bundle
briefcase build

# Package an installer
briefcase package            # -> .dmg (macOS) / installer (Windows)
```

## macOS signing + notarization (NOT yet wired)

Requires an **Apple Developer ID Application** certificate in the login
keychain and an app-specific password / notarization profile. None of this is
present in the repo.

1. Set the signing identity (the owner must supply their Developer ID):

   ```sh
   briefcase package macOS --identity "Developer ID Application: Michael Ayles (TEAMID)"
   ```

2. Notarization is handled by Briefcase via `notarytool`; it needs a stored
   credential profile created with `xcrun notarytool store-credentials`.

3. Entitlements: see [`macos/Entitlements.plist`](macos/Entitlements.plist).
   It is a **stub** — it currently only relaxes the hardened-runtime checks
   that the bundled Python + native extensions need. Tighten before shipping.

## Windows signing (NOT yet wired)

Requires an Authenticode code-signing certificate (e.g. via `signtool`). The
cert and password are supplied at build time and must never be committed.

```sh
briefcase package windows --adhoc-sign   # placeholder until a real cert exists
```

## Open items before a real release

- [ ] Obtain + configure the Apple Developer ID and notarization profile.
- [ ] Obtain + configure the Windows Authenticode certificate.
- [ ] Verify universal2 wheels (or source builds) for torch / opencv / rawpy.
- [ ] Decide whether model weights ship in the bundle or download on first run.
- [ ] Tighten `Entitlements.plist` to the minimum the app actually needs.
- [ ] Add a signed-build job to CI (gated on secrets, not on every push).

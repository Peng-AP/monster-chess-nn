# Build the native crate and install it where Python can import it.
#
# Three things have to be right and none of them are defaults on this box:
#   * MSVC env  -- rustc finds link.exe via vcvars; without it the SDK libs
#                  (dbghelp.lib) are not on LIB.
#   * PYO3_PYTHON -- this machine uses the `py` launcher, so there is no
#                  `python` on PATH for pyo3-build-config to discover.
#   * the copy   -- cargo emits monster_native.dll; Python imports .pyd.
#                  Forgetting this silently tests the PREVIOUS build, which is
#                  worse than a failure because it looks like a passing run.
#
# Do not run this from Git Bash: its /usr/bin/link shadows MSVC's link.exe and
# the failure ("extra operand") looks like a Rust error rather than a PATH one.
#
#   powershell -File tools/build_native.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found. Install the 'Desktop development with C++' workload."
}
cmd /c "call `"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
    }
}

$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$env:PYO3_PYTHON = (& py -3 -c "import sys; print(sys.executable)")

Set-Location (Join-Path $root "native")
cargo build --release
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

$dll = Join-Path $root "native\target\release\monster_native.dll"
$pyd = Join-Path $root "native\monster_native.pyd"
Copy-Item $dll $pyd -Force

Set-Location $root
$check = & py -3 -c "import sys; sys.path.insert(0,'native'); import monster_native as m; print(m.version())"
Write-Output "built and installed monster_native $check"

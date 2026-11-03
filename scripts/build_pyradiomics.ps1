param([string]$Python = '.\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$source = Join-Path $root '.build\pyradiomics'
if (-not (Test-Path -LiteralPath $source)) {
    git clone --branch v3.1.0 --depth 1 https://github.com/AIM-Harvard/pyradiomics.git $source
    if ($LASTEXITCODE -ne 0) { throw 'PyRadiomics source retrieval failed' }
}
$revision = git -C $source rev-parse HEAD
if ($revision -ne '6a761c4eb01905cfcb1139b0b158547c55edd4df') { throw 'Unexpected PyRadiomics source revision' }
& $pythonPath (Join-Path $root 'scripts\prepare_pyradiomics_build.py')
if ($LASTEXITCODE -ne 0) { throw 'PyRadiomics release verification failed' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Microsoft Visual Studio C++ Build Tools and Windows SDK first' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'C++ x64 build tools are unavailable' }
$sdkRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\Include'
$sdkHeaders = Get-ChildItem -Path (Join-Path $sdkRoot '*\ucrt\stdlib.h') -ErrorAction SilentlyContinue
if (-not $sdkHeaders) { throw 'Windows SDK UCRT headers are missing; complete or repair the Windows11SDK.26100 component before building' }
$developerShell = Join-Path $installation 'Common7\Tools\Launch-VsDevShell.ps1'
& $developerShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
Set-Location -LiteralPath $source
$priorPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $source
& $pythonPath -m pip wheel --no-build-isolation --no-deps . --wheel-dir (Join-Path $root '.build\wheels')
$buildExitCode = $LASTEXITCODE
$env:PYTHONPATH = $priorPythonPath
if ($buildExitCode -ne 0) { throw 'PyRadiomics wheel compilation failed' }
$wheel = Get-ChildItem -LiteralPath (Join-Path $root '.build\wheels') -Filter 'pyradiomics-3.1.0-*.whl' | Select-Object -First 1
if (-not $wheel) { throw 'Expected PyRadiomics 3.1.0 wheel was not produced' }
& $pythonPath -m pip install $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw 'PyRadiomics wheel installation failed' }
Get-FileHash -LiteralPath $wheel.FullName -Algorithm SHA256 | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root '.build\pyradiomics-wheel-sha256.json')
Set-Location -LiteralPath $root

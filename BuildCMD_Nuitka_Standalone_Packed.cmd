@echo off
rem Packed standalone build: the game code compiled by Nuitka, every asset shipped as an encrypted blob.
rem Steam then only re-sends the files that changed (see Modules/Assets/pack_format.py for why).
rem
rem Before the FIRST build, and whenever models/maps change: run the game once from source (python app.py) so
rem the generated caches under Assets (collision shapes, UV layouts, lightmaps) are complete - anything missing
rem from the pack is rebuilt on the player's machine at first load (slower, and written into the install folder).
rem
rem KEEP asset_key.bin SAFE (project root, git-ignored): a new key re-encrypts everything, so every player
rem re-downloads the whole game on the next update. tools\pack_assets.py creates it on its first run.
rem
rem The macOS build runs on GitHub (.github/workflows/build-macos.yml, via tools\mac_build.py) while Windows
rem compiles here, so it builds what is PUSHED to master - push first, or it will ask. Both then go up to
rem Steam as one build (STEAM_APP_VDF: Windows from STEAM_CONTENT, macOS from STEAM_CONTENT_MAC). Set
rem STEAM_BUILD_USER to your Steamworks login to skip the prompt; steamcmd asks for the password/Steam Guard
rem code the first time and caches the login after that.
rem
rem   --no-upload   stop once both builds are in the Steam content folders
rem   --no-mac      Windows only (uploads with simple_app_build.vdf, a build WITHOUT the macOS depot)
rem
rem Every build that uploads raises the game version's last number first (Modules\version.py, via
rem tools\bump_version.py - 0.1.4 becomes 0.1.5; a new minor version or 1.0 is only ever set by hand). Lobbies carry
rem the version and players on a different one can't join them. The macOS build is handed the same number, so it
rem matches even though version.py isn't pushed; commit the bumped file with your next commit. --no-upload builds
rem keep the current number.

cd /d "%~dp0"

set "STEAM_CONTENT=C:\Steambuilds\Content"
set "STEAM_CONTENT_MAC=C:\Steambuilds\ContentMac"
set "STEAM_APP_VDF=%STEAM_CONTENT%\app_build_win_mac.vdf"
set "STEAMCMD=C:\steamcmd\steamcmd\steamcmd.exe"

set "NO_UPLOAD="
set "NO_MAC="
for %%A in (%*) do (
    if /i "%%~A"=="--no-upload" set "NO_UPLOAD=1"
    if /i "%%~A"=="--no-mac" set "NO_MAC=1"
)
if defined NO_MAC set "STEAM_APP_VDF=%STEAM_CONTENT%\simple_app_build.vdf"

echo === 0/8 game version ===
set "GAME_VERSION="
if defined NO_UPLOAD (
    for /f "delims=" %%V in ('python tools\bump_version.py --print') do set "GAME_VERSION=%%V"
) else (
    for /f "delims=" %%V in ('python tools\bump_version.py') do set "GAME_VERSION=%%V"
)
if not defined GAME_VERSION (echo Couldn't read the game version from Modules\version.py & exit /b 1)
if defined NO_UPLOAD (echo %GAME_VERSION% ^(unchanged: not uploading^)) else (echo %GAME_VERSION%)

echo === 1/8 starting the macOS build on GitHub ===
if defined NO_MAC (echo skipped) else (python tools\mac_build.py start %GAME_VERSION% || exit /b 1)

echo === 2/8 packing assets ===
python tools\pack_assets.py --out build_nuitka\packed --verify || exit /b 1

rem Size trims (each verified before adding): scipy (only an optional extra of trimesh), PIL's AVIF codec, and 19 Panda3D
rem DLLs the game never loads (it only uses Panda3D for Bullet physics: no Panda windows, audio, video or model loaders).
echo === 3/8 compiling (NO --include-data-dir=Assets: plain assets must not go into the build) ===
python -m nuitka --standalone --assume-yes-for-downloads --disable-cache=ccache --windows-console-mode=disable --windows-icon-from-ico="Assets\icon\icon.ico" --include-data-files=steam_api64.dll=steam_api64.dll --nofollow-import-to=Frontend --nofollow-import-to=scipy --nofollow-import-to=PIL.AvifImagePlugin --nofollow-import-to=PIL._avif --noinclude-dlls=avcodec-*.dll --noinclude-dlls=avformat-*.dll --noinclude-dlls=avutil-*.dll --noinclude-dlls=swscale-*.dll --noinclude-dlls=swresample-*.dll --noinclude-dlls=d3dx9_*.dll --noinclude-dlls=fmodex64.dll --noinclude-dlls=cgd3d9.dll --noinclude-dlls=cggl.dll --noinclude-dlls=libp3tinydisplay.dll --noinclude-dlls=libp3assimp.dll --noinclude-dlls=libp3openal_audio.dll --noinclude-dlls=libp3ptloader.dll --noinclude-dlls=libp3fmod_audio.dll --noinclude-dlls=libp3ffmpeg.dll --noinclude-dlls=libp3windisplay.dll --noinclude-dlls=libpandagl.dll --noinclude-dlls=libpandadx9.dll --noinclude-dlls=libpandaegg.dll --user-plugin=nuitka_gpu_preference_plugin.py --output-filename=RatWar.exe --output-dir=build_nuitka app.py || exit /b 1

echo === 4/8 adding the packed assets ===
robocopy build_nuitka\packed\Assets build_nuitka\app.dist\Assets /MIR /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 8 exit /b 1

echo === 5/8 checking the build ===
python tools\check_packed_build.py build_nuitka\app.dist RatWar.exe || exit /b 1

rem /XF *.vdf: the build scripts live in the content folder - excluded files are neither copied nor purged.
echo === 6/8 mirroring into %STEAM_CONTENT% ===
robocopy build_nuitka\app.dist "%STEAM_CONTENT%" /MIR /XF *.vdf /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 8 exit /b 1

echo === 7/8 waiting for the macOS build, then unpacking it into %STEAM_CONTENT_MAC% ===
if defined NO_MAC (echo skipped) else (python tools\mac_build.py fetch "%STEAM_CONTENT_MAC%" || exit /b 1)

if defined NO_UPLOAD (
    echo.
    echo Done: %STEAM_CONTENT% and %STEAM_CONTENT_MAC% ^(upload skipped^)
    exit /b 0
)

echo === 8/8 uploading to Steam ===
if not exist "%STEAMCMD%" (echo steamcmd not found at %STEAMCMD% & exit /b 1)
if not exist "%STEAM_APP_VDF%" (echo App build script not found at %STEAM_APP_VDF% & exit /b 1)
if not defined STEAM_BUILD_USER set /p "STEAM_BUILD_USER=Steamworks login name: "
"%STEAMCMD%" +login %STEAM_BUILD_USER% +run_app_build "%STEAM_APP_VDF%" +quit || exit /b 1

echo.
echo Done: version %GAME_VERSION% uploaded with %STEAM_APP_VDF% - set the new build live on its branch in Steamworks
echo (launch options: RatWar.exe on Windows, RatWar.app on macOS)

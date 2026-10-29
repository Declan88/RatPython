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
rem Steps 5-6 mirror the build into STEAM_CONTENT and upload it with steamcmd using STEAM_APP_VDF. Set
rem STEAM_BUILD_USER to your Steamworks login to skip the prompt; steamcmd asks for the password/Steam Guard
rem code the first time and caches the login after that. Pass --no-upload to stop after the local build.

cd /d "%~dp0"

set "STEAM_CONTENT=C:\Steambuilds\Content"
set "STEAM_APP_VDF=%STEAM_CONTENT%\simple_app_build.vdf"
set "STEAMCMD=C:\steamcmd\steamcmd\steamcmd.exe"

echo === 1/6 packing assets ===
python tools\pack_assets.py --out build_nuitka\packed --verify || exit /b 1

rem Size trims (each verified before adding): scipy (only an optional extra of trimesh), PIL's AVIF codec, and 19 Panda3D
rem DLLs the game never loads (it only uses Panda3D for Bullet physics: no Panda windows, audio, video or model loaders).
echo === 2/6 compiling (NO --include-data-dir=Assets: plain assets must not go into the build) ===
python -m nuitka --standalone --assume-yes-for-downloads --disable-cache=ccache --windows-console-mode=disable --windows-icon-from-ico="Assets\icon\icon.ico" --include-data-files=steam_api64.dll=steam_api64.dll --nofollow-import-to=Frontend --nofollow-import-to=scipy --nofollow-import-to=PIL.AvifImagePlugin --nofollow-import-to=PIL._avif --noinclude-dlls=avcodec-*.dll --noinclude-dlls=avformat-*.dll --noinclude-dlls=avutil-*.dll --noinclude-dlls=swscale-*.dll --noinclude-dlls=swresample-*.dll --noinclude-dlls=d3dx9_*.dll --noinclude-dlls=fmodex64.dll --noinclude-dlls=cgd3d9.dll --noinclude-dlls=cggl.dll --noinclude-dlls=libp3tinydisplay.dll --noinclude-dlls=libp3assimp.dll --noinclude-dlls=libp3openal_audio.dll --noinclude-dlls=libp3ptloader.dll --noinclude-dlls=libp3fmod_audio.dll --noinclude-dlls=libp3ffmpeg.dll --noinclude-dlls=libp3windisplay.dll --noinclude-dlls=libpandagl.dll --noinclude-dlls=libpandadx9.dll --noinclude-dlls=libpandaegg.dll --user-plugin=nuitka_gpu_preference_plugin.py --output-filename=RatWar.exe --output-dir=build_nuitka app.py || exit /b 1

echo === 3/6 adding the packed assets ===
robocopy build_nuitka\packed\Assets build_nuitka\app.dist\Assets /MIR /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 8 exit /b 1

echo === 4/6 checking the build ===
python tools\check_packed_build.py build_nuitka\app.dist RatWar.exe || exit /b 1

rem /XF *.vdf: the build scripts live in the content folder - excluded files are neither copied nor purged.
echo === 5/6 mirroring into %STEAM_CONTENT% ===
robocopy build_nuitka\app.dist "%STEAM_CONTENT%" /MIR /XF *.vdf /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 8 exit /b 1

if /i "%~1"=="--no-upload" (
    echo.
    echo Done: %STEAM_CONTENT% ^(upload skipped^)
    exit /b 0
)

echo === 6/6 uploading to Steam ===
if not exist "%STEAMCMD%" (echo steamcmd not found at %STEAMCMD% & exit /b 1)
if not exist "%STEAM_APP_VDF%" (echo App build script not found at %STEAM_APP_VDF% & exit /b 1)
if not defined STEAM_BUILD_USER set /p "STEAM_BUILD_USER=Steamworks login name: "
"%STEAMCMD%" +login %STEAM_BUILD_USER% +run_app_build "%STEAM_APP_VDF%" +quit || exit /b 1

echo.
echo Done: uploaded %STEAM_CONTENT% - set the new build live on its branch in Steamworks (launch exe: RatWar.exe)

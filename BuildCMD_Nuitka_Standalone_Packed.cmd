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

cd /d "%~dp0"

echo === 1/4 packing assets ===
python tools\pack_assets.py --out build_nuitka\packed --verify || exit /b 1

rem Size trims (each verified before adding): scipy (only an optional extra of trimesh), PIL's AVIF codec, and 19 Panda3D
rem DLLs the game never loads (it only uses Panda3D for Bullet physics: no Panda windows, audio, video or model loaders).
echo === 2/4 compiling (NO --include-data-dir=Assets: plain assets must not go into the build) ===
python -m nuitka --standalone --assume-yes-for-downloads --disable-cache=ccache --windows-console-mode=disable --windows-icon-from-ico="Assets\icon\icon.ico" --include-data-files=steam_api64.dll=steam_api64.dll --nofollow-import-to=Frontend --nofollow-import-to=scipy --nofollow-import-to=PIL.AvifImagePlugin --nofollow-import-to=PIL._avif --noinclude-dlls=avcodec-*.dll --noinclude-dlls=avformat-*.dll --noinclude-dlls=avutil-*.dll --noinclude-dlls=swscale-*.dll --noinclude-dlls=swresample-*.dll --noinclude-dlls=d3dx9_*.dll --noinclude-dlls=fmodex64.dll --noinclude-dlls=cgd3d9.dll --noinclude-dlls=cggl.dll --noinclude-dlls=libp3tinydisplay.dll --noinclude-dlls=libp3assimp.dll --noinclude-dlls=libp3openal_audio.dll --noinclude-dlls=libp3ptloader.dll --noinclude-dlls=libp3fmod_audio.dll --noinclude-dlls=libp3ffmpeg.dll --noinclude-dlls=libp3windisplay.dll --noinclude-dlls=libpandagl.dll --noinclude-dlls=libpandadx9.dll --noinclude-dlls=libpandaegg.dll --user-plugin=nuitka_gpu_preference_plugin.py --output-filename=RatWar.exe --output-dir=build_nuitka app.py || exit /b 1

echo === 3/4 adding the packed assets ===
robocopy build_nuitka\packed\Assets build_nuitka\app.dist\Assets /MIR /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 8 exit /b 1

echo === 4/4 checking the build ===
python tools\check_packed_build.py build_nuitka\app.dist RatWar.exe || exit /b 1

echo.
echo Done: build_nuitka\app.dist  (copy its CONTENTS to the Steam content folder, keeping the *.vdf files there;
echo the launch executable in Steamworks is RatWar.exe)

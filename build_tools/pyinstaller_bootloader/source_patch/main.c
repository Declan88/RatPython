/*
 * ****************************************************************************
 * Copyright (c) 2013-2023, PyInstaller Development Team.
 *
 * Distributed under the terms of the GNU General Public License (version 2
 * or later) with exception for distributing the bootloader.
 *
 * The full license is in the file COPYING.txt, distributed with this software.
 *
 * SPDX-License-Identifier: (GPL-2.0-or-later WITH Bootloader-exception)
 * ****************************************************************************
 */

/* This file contains three different entry points, for different
 * combinations of OS and pre-processor definitions:
 *   - wWinMain: for Windows with console=False
 *   - wmain: for Windows with console=True
 *   - main: for POSIX systems
 */

#ifdef _WIN32
    #include <windows.h>
#endif

#ifdef __FreeBSD__
    #include <floatingpoint.h>
#endif

#include "pyi_main.h"


#if defined(_WIN32)

/* Prevent programs compiled with MinGW (gcc) from performing glob-style
 * wildcard expansion of command-line arguments. This keeps behavior
 * consistent between applications that use MinGW-compiled and MSVC-compiled
 * bootloader. */
extern int _CRT_glob;
int _CRT_glob = 0;

/* Hint to NVIDIA's and AMD's drivers, on a laptop with switchable/Optimus
 * graphics, to run this process on the dedicated GPU instead of the
 * integrated one. These MUST be exported from the actual .exe's own PE
 * export table to be honored - confirmed that a DLL loaded by the process
 * does NOT work, only the main executable module does. Since this
 * bootloader IS the resulting app.exe for any PyInstaller build, this is
 * the only place that can actually carry this hint for a frozen Python
 * application - there is no other point in the build where the final
 * exe's own export table can be influenced. */
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

#if defined(WINDOWED)

/* Entry point for Windows when console=False */
int WINAPI
wWinMain(
    HINSTANCE hInstance,      /* handle to current instance */
    HINSTANCE hPrevInstance,  /* handle to previous instance */
    LPWSTR lpCmdLine,         /* pointer to command line */
    int nCmdShow              /* show state of window */
    )
{
    /* Store arguments in global context structure. */
    global_pyi_ctx->argc = __argc;
    global_pyi_ctx->argv_w = __wargv;

    return pyi_main(global_pyi_ctx);
}

#else /* defined(WINDOWED) */

/* Entry point for Windows when console=True */
int
wmain(int argc, wchar_t **argv)
{
    /* Store arguments in global context structure. */
    global_pyi_ctx->argc = argc;
    global_pyi_ctx->argv_w = argv;

    return pyi_main(global_pyi_ctx);
}

#endif /* defined(WINDOWED) */

#else /* defined(_WIN32) */

/* Entry point for POSIX */
int
main(int argc, char **argv)
{
#ifdef __FreeBSD__
    /* PEP-754 requires that FP exceptions run in "no stop" mode by default,
     * and until C vendors implement C99's ways to control FP exceptions,
     * Python requires non-stop mode.  Alas, some platforms enable FP
     * exceptions by default. Here we disable them. */
    fpsetmask(fpgetmask() & ~FP_X_OFL);
#endif

    /* Store arguments in global context structure. */
    global_pyi_ctx->argc = argc;
    global_pyi_ctx->argv = argv;

    return pyi_main(global_pyi_ctx);
}

#endif /* defined(WIN32) */

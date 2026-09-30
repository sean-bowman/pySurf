# -- Animation Render Script -- #

'''

CLI tool for rendering SurfAnimations Manim scenes.

Usage:
    python SurfAnimations/render.py --scene wave_intro
    python SurfAnimations/render.py --scene board_side
    python SurfAnimations/render.py --scene board_perspective
    python SurfAnimations/render.py --scene performance
    python SurfAnimations/render.py --all
    python SurfAnimations/render.py --all --quality low

Sean Bowman [02/04/2026]

'''

import argparse
import os
import shutil
import subprocess
import sys

#--------------------------------------------------------------------#
# -- Scene Registry -- #
#--------------------------------------------------------------------#

SCENES = {
    'wave_intro': {
        'file': 'src/pysurf/Surfboard/SurfAnimations/scenes/waveIntro.py',
        'class': 'WavePhysicsIntro',
        'description': 'Linear wave theory introduction',
    },
    'board_side': {
        'file': 'src/pysurf/Surfboard/SurfAnimations/scenes/boardOnWave.py',
        'class': 'BoardOnWaveSideView',
        'description': 'Board riding wave (2D side view)',
    },
    'board_perspective': {
        'file': 'src/pysurf/Surfboard/SurfAnimations/scenes/boardOnWave.py',
        'class': 'BoardOnWavePerspective',
        'description': 'Board riding wave (2.5D perspective)',
    },
    'performance': {
        'file': 'src/pysurf/Surfboard/SurfAnimations/scenes/performanceComparison.py',
        'class': 'PerformanceComparison',
        'description': 'Multi-board performance comparison',
    },
    'sloshing_tank': {
        'file': 'src/pysurf/Surfboard/SurfAnimations/scenes/sloshingTank.py',
        'class': 'SloshingTankAnimation',
        'description': 'SPH sloshing tank simulation',
    },
}

QUALITY_FLAGS = {
    'low': '-ql',       # 480p, 15fps
    'medium': '-qm',    # 720p, 30fps
    'high': '-qh',      # 1080p, 60fps
    'fourk': '-qk',     # 4K, 60fps
}


#--------------------------------------------------------------------#
# -- Helpers -- #
#--------------------------------------------------------------------#

def _ensureFfmpegPath() -> None:

    '''

    Add conda Library/bin to PATH if ffmpeg is not already findable.
    This is needed on Windows when using conda-installed ffmpeg.

    '''
    if shutil.which('ffmpeg') is not None:
        return

    # Try common conda locations
    condaPaths = [
        os.path.join(sys.prefix, 'Library', 'bin'),
        os.path.join(os.path.dirname(sys.executable), '..', 'Library', 'bin'),
    ]

    for condaPath in condaPaths:
        condaPath = os.path.abspath(condaPath)
        ffmpegPath = os.path.join(condaPath, 'ffmpeg.exe')
        if os.path.isfile(ffmpegPath):
            os.environ['PATH'] = condaPath + os.pathsep + os.environ.get('PATH', '')
            print(f'  Added {condaPath} to PATH for ffmpeg')
            return

    print('  Warning: ffmpeg not found. Video rendering may fail.')
    print('  Install with: conda install -c conda-forge ffmpeg')

def _projectRoot() -> str:

    '''Get the project root directory.'''
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def renderScene(sceneName: str, quality: str = 'high') -> bool:

    '''

    Render a single Manim scene.

    Parameters:
    -----------
    sceneName : str
        Key from SCENES dict
    quality : str
        Quality preset: low, medium, high, fourk

    Returns:
    --------
    bool : True if rendering succeeded, False otherwise

    '''
    if sceneName not in SCENES:
        print(f'Error: Unknown scene "{sceneName}"')
        print(f'Available scenes: {", ".join(SCENES.keys())}')
        return False

    sceneInfo = SCENES[sceneName]
    qualityFlag = QUALITY_FLAGS.get(quality, '-qh')
    projectRoot = _projectRoot()

    print(f'\nRendering: {sceneInfo["description"]}')
    print(f'  File:    {sceneInfo["file"]}')
    print(f'  Class:   {sceneInfo["class"]}')
    print(f'  Quality: {quality} ({qualityFlag})')
    print()

    cmd = [
        sys.executable, '-m', 'manim', 'render',
        qualityFlag,
        '--media_dir', os.path.join(projectRoot, 'src', 'pysurf', 'Surfboard', 'SurfAnimations', 'media'),
        os.path.join(projectRoot, sceneInfo['file']),
        sceneInfo['class'],
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=projectRoot,
            check=True,
        )
        print(f'\nCompleted: {sceneName}')
        return True
    except subprocess.CalledProcessError as e:
        print(f'\nFailed to render {sceneName}: {e}')
        return False

def renderAll(quality: str = 'high') -> dict:

    '''

    Render all registered scenes.

    Parameters:
    -----------
    quality : str
        Quality preset: low, medium, high, fourk

    Returns:
    --------
    dict : Mapping of scene name to success (bool)

    '''
    _ensureFfmpegPath()
    results = {}
    for name in SCENES:
        results[name] = renderScene(name, quality)
    return results


#--------------------------------------------------------------------#
# -- Main -- #
#--------------------------------------------------------------------#

def main() -> None:
    '''CLI entry point for rendering Manim scenes.'''
    parser = argparse.ArgumentParser(
        description='Render SurfAnimations Manim scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Available scenes:\n'
            + '\n'.join(
                f'  {name:<22s} {info["description"]}'
                for name, info in SCENES.items()
            )
        ),
    )

    parser.add_argument(
        '--scene', '-s',
        choices=list(SCENES.keys()),
        help='Scene to render',
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Render all scenes',
    )
    parser.add_argument(
        '--quality', '-q',
        choices=list(QUALITY_FLAGS.keys()),
        default='high',
        help='Render quality (default: high = 1080p)',
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available scenes and exit',
    )

    args = parser.parse_args()

    if args.list:
        print('Available scenes:')
        for name, info in SCENES.items():
            print(f'  {name:<22s} {info["description"]}')
        return

    if not args.scene and not args.all:
        parser.print_help()
        return

    # Ensure ffmpeg is available
    _ensureFfmpegPath()

    if args.all:
        print('Rendering all scenes...')
        results = {}
        for name in SCENES:
            results[name] = renderScene(name, args.quality)

        print('\n' + '=' * 50)
        print('Render Summary:')
        for name, success in results.items():
            status = 'OK' if success else 'FAILED'
            print(f'  {name:<22s} [{status}]')
    else:
        renderScene(args.scene, args.quality)


if __name__ == '__main__':
    main()

"""Loopback-only entry point for source and packaged execution."""
import argparse

import uvicorn


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('port must be an integer from 1 to 65535') from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be an integer from 1 to 65535')
    return port


def run_app(app, argv=None) -> None:
    parser = argparse.ArgumentParser(description='StereoLab local media workbench')
    parser.add_argument('--port', type=_port, default=8765,
                        help='local HTTP port (default: 8765)')
    args = parser.parse_args(argv)
    print(f'StereoLab is running at http://127.0.0.1:{args.port}')
    print('All processing stays on this computer. Press Ctrl+C to stop.')
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')

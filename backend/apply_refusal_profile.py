"""Apply a reversible retrieval experiment without printing API keys.

Run from backend: uv run python apply_refusal_profile.py --profile hybrid
Restore the complete prior .env: --restore PATH_PRINTED_BY_THE_FIRST_RUN
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import tempfile

PROFILES = {
    'hybrid': {
        'RETRIEVAL_MODE': 'hybrid', 'HYBRID_LEXICAL_EXTRA': '4',
        'RETRIEVAL_TOP_K': '8', 'KSU_RERANK_POOL_SIZE': '8',
        'KAU_RERANK_POOL_SIZE': '8', 'KAU_GATE_MODE': 'rerank',
        'RETRIEVAL_DIAGNOSTICS': 'true', 'ANSWER_CACHE_SIZE': '0',
    },
    'baseline': {
        'RETRIEVAL_MODE': 'dense', 'HYBRID_LEXICAL_EXTRA': '0',
        'RETRIEVAL_TOP_K': '5', 'KSU_RERANK_POOL_SIZE': '6',
        'KAU_RERANK_POOL_SIZE': '6', 'KAU_GATE_MODE': 'rerank',
        'RETRIEVAL_DIAGNOSTICS': 'true', 'ANSWER_CACHE_SIZE': '0',
    },
}


def update_text(text: str, overrides: dict[str, str]) -> str:
    """Update all duplicate occurrences of owned keys; preserve other lines."""
    lines = text.splitlines()
    seen = set()
    for index, line in enumerate(lines):
        match = re.match(r'^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z_0-9]*)\s*=', line)
        if match and match[2] in overrides:
            key = match[2]
            lines[index] = f'{match[1]}{key}={overrides[key]}'
            seen.add(key)
    missing = [key for key in overrides if key not in seen]
    if missing:
        lines.extend(['', '# Retrieval experiment (local, reversible)'])
        lines.extend(f'{key}={overrides[key]}' for key in missing)
    return '\n'.join(lines) + '\n'


def replace_bytes(path: Path, content: bytes) -> None:
    """Replace .env atomically on Windows and Linux."""
    fd, name = tempfile.mkstemp(prefix='.retrieval-env-',suffix='.tmp',dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd,'wb') as file:
            file.write(content);file.flush();os.fsync(file.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--profile',choices=sorted(PROFILES))
    group.add_argument('--restore',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    env=root/'.env'
    if not env.is_file():parser.error('Existing backend/.env is required. Configure your API key first.')
    original=env.read_bytes()
    replacement=args.restore.read_bytes() if args.restore else update_text(original.decode('utf-8-sig'),PROFILES[args.profile]).encode('utf-8')
    backup_dir=root/'data'/'config_backups';backup_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=backup_dir/f'.env.before_retrieval_{stamp}'
    shutil.copy2(env,backup)
    replace_bytes(env,replacement)
    print('Saved configuration. API keys were not printed.')
    print(f'Backup (contains secrets; keep local): {backup}')
    print('Restart the backend. Shell environment variables override .env values.')
    if args.profile:
        for key,value in PROFILES[args.profile].items():print(f'{key}={value}')


if __name__=='__main__':
    main()

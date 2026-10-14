from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def strip_fenced_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)

def test_readme_referenced_local_paths_exist():
    text = strip_fenced_blocks((ROOT / 'README.md').read_text(encoding='utf-8'))
    candidates = re.findall(r'`([^`\n]+)`', text)
    path_like = [
        c for c in candidates
        if '/' in c and not c.startswith(('python ', 'pip ', 'jupyter ')) and not c.endswith('/')
    ]
    missing = [p for p in path_like if not (ROOT / p).exists()]
    assert not missing

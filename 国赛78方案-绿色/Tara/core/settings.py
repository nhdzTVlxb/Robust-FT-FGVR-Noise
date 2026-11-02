try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
RECIPE_FILE = Path(__file__).resolve().parents[1] / 'recipe.toml'

def load(overrides: Optional[dict]=None) -> SimpleNamespace:
    raw = tomllib.loads(RECIPE_FILE.read_text())
    for (section, pairs) in (overrides or {}).items():
        assert section in raw, 'unknown recipe section %s' % section
        raw[section].update({k: v for (k, v) in pairs.items() if v is not None})
    return SimpleNamespace(**{name: SimpleNamespace(**table) for (name, table) in raw.items()})

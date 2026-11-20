import yaml


def load_config(*paths: str) -> dict:
    """Shallow-merge YAML files; later files override earlier ones."""
    cfg: dict = {}
    for p in paths:
        with open(p) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg

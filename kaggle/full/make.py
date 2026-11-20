"""Write kaggle/full/<arm>/{run.py,kernel-metadata.json}. Usage: python kaggle/full/make.py STEPS TIMEOUT_SECONDS (GNU timeout rejects compound units like 10h20m)"""
import json
import pathlib
import sys

steps, timeout = sys.argv[1], sys.argv[2]
here = pathlib.Path(__file__).parent
tpl = (here / "run_template.py").read_text()
meta = json.loads((here.parent / "baseline" / "kernel-metadata.json").read_text())
for arm in ("en", "indic"):
    d = here / arm
    d.mkdir(exist_ok=True)
    (d / "run.py").write_text(tpl.replace("__ARM__", arm).replace("__STEPS__", steps).replace("__TIMEOUT__", timeout))
    m = dict(meta, id=f"dhairyaankurpandya/indic-rl-transfer-full-{arm}", title=f"indic-rl-transfer full {arm}")
    (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
print("ok", steps, timeout)

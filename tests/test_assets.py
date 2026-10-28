import hashlib
import json
from pathlib import Path

import pytest

from qat.assets import ASSETS, PROFILES, Asset, fetch, is_present, select, verify
from qat.data.kws_dataset import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_consistent():
    dests = [a.dest for a in ASSETS]
    assert len(dests) == len(set(dests))
    for a in ASSETS:
        assert a.size > 0
        assert a.sha256 or a.md5, f"{a.dest}: нет хеша"
        assert a.url.startswith("https://")
        assert not Path(a.dest).is_absolute() and ".." not in Path(a.dest).parts
        assert a.group in {g for gs in PROFILES.values() for g in gs}


def test_every_configured_voice_has_assets():
    voices = {Path(a.dest).name for a in ASSETS if a.group == "piper"}
    for v in load_config()["voices"]:
        assert f"{v['name']}.onnx" in voices and f"{v['name']}.onnx.json" in voices


def test_pinned_revisions_in_urls():
    # Hugging Face — только по закреплённому коммиту, не по main
    for a in ASSETS:
        if "huggingface.co" in a.url:
            assert "/resolve/main/" not in a.url


def test_verify_detects_size_and_hash(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    good = Asset("t", "https://x", "x.bin", 5, sha256=hashlib.sha256(b"hello").hexdigest(),
                 md5=hashlib.md5(b"hello").hexdigest())
    assert verify(good, f) == []
    assert len(verify(Asset("t", "https://x", "x.bin", 6, sha256=good.sha256), f)) == 1
    bad = verify(Asset("t", "https://x", "x.bin", 5, md5="0" * 32), f)
    assert len(bad) == 1 and bad[0].startswith("md5")


def test_fetch_refuses_over_limit_without_downloading(tmp_path):
    big = Asset("t", "https://example.invalid/big", "big.bin", 2_000_000_000, sha256="0" * 64)
    with pytest.raises(SystemExit, match="лимита"):
        fetch([big], root=tmp_path, max_bytes=500_000_000)
    assert not (tmp_path / "big.bin").exists()


def test_is_present_for_unzipped_assets(tmp_path):
    a = Asset("vosk", "https://x", "models/vosk/m-1.zip", 1, sha256="0" * 64,
              unzip_to="models/vosk")
    assert not is_present(a, tmp_path)
    (tmp_path / "models/vosk/m-1").mkdir(parents=True)
    assert is_present(a, tmp_path)


def test_profiles_cover_what_steps_need():
    kws = {a.group for a in select("kws")}
    assert {"piper", "vosk", "kscd", "fleurs", "fsd50k"} <= kws
    assert {"panns", "vosk"} <= {a.group for a in select("detectors")}


def test_colab_notebook_is_valid_json():
    nb = json.loads((ROOT / "notebooks" / "colab.ipynb").read_text())
    assert nb["nbformat"] == 4
    sources = "".join("".join(c["source"]) for c in nb["cells"])
    assert "colab_setup.sh" in sources and "kws_dataset synth" in sources

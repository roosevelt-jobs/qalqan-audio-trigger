import csv
import json

import numpy as np
import pytest
import soundfile as sf

from qat.detectors.energy import EnergyDetector
from qat.eval.bench import evaluate, load_manifest, write_report

SR = 16000


def tone_file(path, amplitude):
    """3 с: тишина 1 с, тон 440 Гц 1 с, тишина 1 с."""
    x = np.zeros(3 * SR, dtype=np.float32)
    t = np.arange(SR) / SR
    x[SR : 2 * SR] = amplitude * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), x, SR, subtype="FLOAT")


def noise_file(path, seconds, dbfs):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(seconds * SR) * 10 ** (dbfs / 20)).astype(np.float32)
    sf.write(str(path), x, SR, subtype="FLOAT")


@pytest.fixture
def manifest(tmp_path):
    rows = []
    # «близко»: тон ~ -9 dBFS -> оценка energy ~0.85; «далеко»: ~ -29 dBFS -> ~0.52
    for i, (cond, amp) in enumerate([("near", 0.5), ("near", 0.5), ("far", 0.05), ("far", 0.05)]):
        tone_file(tmp_path / f"pos{i}.wav", amp)
        rows.append({"path": f"pos{i}.wav", "label": "positive", "condition": cond,
                     "target_start_sec": 1.0, "target_end_sec": 2.0})
    # негатив: шум -35 dBFS 20 с -> оценка energy ~0.42 постоянно
    noise_file(tmp_path / "neg.wav", 20, -35)
    rows.append({"path": "neg.wav", "label": "negative"})
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def by_threshold(report):
    return {p["threshold"]: p for p in report["det"]}


def test_load_manifest(manifest, tmp_path):
    items = load_manifest(manifest)

    assert [i.label for i in items] == ["positive"] * 4 + ["negative"]
    assert items[0].path == tmp_path / "pos0.wav"
    assert items[0].target == (1.0, 2.0)
    assert items[2].condition == "far"
    assert items[4].target is None and items[4].condition == "unspecified"


def test_load_manifest_rejects_bad_label(tmp_path):
    bad = tmp_path / "m.jsonl"
    bad.write_text('{"path": "a.wav", "label": "maybe"}')
    with pytest.raises(ValueError, match="label"):
        load_manifest(bad)


def test_bench_known_answer_without_wake(manifest):
    report = evaluate(load_manifest(manifest), EnergyDetector(), use_wake=False)
    det = by_threshold(report)

    # 0.8: «близко» ловится, «далеко» нет; шум 0.42 ниже порога
    assert det[0.8]["frr_by_condition"] == {"far": 1.0, "near": 0.0}
    assert det[0.8]["false_alarms"] == 0
    # 0.3: всё ловится, но шум даёт одну тревогу (один эпизод) за 20 с -> 180 FA/h
    assert det[0.3]["frr"] == 0.0
    assert det[0.3]["false_alarms"] == 1
    assert det[0.3]["fa_per_hour"] == pytest.approx(180.0)
    # рабочий порог: самый низкий без тревог на шуме — 0.45; «далеко» (0.52) при нём ловится
    op = report["operating_point"]
    assert op["threshold"] == 0.45
    assert op["frr"] == 0.0
    # 20 с негатива — предупреждение о недостоверности FA/h
    assert any("негативного потока" in w for w in report["warnings"])


def test_bench_with_wake_rejects_noise(manifest):
    report = evaluate(load_manifest(manifest), EnergyDetector(), use_wake=True)
    det = by_threshold(report)

    # шум -35 dBFS отклоняется VAD: тревог нет даже на низком пороге
    assert det[0.3]["false_alarms"] == 0
    # тон -9 dBFS будит без VAD и ловится; тон -29 dBFS — не речь, VAD его не пропускает
    assert det[0.3]["frr_by_condition"] == {"far": 1.0, "near": 0.0}


def test_bench_is_deterministic_and_writes_files(manifest, tmp_path):
    items = load_manifest(manifest)
    r1 = evaluate(items, EnergyDetector(), use_wake=False)
    r2 = evaluate(items, EnergyDetector(), use_wake=False)
    assert r1 == r2

    write_report(r1, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "report.json").read_text())
    assert saved == r1
    with (tmp_path / "out" / "det.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(r1["det"])
    assert float(rows[0]["threshold"]) == r1["det"][0]["threshold"]

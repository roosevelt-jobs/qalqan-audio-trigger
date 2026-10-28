from pathlib import Path

import pytest
import soundfile as sf

from qat.data.kws_dataset import ROOT, _split_by, contains_phrase, fleurs_sentences, load_config
from qat.data.tts_synth import SynthJob, run_piper

VOICES = ROOT / "models" / "piper"
FLEURS = ROOT / "data" / "census" / "fleurs"
needs_voice = pytest.mark.skipif(
    not (VOICES / "ru_RU-denis-medium.onnx").is_file(), reason="нет голосов Piper"
)


def test_contains_phrase():
    assert contains_phrase("на помощь на помощь", ["на помощь"])
    assert contains_phrase("Позовите НА ПОМОЩЬ", ["на помощь"])
    assert not contains_phrase("скорая помощь", ["на помощь"])
    assert not contains_phrase("помощь на", ["на помощь"])  # порядок слов важен
    assert contains_phrase("көмектесіңдер", ["көмектесіңдер"])


def test_config_val_speakers_are_people_present_in_voices():
    cfg = load_config()
    people = {p for v in cfg["voices"] for p in v["speakers"].values()}
    assert set(cfg["val_speakers"]) <= people
    # для каждого языка в обучении остаются люди
    for lang in ("ru", "kk"):
        lang_people = {p for v in cfg["voices"] if v["lang"] == lang
                       for p in v["speakers"].values()}
        assert lang_people - set(cfg["val_speakers"])


def test_split_by_is_deterministic():
    keys = [f"k{i}" for i in range(2000)]
    a = [_split_by(k, 0.1, 0) for k in keys]
    assert a == [_split_by(k, 0.1, 0) for k in keys]
    assert 0.07 < a.count("val") / len(a) < 0.13


@pytest.mark.skipif(not FLEURS.is_dir(), reason="нет расшифровок FLEURS")
def test_fleurs_sentences_exclude_phrase():
    s = fleurs_sentences(FLEURS, "ru", ["на помощь"], 20, seed=0)
    assert len(s) == 20 and len(set(s)) == 20
    assert not any(contains_phrase(x, ["на помощь"]) for x in s)
    assert s == fleurs_sentences(FLEURS, "ru", ["на помощь"], 20, seed=0)


@needs_voice
def test_run_piper_maps_texts_to_16k_files(tmp_path):
    job = SynthJob("ru_RU-denis-medium", 0, 1.0, 0.667, 0.8)
    texts = ["На помощь!", "Скорая помощь."]

    files = run_piper(job, texts, VOICES, tmp_path)

    assert set(files) == set(texts)
    for t in texts:
        info = sf.info(str(files[t]))
        assert (info.samplerate, info.channels) == (16000, 1)
        assert 0.3 < info.duration < 4.0
    # «Скорая помощь.» длиннее «На помощь!» — файлы не перепутаны
    long, short = (sf.info(str(files[t])).duration for t in ("Скорая помощь.", "На помощь!"))
    assert long > short
    assert isinstance(files["На помощь!"], Path)

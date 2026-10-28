# ruff: noqa: E501 — таблица URL и хешей читается лучше в одну строку на файл
"""Загрузка моделей и данных из первоисточников: закреплённые версии, проверка хешей.

Одна команда восстанавливает всё, что не хранится в git (локально или в Google Colab):

    uv run python -m qat.assets fetch --profile kws
    uv run python -m qat.assets fetch --profile detectors --max-gb 0.5

Источники и лицензии — models/*/README.md и docs/data_card.md.
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/5041a7a49ae5b3a0f30e6837769eb03fdbb231bd"
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data"
KSCD = "https://huggingface.co/datasets/issai/kazakh-speech-commands/resolve/8efc3b804520cecdcc38e5bc935f0575c7c25517"
ZENODO_FSD50K = "https://zenodo.org/records/4060432/files"
ZENODO_PANNS = "https://zenodo.org/records/3987831/files"
VOSK = "https://alphacephei.com/vosk/models"


@dataclass(frozen=True)
class Asset:
    group: str
    url: str
    dest: str  # путь относительно корня репозитория
    size: int
    sha256: str | None = None
    md5: str | None = None
    unzip_to: str | None = None  # распаковать архив сюда и удалить архив


def _piper(path: str, size: int, sha: str, json_sha: str, json_size: int) -> list[Asset]:
    name = path.rsplit("/", 1)[-1]
    return [
        Asset("piper", f"{PIPER}/{path}.onnx", f"models/piper/{name}.onnx", size, sha),
        Asset("piper", f"{PIPER}/{path}.onnx.json", f"models/piper/{name}.onnx.json",
              json_size, json_sha),
    ]


ASSETS: list[Asset] = [
    # PANNs Cnn14_16k (детектор B): md5 — из записи Zenodo, SHA-256 — посчитан при загрузке.
    Asset("panns", f"{ZENODO_PANNS}/Cnn14_16k_mAP%3D0.438.pth?download=1",
          "models/panns/Cnn14_16k_mAP=0.438.pth", 358668570,
          sha256="e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6",
          md5="362fc5ff18f1d6ad2f6d464b45893f2c"),
    # Vosk (детектор C и проверка синтеза).
    Asset("vosk", f"{VOSK}/vosk-model-small-ru-0.22.zip", "models/vosk/vosk-model-small-ru-0.22.zip",
          46236750, "961d5ff98a17f4aa6de69864d0aa71fa5bac682301d2b5d17a3f24c5c99a46d4",
          unzip_to="models/vosk"),
    Asset("vosk", f"{VOSK}/vosk-model-small-kz-0.42.zip", "models/vosk/vosk-model-small-kz-0.42.zip",
          59697294, "b6d6bfe6195c866805bd18b1f56723e4d2ba4866e81df1481006d0c5704fda4f",
          unzip_to="models/vosk"),
    # Голоса Piper (синтез обучающих данных для A).
    *_piper("kk/kk_KZ/issai/high/kk_KZ-issai-high", 127864258,
            "4dee767c893e8535da821447d12cb030e3569e11254c14030a1da5d8b2222c16",
            "eb145ac8712b87beda9f0266e95b98cad76acce89d4626cd8b1408234e9842f0", 4358),
    *_piper("kk/kk_KZ/iseke/x_low/kk_KZ-iseke-x_low", 28130791,
            "dcf526e1c089823c9f896f8f7fddd9b05d538752e139c8a2b4cf4fba41c9715c",
            "7af909edcbec3d595cb8cfb810dbf19ea3d6bed5edc18ed9e5ed80631fd5e3a9", 4168),
    *_piper("kk/kk_KZ/raya/x_low/kk_KZ-raya-x_low", 28130791,
            "75552af00254baee9f85b976877e039f9530ea86ddb83d70262b72925785a0aa",
            "b8a239282afd4094e19a1aab0d44623e3bc4c7b4150d57c39ce972193c9e70e3", 4167),
    *_piper("ru/ru_RU/denis/medium/ru_RU-denis-medium", 63201294,
            "15fab56e11a097858ee115545d0f697fc2a316c41a291a5362349fb870411b0a",
            "831c860dac0b5073eaa81610a0a638ec23d90a6cf8e5f871b4485c2cec3767c8", 4823),
    *_piper("ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium", 63201294,
            "f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb",
            "667ef3117bc642c2892dff7690d8bdc8ca4228aeaa783b2dc1416df632855e0d", 4824),
    *_piper("ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium", 63201294,
            "72a5f88e0b20928064eb45d88e1daa21f8af62d18613580d32cbb4aed48dcf7f",
            "706a4fb17bc304abd07809b552deae615e64dcbffbfbd09854ba37ca59e88117", 4882),
    # Kazakh Speech Commands (живые негативы для A).
    Asset("kscd", f"{KSCD}/data.zip", "data/kscd/data.zip", 75445212,
          "323aa13a2856ae7c012dd0a16bd9bd9139d66591f4058351e50fdf475144a3b9",
          unzip_to="data/kscd/extracted"),
    # Расшифровки FLEURS (тексты для синтеза обычной речи).
    *[Asset("fleurs", f"{FLEURS}/{lang}/{split}.tsv", f"data/census/fleurs/{lang}_{split}.tsv",
            size, sha) for lang, split, size, sha in [
        ("ru_ru", "train", 2358557, "a18a051553daba44130d323ff8c7387dcde64ce855ade317c53209c0922885dc"),
        ("ru_ru", "dev", 316328, "25d9dc88a7533fc0f303354a8e6d3c8186c557229c9a0024487090a11226171e"),
        ("ru_ru", "test", 735258, "cd54f261220f49afbb4c128633a737eca4a22f6c0a8233d3cc891478d06676e6"),
        ("kk_kz", "train", 2812850, "618a8e83ab0b5fd1cf4c7dfb9f6777fc3dad145e9daf250e8698e56d69f944df"),
        ("kk_kz", "dev", 312534, "24195834bd3720cbbb4c57cd6e6ce51e8335b15239054995b8f3c31b69f4293c"),
        ("kk_kz", "test", 787758, "16b40d7a3c0a3be482f3bea949d3297ccdd13e42042683ffd298f8b6e469508a"),
    ]],
    # FSD50K eval (тест B; шумы и негативы для A). md5 — из записи Zenodo.
    *[Asset("fsd50k", f"{ZENODO_FSD50K}/{name}?download=1", f"data/fsd50k/raw/{name}", size,
            md5=md5) for name, size, md5 in [
        ("FSD50K.eval_audio.zip", 3037675767, "6fa47636c3a3ad5c7dfeba99f2637982"),
        ("FSD50K.eval_audio.z01", 3221225472, "3090670eaeecc013ca1ff84fe4442aeb"),
        ("FSD50K.ground_truth.zip", 334701, "ca27382c195e37d2269c4c866dd73485"),
        ("FSD50K.metadata.zip", 6700838, "b9ea0c829a411c1d42adb9da539ed237"),
        ("FSD50K.doc.zip", 6984, "3516162b82dc2945d3e7feba0904e800"),
    ]],
]

PROFILES = {
    "detectors": ["panns", "vosk"],
    "kws": ["vosk", "piper", "kscd", "fleurs", "fsd50k"],
    "all": ["panns", "vosk", "piper", "kscd", "fleurs", "fsd50k"],
}


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def verify(asset: Asset, path: Path) -> list[str]:
    """Список несовпадений размера и хешей (пустой — файл тот самый)."""
    problems = []
    if path.stat().st_size != asset.size:
        problems.append(f"размер {path.stat().st_size}, ожидался {asset.size}")
    if asset.sha256 and (got := file_hash(path, "sha256")) != asset.sha256:
        problems.append(f"SHA-256 {got}, ожидался {asset.sha256}")
    if asset.md5 and (got := file_hash(path, "md5")) != asset.md5:
        problems.append(f"md5 {got}, ожидался {asset.md5}")
    return problems


def is_present(asset: Asset, root: Path = ROOT) -> bool:
    """Распакованные архивы считаются на месте, если папка распаковки не пуста."""
    if asset.unzip_to:
        name = Path(asset.dest).stem
        target = root / asset.unzip_to
        return (target / name).is_dir() or (asset.group == "kscd" and any(target.glob("*")))
    return (root / asset.dest).is_file()


def select(profile: str) -> list[Asset]:
    return [a for a in ASSETS if a.group in PROFILES[profile]]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.replace(dst)


def fetch(assets: Iterable[Asset], root: Path = ROOT, max_bytes: int | None = None) -> None:
    todo = [a for a in assets if not is_present(a, root)]
    total = sum(a.size for a in todo)
    print(f"нужно скачать {len(todo)} файлов, {total / 1e9:.2f} ГБ")
    if max_bytes is not None and total > max_bytes:
        raise SystemExit(f"больше лимита {max_bytes / 1e9:.2f} ГБ — уберите группы или поднимите --max-gb")
    for a in todo:
        dst = root / a.dest
        print(f"скачиваю {a.dest} ({a.size / 1e6:.0f} МБ)", flush=True)
        _download(a.url, dst)
        problems = verify(a, dst)
        if problems:
            dst.unlink()
            raise SystemExit(f"{a.dest}: {'; '.join(problems)} — файл удалён")
        if a.unzip_to:
            with zipfile.ZipFile(dst) as z:
                z.extractall(root / a.unzip_to)
            dst.unlink()
        print(f"  ок: хеши совпали{', распаковано' if a.unzip_to else ''}", flush=True)


def prepare_fsd50k(negatives: int = 3000, seed: int = 0) -> Path:
    """Как в S11a: md5, склейка и проверка архива, распаковка, манифест, перевод в 16 кГц."""
    import json

    import soundfile as sf

    from qat.data import fsd50k as F

    expected = F.RAW_DIR / "expected.md5"
    expected.write_text("".join(f"{a.md5}  {Path(a.dest).name}\n" for a in ASSETS
                                if a.group == "fsd50k"))
    if problems := F.verify_md5(F.RAW_DIR, expected):
        raise SystemExit("FSD50K: " + "; ".join(problems))
    for name in ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip", "FSD50K.doc.zip"):
        with zipfile.ZipFile(F.RAW_DIR / name) as z:
            z.extractall(F.EXTRACTED_DIR)
    if not (F.EXTRACTED_DIR / "FSD50K.eval_audio").is_dir():
        F.extract_eval_audio(F.RAW_DIR, F.EXTRACTED_DIR)
        (F.RAW_DIR / "FSD50K.eval_audio.full.zip").unlink(missing_ok=True)
    gt = F.read_ground_truth(F.EXTRACTED_DIR / "FSD50K.ground_truth" / "eval.csv")
    info = json.loads((F.EXTRACTED_DIR / "FSD50K.metadata" / "eval_clips_info_FSD50K.json")
                      .read_text())
    rows = F.sample_negatives(
        F.build_manifest(gt, uploaders={k: v["uploader"] for k, v in info.items()}),
        negatives, seed)
    manifest = F.PROCESSED_DIR / f"manifest_sed_neg{negatives}_seed{seed}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for r in rows:
        dst = F.PROCESSED_DIR / r["path"]
        if not dst.is_file() or sf.info(str(dst)).samplerate != 16000:
            F.to_16k_mono(F.EXTRACTED_DIR / "FSD50K.eval_audio" / r["path"], dst)
    print(f"FSD50K готов: {manifest} ({len(rows)} клипов)")
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка моделей и данных с проверкой хешей")
    parser.add_argument("action", choices=["fetch", "list", "prepare-fsd50k"])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="kws")
    parser.add_argument("--max-gb", type=float, default=None, help="не качать больше этого")
    args = parser.parse_args()
    assets = select(args.profile)
    if args.action == "prepare-fsd50k":
        prepare_fsd50k()
        return
    if args.action == "list":
        for a in assets:
            mark = "есть" if is_present(a) else "нет "
            print(f"{mark} {a.group:7} {a.size / 1e6:9.1f} МБ  {a.dest}")
        return
    fetch(assets, max_bytes=None if args.max_gb is None else int(args.max_gb * 1e9))


if __name__ == "__main__":
    _main()

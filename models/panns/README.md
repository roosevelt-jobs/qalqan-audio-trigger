# PANNs Cnn14_16k (детектор B)

| Файл | Откуда | Лицензия |
|---|---|---|
| `Cnn14_16k_mAP=0.438.pth` (не в git, 359 МБ) | Zenodo, DOI [10.5281/zenodo.3987831](https://doi.org/10.5281/zenodo.3987831) | CC BY 4.0 |
| `class_labels_indices.csv` | github.com/qiuqiangkong/audioset_tagging_cnn, `metadata/`, коммит `d2f4b8c` (список классов Google AudioSet) | CC BY 4.0 (AudioSet ontology) |
| код модели `src/qat/detectors/panns_cnn14.py` | тот же репозиторий, `pytorch/models.py`, класс `Cnn14_16k` | MIT, см. `LICENSE-code.MIT` |

Авторы: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley.
«PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition», 2020.

Скачать веса:

```bash
curl -L -o "models/panns/Cnn14_16k_mAP=0.438.pth" "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
```

Проверить: `md5 -q` должен дать `362fc5ff18f1d6ad2f6d464b45893f2c` (md5 из записи Zenodo).
Детектор при загрузке сверяет SHA-256 файла.

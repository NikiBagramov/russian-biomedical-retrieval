# Русский биомедицинский поиск: RusBEIR + Qwen

**Никита Баграмов · Альфия Исхакова**  
Natural Language Processing & LLMs, autumn 2026.

Полный официальный rus-NFCorpus: 3633 документа, 323 test queries. Текст запроса и документа - основной вход.

RRF(BM25, Qwen embedding), top-20 reranking BGE/Qwen и dev-selected fusion. E5-small/large и BM25+BGE - реально запущенные конкуренты из семейства опубликованных RusBEIR baselines.

Все числа ниже - реальные локальные запуски. Выбор `selected` зафиксирован по dev до открытия test; порядок строк не используется для выбора модели.

| Метод | nDCG@10 |
|---|---:|
| bm25 | 0.3038 |
| bm25_bge_reranker512 | 0.3282 |
| bm25_raw | 0.2361 |
| bm25_twofield | 0.3034 |
| e5_large | 0.3057 |
| e5_large_bge_reranker512 | 0.3310 |
| e5_small | 0.2672 |
| qwen_embedding | 0.2960 |
| rrf | 0.3297 |
| rrf_bge_reranker512 | 0.3444 |
| rrf_qwen_reranker2048 | 0.3482 |
| rrf_qwen_reranker512 | 0.3396 |
| selected | 0.3525 |

Отчёт: [PDF](output/pdf/russian-biomedical-retrieval_report.pdf) · [текст](report/report.md) · [notebook](reproduce.ipynb).

## Что проверяется

Публичные предсказания и численные artifacts позволяют пересчитать метрики без скачивания моделей:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --verify
```

Точный эксперимент, версии, источники, ограничения и абляции описаны в [PROTOCOL.md](PROTOCOL.md), [ADAPTATIONS.md](ADAPTATIONS.md) и PDF. `artifacts/results.json` содержит интервалы bootstrap (seed 10, 2000 повторов); `artifacts/selection.json` - dev trials. Случайные seeds начинаются с 7; повторы обучения не выбираются по test.

## Полное воспроизведение

```bash
python scripts/run_pipeline.py --reproduce --download --report
```

Команда создаёт новый каталог `runs/reproduce_*`, скачивает только pinned benchmark files и открытые pretrained weights, проверяет SHA256 и выполняет обучение/inference/оценку заново. Исходные зафиксированные результаты не перезаписываются. Нужны Python 3.10-3.12 и CUDA GPU с поддержкой BF16. Измеренный запуск выполнен на RTX 3060 Laptop 6GB; T4/FP16 не проверялись. Полный запуск занимает часы, notebook по умолчанию только проверяет опубликованные artifacts. Конкретные wall-clock значения и версии находятся в manifests; stable PyTorch 2.6 в requirements - переносимая среда, фактически использовалась указанная в environment.json development CUDA-сборка.

Для создания PDF нужен ReportLab и шрифт Times New Roman на Windows либо DejaVu Serif на Linux. Веса и сырые benchmark-файлы в Git не включены; публичные источники и точные ревизии записаны в `artifacts/*provenance.json`.

## Ограничения

CPU BM25s отличается от Elasticsearch статьи. BGE 512-token cap взят из pinned source; статья указывает другую длину. Финальная композиция выбирает веса на dev и не является строго zero-shot. Глобальное SotA не доказано.

Опубликованные paper/model-card значения в PDF отделены от наших повторов. Лучший результат этой таблицы не объявляется мировым SotA. Runtime отдельных этапов мог измеряться одновременно с другим независимым GPU экспериментом, поэтому он не является контролируемым сравнением скорости систем.

## Источники

- https://arxiv.org/abs/2504.12879
- https://arxiv.org/abs/2402.05672
- https://arxiv.org/abs/2506.05176

## Сдача

В поле «Решение»: `https://github.com/NikiBagramov/russian-biomedical-retrieval`. Приложить PDF из `output/pdf/`. Покрытие критериев и условия бонусов: [SUBMISSION.md](SUBMISSION.md).

# Русский биомедицинский поиск: обновлённый эксперимент

**Никита Баграмов · Альфия Исхакова**  
Natural Language Processing & LLMs, autumn 2026.

Полный официальный **RusBEIR rus-NFCorpus**: 3633 документа, 324 dev и 323 test queries. Задача - ранжирование документов по текстовому запросу.

Новая система получила **0.3756 nDCG@10**, против опубликованного **0.3483 BM25+BGE**: наблюдаемый относительный прирост **7.83%**. По сравнению с прежней собственной системой **0.3525** разность **+0.0231**, парный 95% CI **[0.0121,0.0342]**. Сравнение с агрегатом статьи не является парным тестом; различия протоколов описаны ниже.

**Итоговый метод:** z-score fusion BM25/E5-large/Qwen3-Embedding-0.6B с dev-выбранными весами 0.25/0.5/0.5; top-100 кандидатов; Qwen3-Reranker-4B NF4/BF16 переоценивает top-20 с лимит 2048 токенов. Веса foundation моделей не дообучались на benchmark. Добавление прежнего ансамбля ухудшало dev, поэтому итог содержит один reranker.

| Метод | nDCG@10 | MAP@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Новая итоговая система | 0.3756 | 0.1456 | 0.1839 | 0.5964 |
| Новый первый этап поиска | 0.3533 | 0.1355 | 0.1722 | 0.5672 |
| Предыдущая система | 0.3525 | 0.1366 | 0.1743 | 0.5661 |
| BM25 local | 0.3038 | 0.1177 | 0.1541 | 0.4913 |
| BM25+BGE local512 | 0.3282 | 0.1264 | 0.1579 | 0.5411 |
| RRF+Qwen 0.6B | 0.3482 | 0.1332 | 0.1687 | 0.5556 |

Все строки таблицы - реальные локальные запуски на одном тесте. Результаты статьи показаны отдельно. Архитектуры Jina v3.5, Qwen 0.6B с большей глубиной и grounded bilingual query expansion проверялись на заранее закреплённых 64 dev-запросах; Qwen 4B прошёл также полный dev: 324 запроса. Полные положительные и отрицательные screens опубликованы.

Отчёт: [PDF](output/pdf/russian-biomedical-retrieval_report.pdf) · [текст](report/report.md) · [выполненный notebook](reproduce.ipynb).

## Проверка без моделей и исходных данных

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --verify
```

Проверяются первоначальные результаты и новая фаза: все dev ranks, все 15 финальных trials, все шесть систем на 323 test-запросах, основные/дополнительные метрики и bootstrap CI. Публичные rank/grade artifacts находятся в [experiments/phase2/artifacts](experiments/phase2/artifacts). Старые данные `artifacts/results.json` сохраняют фазу1; актуальная таблица - [phase2/results.csv](experiments/phase2/artifacts/results.csv).

## Полное независимое воспроизведение

```bash
python experiments/phase2/scripts/reproduce.py --plan
python scripts/run_pipeline.py --reproduce --download --report
```

Создаётся новый `runs/phase2_reproduce_*`. Сначала повторяется первоначальный baseline pipeline, затем screening и полный новый метод. Pinned weights скачиваются с официального Hugging Face, проверяются SHA256; при необходимости загрузка использует32 MiB HTTP ranges. Никаких платных API. Нужны BF16 CUDA GPU и часы вычислений; измеренный запуск использовал RTX 3060 Laptop 6 GB. T4/FP16 не проверялись. Notebook по умолчанию проверяет artifacts, а полный новый прогон выключен. При отличающемся выборе финальных весов повтор останавливается явно, поскольку опубликованная конфигурация 1/0/0 должна быть воспроизведена, а не заменена другой.

Стабильный PyTorch 2.6 в requirements - переносимая среда; фактически использовалась 2.6 development CUDA-сборка из environment manifests. Кросс-аппаратное битовое равенство не обещается. Полный повтор отдельных stages уже выполнен в зафиксированном эксперименте; быстрый notebook не выдаётся за повторное обучение/inference.

## Ограничения и сравнимость

Исходный test уже оценивался в фазе 1. Новая фаза подбирала конфигурацию только по dev и фиксировала SHA до нового test inference; это не совершенно новый скрытый benchmark. CPU BM25s отличается от Elasticsearch статьи, различаются token budgets и число параметров. Dev-adapted fusion не равен строго zero-shot protocol. В Jina listwise context разбит на небольшие блоки под 6 GB; результат этого адаптированного режима не переносится на полный 131K protocol. Неполные qrels и неизвестная экспозиция pretrained моделей ограничивают выводы. **Мировое SotA и автоматическое начисление бонусов не заявляются.**

Предыдущая опубликованная версия сохранена в commit [aab0c59](https://github.com/NikiBagramov/russian-biomedical-retrieval/tree/aab0c59a4a08736dac40117cda2e759a5818732c). [PROTOCOL.md](PROTOCOL.md), [ADAPTATIONS.md](ADAPTATIONS.md), [SUBMISSION.md](SUBMISSION.md) описывают обе фазы.

## Первичные источники

- [RusBEIR2025](https://arxiv.org/abs/2504.12879)
- [Расширенный RusBEIR](https://arxiv.org/abs/2511.05079)
- [Qwen3 Embedding/Reranking](https://arxiv.org/abs/2506.05176)
- [Jina reranker v3.5, 2026](https://arxiv.org/abs/2607.18152)
- [ThinkQE](https://arxiv.org/abs/2506.09260)
- [QA-Expand](https://arxiv.org/abs/2502.08557)

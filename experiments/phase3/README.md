# Текущий эксперимент: Giga + Qwen

Полный test: **0.39902 nDCG@10**, предыдущая собственная система **0.37558**, относительный прирост **6.24%**. Все 323 официальных test-запроса сохранены.

## 1. Код и запуск

- [scripts/reproduce.py](scripts/reproduce.py) — полный повтор в свежем каталоге; `--plan` показывает этапы.
- [scripts/giga_encode.py](scripts/giga_encode.py) — Giga embeddings без доступа к relevance.
- [scripts/pointwise_phase2.py](scripts/pointwise_phase2.py) — byte-exact ранее проверенный Qwen scorer.
- [scripts/ranking_math.py](scripts/ranking_math.py) — правила согласования scores и рангов.
- [scripts/choose_phase3.py](scripts/choose_phase3.py) — выбор только среди полных dev-конфигураций.
- [scripts/verify_phase3.py](scripts/verify_phase3.py) — независимая проверка без моделей и текстов.

## 2. Подтверждение результатов

| Файл в artifacts/ | Что подтверждает |
|---|---|
| [final_results.json](artifacts/final_results.json) | Метрики, все per-query значения, bootstrap CI и цель +5% |
| [final_predictions.jsonl](artifacts/final_predictions.jsonl) | Ранги и grades двух этапов для всех test-запросов |
| [selection_phase3.json](artifacts/selection_phase3.json) | Все 124 полных dev-конфигурации и выбранный метод |
| [dev_candidate_evidence.jsonl](artifacts/dev_candidate_evidence.jsonl) | Кандидаты, исходные scores, Qwen probabilities и grades |
| [evaluation_lock.json](artifacts/evaluation_lock.json) | Конфигурация и SHA до test |
| [inference_parity.json](artifacts/inference_parity.json) | Совпадение scorer на 40 dev-парах, разница 0 |
| [public_evidence_manifest.json](artifacts/public_evidence_manifest.json) | SHA численных доказательств |
| [published_comparison.json](artifacts/published_comparison.json) | Источники внешних агрегатов и ограничения сравнения |

## 3. Как выбирался метод

1. Проверены более глубокие 50/100 на 64 dev и полный 50 старого candidate pool.
2. Giga обработал 3633 документа и 324 dev-запроса с двумя инструкциями; проверены 54 retrieval-комбинации.
3. Три лучших pool сравнивались сначала на 64 dev. Для полного top-20 двух других pool после первого понадобилось только 457 дополнительных пар, поэтому все три оценены на 324 dev.
4. Финальная сетка содержит 124 полные конфигурации. Неполные глубины исключены. Выбран `{'pool': 'giga_pool1_dev', 'depth': 20, 'tie': 'doc_id', 'mode': 'zlogit', 'weight': 1.0}`; dev nDCG@10 **0.37948**.
5. До test записаны lock и parity. После этого выполнены одни фиксированные предсказания и одна relevance evaluation.

Глубокие Giga50/100 были отложены по записанной до test ресурсной политике. Первоначальный protocol и amendments сохранены; адаптивное исследование не выдаётся за полностью preregistered. [history/](history/) содержит архив служебных контроллеров.

## 4. Границы воспроизводимости

Реально выполнены отдельные core stages и независимый пересчёт. Полный многочасовой wrapper проверен синтаксически и в `--plan`, но целиком повторно не запускался. При другом победителе на dev он остановится до test. Точный Torch development wheel и hardware записаны в [runtime_manifest.json](artifacts/runtime_manifest.json).

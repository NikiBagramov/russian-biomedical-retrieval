"""Keep the concise project overview aligned with the latest completed artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    old=read(ROOT/'experiments/phase2/artifacts/final_results.json')
    phase=3 if (ROOT/'experiments/phase3/artifacts/final_results.json').exists() else 2
    art=f'experiments/phase{phase}/artifacts';r=read(ROOT/art/'final_results.json')
    if phase==3:
        current=r['metrics'];previous=r['baseline_phase2'];retrieval=r['retrieval_metrics'];gain=r['relative_gain_vs_own'];ci=r['paired_bootstrap']['ci95']
        choice=read(ROOT/art/'evaluation_lock.json')['selection']
        method=f'''1. **Поиск кандидатов:** Giga-Embeddings-instruct кодирует запрос и документы; по косинусному сходству выбираются 100 документов. Для запроса используется biomedical-инструкция.
2. **Переоценка:** Qwen3-Reranker-4B читает запрос вместе с каждым из первых {choice['depth']} документов и выдаёт оценку релевантности.
3. **Итоговый порядок:** нормализованный logit оценки Qwen объединяется с исходным рангом Giga, вес поправки **{choice['weight']:g}**. Хвост списка сохраняет порядок первого этапа.

```text
Запрос → Giga → 100 кандидатов → Qwen для top-20 → итоговое ранжирование
```

Веса моделей сохранены. Инструкция и правило объединения выбраны по dev; код и конфигурация зафиксированы до оценки на test. Режим inference: NF4/BF16; Giga — до 1024 токенов, Qwen — до 2048.'''
    else:
        current=r['metrics']['phase2_selected'];previous=r['metrics']['phase1_selected'];retrieval=r['metrics']['phase2_retrieval'];gain=current['ndcg_cut_10']/previous['ndcg_cut_10']-1;ci=r['bootstrap']['comparisons']['phase1_selected']['ci95']
        method='''1. **Поиск кандидатов:** BM25, E5-large и Qwen3-Embedding-0.6B оценивают документы. Их оценки нормализуются через z-score и складываются с весами 0.25/0.5/0.5.
2. **Переоценка:** Qwen3-Reranker-4B читает запрос вместе с каждым из первых 20 документов и выдаёт оценку релевантности.
3. **Итоговый порядок:** первые 20 документов сортируются по оценке Qwen; хвост до 100 документов сохраняет исходный порядок.

```text
Запрос → BM25/E5/Qwen → 100 кандидатов → Qwen top-20 → итоговое ранжирование
```

Веса моделей сохранены. Параметры композиции выбраны по dev; код и конфигурация зафиксированы до оценки на test. Qwen работает в NF4/BF16 с ограничением 2048 токена.'''
    local=[('BM25',old['metrics']['bm25']),('BM25 + BGE, локальный cap 512',old['metrics']['bm25_bge512']),('Предыдущая собственная система',previous),('Первый этап текущего метода',retrieval),('**Наш итоговый метод**',current)]
    table='\n'.join('| '+name+' | '+' | '.join(f'{v[k]:.4f}' for k in ['ndcg_cut_10','map_cut_10','recall_10','mrr_10'])+' |' for name,v in local)
    latest=f'experiments/phase{phase}'
    giga_usage='Опубликованный ориентир; released encoder используется в нашем методе' if phase==3 else 'Опубликованный ориентир из RusBEIR-lite'
    tree_extra='│   ├── phase2/             # Предыдущий эксперимент\n│   └── phase3/             # Текущий метод: scripts/ и artifacts/' if phase==3 else '│   └── phase2/             # Текущий метод: scripts/ и artifacts/'
    text=f'''# Русскоязычный биомедицинский поиск

Никита Баграмов · Альфия Исхакова · NLP & LLMs, 2026.

## 1. Задача

Найти релевантные научные документы по короткому запросу на русском языке.

- **Вход:** текст запроса, например «влияние соли на давление».
- **Выход:** ранжированный список идентификаторов документов из фиксированного корпуса.
- **Основная метрика:** nDCG@10 — учитывает релевантность и позиции первых десяти результатов; выше — лучше.

## 2. Датасет

**RusBEIR rus-NFCorpus** — русская версия биомедицинского корпуса NFCorpus.

| Часть | Размер | Для чего используется |
|---|---:|---|
| Корпус | 3633 документа | Заголовки и тексты научных документов |
| Train | 2590 запросов | Официальная часть; градиентное обучение на ней не проводилось |
| Dev | 324 запроса | Выбор конфигурации |
| Test | 323 запроса | Итоговая оценка всех запросов |

Документы и запросы хранятся в JSONL, эталонная релевантность — в qrels TSV с оценками 1/2. Проверяется порядок найденных документов по этой разметке.

Источники: [тексты](https://huggingface.co/datasets/kaengreg/rus-nfcorpus), [qrels](https://huggingface.co/datasets/kaengreg/rus-nfcorpus-qrels). Версии и контрольные суммы закреплены в [data provenance](artifacts/data_provenance.json).

## 3. Конкуренты и базовые методы

| Подход | Что делает | Как использован |
|---|---|---|
| BM25 | Ищет по совпадениям слов | Локальный baseline на полном test |
| E5-small / E5-large | Ищет по сходству текстовых векторов | Локальные dense baselines |
| BGE reranker | Переоценивает пары «запрос — документ» | Локальные BM25+BGE и E5+BGE |
| Qwen embedding / reranker 0.6B | Семантический поиск и переоценка | Компоненты предыдущей собственной системы |
| Giga-Embeddings-instruct | Семантический поиск с инструкцией | {giga_usage} |
| Jina v3.5 и расширение запросов | Другие способы улучшить поиск | Только dev-screening; полный test для них не заявляется |

Опубликованный **BM25+BGE: 0.34830** — из [статьи RusBEIR](https://arxiv.org/html/2504.12879v1#S5). Более сильный найденный ориентир **Giga: 0.36834** — из [независимого RusBEIR-lite](https://huggingface.co/spaces/sergeyzh/rusBEIR-lite/blob/ef8a174bbc1c1d344066bc98fd22813196ae0ef6/results.jsonl). Эти чужие агрегаты отделены от наших локальных запусков.

Подробности сохранённых настроек и ресурсных адаптаций: [ADAPTATIONS.md](ADAPTATIONS.md).

## 4. Наше решение

{method}

Собственный вклад — доменная композиция готовых моделей, согласование оценок, сравнение вариантов и проверяемые результаты. Подробный алгоритм приведён в [отчёте](report/report.md).

## 5. Результаты

Все строки ниже получены локально на одних и тех же **323 test-запросах**.

| Метод | nDCG@10 ↑ | MAP@10 ↑ | Recall@10 ↑ | MRR@10 ↑ |
|---|---:|---:|---:|---:|
{table}

К предыдущей собственной системе: **{100*gain:+.2f}%** по nDCG@10; парный 95% интервал абсолютной разности **[{ci[0]:.4f}, {ci[1]:.4f}]**. К внешнему Giga 0.36834: наблюдаемая относительная разница **{100*(current['ndcg_cut_10']/.36834-1):+.2f}%**. Для внешнего сравнения парный интервал недоступен.

[Полные метрики и интервалы]({art}/final_results.json) · [Предсказания с оценками релевантности]({art}/final_predictions.jsonl).

## 6. Запуск и проверка

Быстрая проверка опубликованных метрик, без моделей и исходного корпуса:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --verify
```

Посмотреть этапы полного воспроизведения:

```bash
python {latest}/scripts/reproduce.py --plan
```

Полный новый прогон в отдельном каталоге:

```bash
python scripts/run_pipeline.py --reproduce --download --report
```

Нужен BF16-совместимый CUDA GPU; фактический запуск — RTX 3060 Laptop 6 GB. Полный эксперимент занимает часы. [Notebook](reproduce.ipynb) выполнен в быстром режиме проверки; полный inference включается отдельно.

## 7. Где что находится

```text
README.md                  # Задача, данные, методы, результаты и запуск
reproduce.ipynb             # Выполненная быстрая проверка
scripts/run_pipeline.py     # Общая точка входа
experiments/
{tree_extra}
artifacts/                 # Исходные baselines, provenance и рисунки
report/report.md           # Текст отчёта
output/pdf/                # PDF для сдачи
PROTOCOL.md                # Разбиения, выбор метода и правила оценки
ADAPTATIONS.md             # Детали моделей и отличия от чужих запусков
SUBMISSION.md              # Материалы и критерии сдачи
```

Текущий эксперимент: [{latest}/]({latest}/). Исходные данные, веса, рабочие кэши и логи в Git не включены.

## 8. Ограничения и материалы

- Этот test уже оценивался в предыдущих фазах. Новый выбор сделан по dev и зафиксирован до новой test-оценки.
- Один переводной домен, неполные qrels и подбор параметров по dev ограничивают обобщение.
- Точная конфигурация внешней строки Giga неизвестна; мировое SotA и бонус преподавателя не заявляются.

[PDF отчёта](output/pdf/russian-biomedical-retrieval_report.pdf) · [Текст отчёта](report/report.md) · [Протокол](PROTOCOL.md) · [Материалы для сдачи](SUBMISSION.md).
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8',newline='\n')
    print('README rebuilt from completed phase',phase,'lines',len(text.splitlines()))

if __name__=='__main__':main()

# Модели и адаптации

## 1. Итоговый метод

Giga-Embeddings-instruct с biomedical query instruction даёт top-100 по cosine similarity. Qwen3-Reranker-4B переоценивает top-20. Внутри этой головы к z-score(logit(Qwen score)) добавляется z-score(-исходный ранг Giga) с весом 1. Хвост 21-100 сохраняется. Веса моделей не дообучаются; выбор сделан по dev.

## 2. Что сохранено и что изменено

| Компонент | Сохранено | Адаптация |
|---|---|---|
| Giga | Pinned weights, released latent attention, 2048dims, L2 | Biomedical prompt, NF4double quant, BF16, 1024 tokens, batch2, SDPA text encoder |
| Qwen4 | Pinned weights, official yes/no template, protected prefix/suffix | NF4/BF16, 2048 tokens, batch2, top20; исходный ранг добавляется отдельным правилом |
| BM25 | Lucene, k1=1.2, b=.75, авторские processed fields | CPU BM25s вместо Elasticsearch статьи |
| E5 | Released prefixes, masked mean, L2 | Локальный 512-token inference |
| Qwen0.6 embedding/reranking | Official pooling/template | Доменная инструкция и budgets512/2048 |
| BGE baseline | Released pair scorer | Top20, 512 tokens по pinned wrapper; статья упоминает 2048 |
| Jina v3.5 | Released LBNL encoder/projector/query fusion | Только 64 dev, блоки 16 docs/4096 tokens вместо полного listwise context |
| Query expansion | Идея corpus-feedback expansion | Один greedy ru/en round от Qwen1.7B; это отдельная адаптация идеи ThinkQE/QA-Expand |

## 3. Ресурсы и отвергнутые варианты

На 6 GB GPU используется один inference-worker за раз. Giga полный corpus/dev encoding занял около 19 минут, peak allocated около 2855 MiB (2.79 GiB);18 из 3633 документов усечены. Увеличенный length-grouped batch Qwen изменил scores и ранги на 40 dev-парах, поэтому итог использует исходный batch2. Неудачные варианты и их реальные dev результаты сохранены.

## 4. Что означает сравнение

Наша система включает dev-selected инструкцию/постобработку и другую точность inference. Строка внешнего Giga не содержит точного revision/команды/query-level predictions. Чужой агрегат не выдаётся за собственный повтор; парный CI относится к нашим локальным predictions. Подробные результаты: experiments/phase3/ и PDF.

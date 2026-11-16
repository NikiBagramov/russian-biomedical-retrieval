# Протокол до оценки test

Полный официальный RusBEIR rus-NFCorpus: 3633 документа, 2590 train queries,
324 dev, 323 test. Ни один test query не подменяется пользовательской выборкой.
Train labels разрешены только для явно описанного обучения; методы этого
эксперимента используют готовые модели и выбирают настройки на dev. Test qrels
не читает ни генератор представлений, ни подбор fusion.

Публикация: RusBEIR (2025), arXiv:2504.12879. На этом же benchmark опубликованы
BM25, multilingual E5 и BM25+BGE reranker. Наш основной baseline BM25s Lucene
портирует текущую CPU реализацию из авторского репозитория. Это не заявление
о битовом совпадении с Elasticsearch из статьи: CPU baseline объединяет поля,
а Elasticsearch использует multi_match best_fields, tie_breaker=.5. Для него
также реализован отдельно обозначенный математический двухполевой контроль.
Публикуемые paper scores и наши измерения показываются раздельно.

Для sparse используются поставляемые авторами processed_title/processed_text
(лемматизация и stopwords уже применены); для dense — title + '\n' + text.
Абляция sparse raw text проверяет эффект нормализации. BM25 k1=1.2, b=.75,
IDF Lucene. Параметры фиксированы, не оптимизируются по test.

Готовые конкуренты E5-small/E5-large: query:/passage: prefixes, masked mean
pooling, L2 norm, 512 tokens. Qwen3-Embedding-0.6B: официальный query instruction,
last-token pooling, L2 norm, 512 tokens. BGE-reranker-v2-m3: top-20, исходный
авторский wrapper ограничивает каждый текст 8192 символами и пару 512 токенами.
Хотя статья упоминает 2048, основное воспроизведение следует зафиксированному
коду; ограничения раскрываются. FP32 для encoders, BF16 для rerankers на 6GB.

Наш подход: равновесный RRF(k=60) top-100 BM25 и Qwen embedding; далее
Qwen3-Reranker-0.6B с официальным yes/no шаблоном, контекст 512/2048 токенов.
Проверяются BM25+BGE, E5-large+BGE и RRF+BGE, RRF+Qwen (512/2048).
Финальный смешанный scorer задаётся суммой reciprocal ranks reranker BGE и
Qwen и исходного retrieval: веса Qwen/BGE перебираются 0,.25,.5,.75,1,
вес исходного RRF 0,.1,.25. Выбор — только dev nDCG@10. Публикуется и лучший
одиночный reranker, даже если fusion не помогает. Это dev-adapted метод,
а не та же строго zero-shot постановка, что у части baselines статьи.

Основная метрика — pytrec_eval nDCG@10 с исходными градуированными qrels
(в dev/test встречаются 1 и 2). Используются точные метрики TREC, линейный gain,
без подмены на sklearn exponential DCG. MAP@10, Recall@10/100, MRR@10
показываются дополнительно. Фикстуры сверяются с исходной функцией evaluator.
Ранжирования и per-query метрики сохраняются, ties детерминированы doc id.
После dev-выбора фиксируется evaluation_lock с SHA. Test открывается один раз;
2000 парных bootstrap повторов по queries, seed 10. Неопределённость и время
обработки приводятся независимо от знака улучшения. SotA не обещается.


# Продолжение, 16 ноября 2026

Начальный commit aab0c59 и его результаты сохранены. Цель нового поиска - относительные 5-15% от опубликованного .3483. В фазе1 test уже был оценён; повторное развитие раскрывается. Новая фаза использует только official dev для выбора: 64SHA256-ordered queries с seed-salt 7, затем полный dev: 324 запроса для перспективного варианта. Исходныйprotocol.json сохранён в experiments/phase2/artifacts.

Фиксированная 36-trial score grid: zscore/minmax; BM25 weights .25/.5/1, E5 weights 0/.5/1, Qwen weights .5/1. Нормализация независима для каждого запроса по всем 3633 документам. Выбраны zscore,.25/.5/.5. Получен top-100; Qwen 4B NF4 double quant/BF16, batch 2,лимит 2048 токенов переставляет top-20. Официальный yes/no template и защищённый suffix сохранены. Никаких новых train gradients.

Проверены Jina v3.5 listwise в блоках 4096 tokens / 16 docs, Qwen 0.6B top-50 на новом pool, однораундовое grounded ru/en query expansion от Qwen3-1.7B. Все эти варианты имеют только 64-query screening; числа не смешиваются с полным dev/test. Jina 5 embedding отдельно прошёл лишь compatibility smoke и не включён в performance comparisons или итоговый метод.

Финальная 15-trial сетка: коэффициент Qwen 4B: 1, коэффициент прежней системы 0/.25/.5/.75/1, коэффициент нового retrieval 0/.1/.25, RRF k=60. Выбраны 1/0/0. Перед test сохранён evaluation_lock с SHA кода и candidates; финальный inference adapter повторил 40 dev scores с максимальной ошибкой 0. После lock выполнены 323 test queries и одна relevance evaluation; дальнейшего подбора по этим test scores нет.

Bootstrap seed 10, 2000 парных query resamples. Published aggregate .3483 - численный ориентир; paired CI вычисляется только для локальных запусков. Публикуются все ranks/grades, dev trials, источники, masks и SHA для независимого пересчёта. Foundation weights/raw corpus в Git отсутствуют.

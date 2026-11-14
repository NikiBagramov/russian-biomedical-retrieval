# Адаптации опубликованных подходов

RRF(BM25, Qwen embedding), top-20 reranking BGE/Qwen и dev-selected fusion. E5-small/large и BM25+BGE - реально запущенные конкуренты из семейства опубликованных RusBEIR baselines.

CPU BM25s отличается от Elasticsearch статьи. BGE 512-token cap взят из pinned source; статья указывает другую длину. Финальная композиция выбирает веса на dev и не является строго zero-shot. Глобальное SotA не доказано.

Точные tokenizer, pooling, длины, targets, выбор thresholds/fusion и конфигурации обучения перечислены в PROTOCOL.md. Версии и SHA входят в provenance. В отчёте есть таблица «сохранено / адаптировано». Чужие scores не переименованы в наши запуски. Ни новые ручные аннотации, ни скрытые коммерческие API не заявляются.

# Адаптации и собственный вклад

Новая итоговая система сочетает три lexical/dense scores через dev-selected z-score fusion и применяет Qwen3-Reranker-4B NF4/BF16 к top20. Собственный вклад - состав/нормализация кандидатов, доменная интеграция, экспериментальная проверка архитектур и неопределённости; новые foundation weights не заявляются.

| Компонент | Сохранено | Адаптация |
|---|---|---|
| BM25 | Lucene,k1=1.2,b=.75,авторские processed поля | CPU BM25s вместо Elasticsearch статьи |
| E5/Qwen embeddings | Официальные prefixes/pooling/normalization/weights | Доменная инструкция Qwen,cap512 |
| BGE baseline | Выпущенный checkpoint,pair scoring | Top20,512tokens по pinned исходному wrapper |
| Qwen4B | Официальный yes/no template,pinned weights | NF4 double quant,BF16,batch2,2048tokens,top20 |
| Jina v3.5 screen | LBNL weights/projector,official query-embedding fusion | Ограниченные4096token/16docblocks,512tokens/document |
| ThinkQE/QA-Expand-inspired screen | Идея query expansion и corpus feedback | Один greedy round от открытого1.7B; ru/en JSON; это не точная paper reproduction |

Сравнение0.3756 с published.3483 не приравнивается к доказанному мировому SotA или одинаковому zero-shot compute budget. Все отрицательные screens сохранены, fulltest использован только после нового lock. Ранее опубликованный test фазы1 уже известен.

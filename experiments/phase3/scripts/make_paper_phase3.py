"""Generate the course report from completed, locked phase3 numerical artifacts."""
import argparse,sys,re
from pathlib import Path
from datetime import datetime
base=Path(__file__).resolve().parents[1]
if not (base/'project.json').exists():base=base.parents[1]
sys.path.insert(0,str(base/'scripts'))
from common import ROOT,read,dump
from paper_utils import Paper,fmt,interval

def typography(text):
    parts=re.split(r'(https?://\S+)',str(text))
    for i in range(0,len(parts),2):
        s=parts[i].replace('RTX3060Laptop6GB','RTX 3060 Laptop 6 GB').replace('16Nov2026','16 Nov 2026').replace('phase3test','phase3 test').replace('Qwen20','Qwen top-20').replace('Grade2','Grade 2').replace('grades1/2','grades 1/2').replace('Giga1','Giga 1')
        for a,b in [('nDCG10','nDCG@10'),('MAP10','MAP@10'),('MRR10','MRR@10')]:s=s.replace(a,b)
        s=re.sub(r'(?<=[0-9])(?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])(?=[0-9])',' ',s)
        s=re.sub(r'(?<=\d)(?=(?:queries|query|tokens|token|dev|test|unique|inference|bytes|rankings|screen|nDCG|GB|MiB|GiB|SHA)\b)',' ',s)
        s=re.sub(r'\b(Python|Torch|Transformers|bitsandbytes|CUDA|seed|revision|cap|batch|SHA)(?=\d)',r'\1 ',s)
        s=re.sub(r',(?=[A-Za-zА-Яа-яЁё0-9])',', ',s);s=re.sub(r':(?=\d)',': ',s);s=re.sub(r'al\.(?=20\d\d)','al. ',s);s=re.sub(r'pinned(?=\d)','pinned ',s)
        parts[i]=s
    return ''.join(parts)

def figures(a,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=ROOT/'artifacts/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.unicode_minus':False})
    names=['Previous system','Giga first stage','Selected system'];values=[r['baseline_phase2']['ndcg_cut_10'],r['retrieval_metrics']['ndcg_cut_10'],r['metrics']['ndcg_cut_10']]
    fig,ax=plt.subplots(figsize=(7.3,3));ax.barh(names,values,color=['#a6b6c5','#6e9dab','#244b67']);ax.axvline(.36834,color='#aa583d',linestyle='--',label='Published Giga .36834');ax.axvline(r['target_test_ndcg10'],color='#777777',linestyle=':',label='Own baseline x1.05')
    ax.set_xlabel('Test nDCG@10 (323 queries)');ax.set_xlim(0,max(values+[r['target_test_ndcg10']])+.04)
    for i,v in enumerate(values):ax.text(v+.002,i,f'{v:.4f}',va='center')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=8,borderaxespad=0);fig.tight_layout();fig.savefig(out/'phase3_main.png',dpi=190);plt.close(fig)
    ids=list(r['per_query']);delta=np.array([r['per_query'][q]['ndcg_cut_10']-r['baseline_phase2_per_query'][q]['ndcg_cut_10'] for q in ids]);fig,ax=plt.subplots(figsize=(7.3,2.6));ax.bar(np.arange(len(delta)),np.sort(delta),color=np.where(np.sort(delta)>=0,'#356b7a','#b56a55'));ax.axhline(0,color='#555555',linewidth=.7);ax.set_xlabel('All 323 test queries, sorted by difference');ax.set_ylabel('Delta nDCG@10');fig.tight_layout();fig.savefig(out/'phase3_regressions.png',dpi=190);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,default=ROOT/'experiments/phase3');args=ap.parse_args();a=args.experiment/'artifacts'
    r=read(a/'final_results.json');lock=read(a/'evaluation_lock.json');selection=read(a/'selection_phase3.json');audit=read(ROOT/'artifacts/audit.json');legacy=read(ROOT/'experiments/phase2/artifacts/final_results.json');stage=lock['first_stage'];config=lock['selection'];assert stage['kind']=='giga_fusion'
    assert r['lock_sha256'] and lock['dev_candidate_parity'] and lock['all324_dev_ranking_parity']
    figures(a,r);s=r['metrics'];old=r['baseline_phase2'];ci=r['paired_bootstrap']['ci95'];encoding=lock['encoding']
    method='Giga без дополнительного реранкера' if config['depth']==0 else f'Giga и Qwen3-Reranker-4B на top-{config["depth"]}'
    p=Paper('Русскоязычный биомедицинский поиск: доменная инструкция Giga, калибровка реранкера и проверка прироста')
    p.h('Abstract')
    p.p(f'На полном официальном rus-NFCorpus исследуется перенос готовых языковых моделей на поиск научных текстов: 3633 документа, 324 dev и 323 test-запроса. После предыдущей композиции BM25/E5/Qwen проведено сравнение глубины переоценки, Giga-Embeddings-instruct и сочетаний исходного ранга с Qwen yes/no scores. Все решения этой фазы выбирались по dev; перед test зафиксированы код, конфигурация, модели и входы. Итог: {method}, test nDCG@10 {fmt(s["ndcg_cut_10"])} против {fmt(old["ndcg_cut_10"])} у предыдущей собственной системы: {100*r["relative_gain_vs_own"]:+.2f}% относительно неё. Парный95% CI разности {interval(ci)}. Сильнейший найденный внешний агрегат .36834 взят из независимого RusBEIR-lite; сравнение с ним не является парным тестом и не доказывает мировой SotA.')
    p.p('Репозиторий: https://github.com/NikiBagramov/russian-biomedical-retrieval')
    p.h('1. Introduction')
    p.p('Вход - короткий русский текстовый запрос; выход - упорядоченный список идентификаторов научных документов. Ответы на медицинские вопросы не генерируются. Исследовательский вопрос: улучшает ли специализированный dense retriever состав кандидатов сильнее, чем увеличение глубины прежнего реранкера, и нужно ли сохранять его исходный порядок при переоценке LLM? Собственный вклад - доменная композиция, контролируемое сравнение и воспроизводимая оценка, без заявления новой foundation architecture или новых pretrained weights.')
    p.h('2. Related Work')
    p.p('RusBEIR [1,2] адаптирует BEIR для русского языка и сравнивает lexical/dense/reranking. В таблицах2025 на rus-NFCorpus BM25+BGE получил .3483 nDCG@10. E5 [3] использует query/document embeddings; Qwen3 [4] предоставляет instruction-aware embeddings и yes/no rerankers. Giga-Embeddings-instruct [5] использует released latent-attention pooling и2048-мерное представление. Мы сохраняем его веса и архитектуру, меняя query-инструкцию и ресурсный режим inference. Ранее проверенные Jina v3.5 и grounded bilingual expansion сохраняются как отрицательные dev-наблюдения фазы2.')
    p.p('Расширенная проверка16ноября2026 изучила24 NFCorpus-строки официального RusBEIR Space и46 независимого RusBEIR-lite [6]. Сильнейший найденный point estimate - Giga .36834. У реестров совпадают corpus/qrels/test; Lite исключает другие задачи, а не заявленную подвыборку NFCorpus. Строка Giga не содержит точного model revision, команды и query-level predictions. Метка verified источника не заменяет независимое воспроизведение. Наша NF4/1024 адаптация не выдаётся за точный повтор этой внешней строки.')
    p.table(['Внешний результат, отдельный запуск','nDCG@10','Статус'],[['BM25+BGE [1,2]','.34830','Статья2025'],['Snowflake + Qwen3-Reranker-8B [6]','.35099','Официальная таблица'],['Giga-Embeddings-3B-0826 [6]','.36375','Независимая таблица'],['Giga-Embeddings-instruct [6]','.36834','Сильнейший найденный агрегат']],[3.2,1,2])
    p.h('3. Model Description')
    p.p('Предыдущая система объединяет BM25/E5-large/Qwen3-Embedding-0.6B через per-query z-score по всему корпусу с dev-выбранными весами .25/.5/.5, сохраняет100 кандидатов и переоценивает top-20 с Qwen3-Reranker-4B NF4. Она остаётся основным собственным baseline. В фазе3 отдельно проверены глубины50/100 и правила согласования рангов, прежде чем заменять retriever.')
    p.p(f'Giga checkpoint закреплён revision2cf0fdc97194aaedf10ac0e6bf798834acd31042; фактический encoder содержит3449624576параметров. Документ представлен title+newline+text. Используется released latent attention, FP32 L2-нормализация и cosine similarity; сохраняется top-100. Режим - NF4 double quantization, BF16 compute, batch2 и cap1024 tokens, SDPA внутри text encoder. В полном dev-корпусе усечены{encoding["groups"]["corpus"]["truncated"]}из3633документов; dev-запросы не усечены. Это ресурсная адаптация к6GB GPU, не доказательство равенства полноточному режиму с большим cap.')
    p.p(f'Сравнивались query-инструкции generic и biomedical. Выбран вариант {stage["prompt"]}: {encoding["prompts"][stage["prompt"]]}. К документам инструкция не добавляется. Фиксированная first-stage сетка содержит54 варианта:2инструкции, веса BM25/E5/Qwen из0/.25/.5 при весе Giga1, z-score по всем3633документам. Итоговые веса в порядке BM25/E5/Qwen/Giga: {stage["weights"]}. Веса нейронных моделей не дообучались на benchmark.')
    p.p('Qwen сохраняет официальный system/user/assistant yes/no template и biomedical instruct; score получается softmax только двух yes/no logits последнего токена. Это не калиброванная вероятность. Prefix/suffix защищены при cap2048. Веса Qwen pinned22e683669bc0f0bd69640a1354a6d0aebcfeede5, NF4/BF16, batch2. Проверка увеличенного length-grouped batch на40реальных dev-парах изменила scores и ранги; поэтому этот режим отвергнут и не используется в итоговом inference.')
    p.p(f'Финальное правило: depth={config["depth"]}, mode={config["mode"]}, retrieval weight={config["weight"]}, tie={config["tie"]}. Для rank-RRF складываются1/(60+reranker_rank) и weight/(60+retrieval_rank). Для zlogit выполняется clip(p,1e-7,1-1e-7), logit и z-score внутри переоцениваемой головы; добавляется weight*z-score(-retrieval_rank). Порядок хвоста до100 сохраняется. В сравнение допускаются только конфигурации с оценёнными всеми требуемыми парами для324dev, без частичных средних.')
    p.h('4. Dataset')
    p.p('Используется полный официальный kaengreg/rus-nfcorpus: corpus revision66f87c5cb6e134ab138e9421290d077a85f17846; qrels revision2dc2e7fe5e08187db84eeb1da7c3a19d70508337. JSONL хранит идентификатор, заголовок, текст и processed-поля документа либо идентификатор и текст запроса. Qrels TSV содержит query-id,corpus-id,score. Существующие grades1/2 сохраняются; отсутствующие пары считаются0 при оценке, хотя могут быть неразмеченными.')
    p.table(['Split','Queries','Qrels','Grade2'],[[n,audit['splits'][n]['queries'],audit['splits'][n]['qrels'],audit['splits'][n]['grades'].get('2',0)] for n in ['train','dev','test']],[2,1,1.5,1])
    p.p('Все официальные query IDs сохранены. Train labels не использовались для градиентного обучения. Нормализованные тексты8dev и6test запросов встречаются в исходном train; этот факт раскрыт в audit. Новый корпус и дополнительная ручная разметка не заявляются. Экспозиция исходных текстов в pretraining любых моделей неизвестна.')
    p.h('5. Experiments')
    p.p('Фазы1/2 уже оценивали тот же test. Фаза3 использует только dev labels для подбора и не объявляется новым скрытым benchmark. Screening использует закреплённые64SHA-sorted dev query с seed-salt7. На полном324dev проверяется перспективная архитектура и три близких candidate pools. Порядок исследований адаптивный: первоначальные глубины, затем Giga, затем полный20 до более дорогих50/100. Изменения политики записаны до первой phase3test-оценки; исследование не выдаётся за полностью preregistered.')
    p.p('Сохранён один последовательный GPU-worker. Три полных top-20 Giga pools сравниваются до расширения глубины, поскольку дополнительные457unique pairs значительно дешевле тысяч пар top-50. Для продвижения глубины использован dev-эвристический запас7.5% к предыдущему dev и разность лучших50/20 внутри одного pool>.002. Эвристика не гарантирует выигрыш на test. Неполные depth-варианты исключаются из выбора, отрицательные screens и неудачная batch-parity сохраняются.')
    p.p(f'До phase3test сохранён lock с конфигурацией, SHA {len(lock["source_sha256"])} inference/selection source files, corpus/query/index, базовыми score matrices, Giga vectors и model revisions/checksums. Candidate builder точно повторяет324dev pools и все выбранные dev rankings. Test prediction читает только тексты и фиксированные scores; relevance читает отдельный evaluation script после готовых323rankings. Прежние pair scores могут переиспользоваться только для совпадающих моделей/шаблонов/пар; test cache отделён от dev.')
    p.p('Основная метрика - TREC/pytrec_eval nDCG@10 с линейным gain исходных grades. Дополнительно MAP@10, P@10, Recall@10/100, MRR@10. Все метрики усредняются по каждому официальному query. Парный bootstrap:2000перевыборок323queries, seed10, относительно сохранённых собственных предсказаний фазы2. Для внешнего Giga query-level CI недоступен. Независимый проверяющий пересчитывает публичные rank/grade/probability artifacts, все полные dev trials и147screen controls без моделей и текстов.')
    p.h('6. Results')
    dev_rows=[['Предыдущая итоговая система',fmt(selection['baseline_phase2_dev']['ndcg_cut_10'])],['Предыдущий pool, Qwen top-50',fmt(read(a/'result_qwen4_fusion_d50_l2048_full.json')['metrics']['ndcg_cut_10'])]]
    for name,label in [('giga_pool1_dev','Giga biomedical'),('giga_pool2_dev','Giga generic'),('giga_pool3_dev','Giga hybrid')]:
        dev_rows.append([label+', retrieval',fmt(read(a/('result_'+name+'.json'))['metrics']['ndcg_cut_10'])]);path=a/('result_adaptive_'+name+'_d20_full.json')
        if path.exists():dev_rows.append([label+', Qwen20 без поправки',fmt(read(path)['metrics']['ndcg_cut_10'])])
    dev_rows.append(['Выбранная конфигурация, полный dev',fmt(selection['metrics']['ndcg_cut_10'])]);p.table(['Полный dev,324queries','nDCG@10'],dev_rows,[4,1])
    local=[['BM25 local',legacy['metrics']['bm25']],['BM25+BGE local512',legacy['metrics']['bm25_bge512']],['Предыдущая итоговая система',old],['Новый retrieval pool',r['retrieval_metrics']],['Новая выбранная система',s]]
    p.table(['Полный test,323queries','nDCG10','MAP10','R@10','MRR10'],[[n]+[fmt(v[k]) for k in ['ndcg_cut_10','map_cut_10','recall_10','mrr_10']] for n,v in local],[2.8,1,1,1,1])
    p.figure('phase3_main.png','Рисунок1. Локальные test-результаты; внешняя цифра Giga показана отдельно как агрегатный ориентир.')
    p.p(f'Новая nDCG@10={fmt(s["ndcg_cut_10"])}: относительный прирост к собственной предыдущей системе {100*r["relative_gain_vs_own"]:+.2f}%, абсолютная разность{fmt(r["absolute_gain_vs_own"])}; парный95% CI{interval(ci)}. Предзаданный порог1.05x={fmt(r["target_test_ndcg10"],6)} {"достигнут" if r["target_5pct_met"] else "не достигнут"}. К внешнему Giga .36834 наблюдаемый относительный прирост{100*r["relative_gain_vs_published"]:+.2f}%; статистическая значимость этого внешнего сравнения не установлена.')
    p.table(['Система','Recall@100','P@10'],[['Предыдущая собственная',fmt(old['recall_100']),fmt(old['P_10'])],['Новая собственная',fmt(s['recall_100']),fmt(s['P_10'])],['Внешний Giga, агрегат','.3440','.27337']],[3,1,1])
    p.figure('phase3_regressions.png','Рисунок2. Разности по всем323queries относительно предыдущей собственной системы; средний результат не исключает отдельных регрессий.')
    p.blocks.append(dict(type='page'))
    p.h('6.1. Limitations and reproducibility')
    p.p('Один переводной домен,323test queries, неполные qrels, повторное использование test между фазами и многократный dev search ограничивают обобщение. Доменная инструкция, NF4, token caps, dev-selected fusion и больший encoder отличаются от чужих настроек. Поэтому результат не изолирует эффект единственной архитектуры и не объявляется строго равным zero-shot protocol или доказанным мировым SotA. Генерация правильных медицинских рекомендаций не проверяется.')
    p.p(f'Реальная среда: Python3.10.6, Torch2.6development+CUDA12.1, Transformers4.57.6, bitsandbytes0.48.2, RTX3060Laptop6GB. Giga full corpus/dev encoding занял около19минут и использовал{encoding["peak_gpu_mb"]:.0f}MiB ({encoding["peak_gpu_mb"]/1024:.2f}GiB) peak allocated GPU memory; это измерение одного этапа, не всего исследования и не сопоставимый speed benchmark. BF16-совместимый CUDA GPU необходим; T4/FP16 и побитовое равенство на другом hardware не проверялись. Raw data, caches и weights не публикуются в Git; pins/checksums и численные предсказания опубликованы.')
    p.p('Быстрая проверка восстанавливает метрики без neural inference. Полный replay создаёт изолированный каталог, повторяет baseline/phase2 либо принимает явный готовый phase2 cache, затем phase3. Wrapper проверен синтаксически и в --plan; отдельные реальные этапы выполнены, но полный многочасовой wrapper повторно целиком не запускался. Если dev выберет другую финальную конфигурацию, replay останавливается до нового phase3test и не подменяет опубликованный метод.')
    p.h('7. Conclusion')
    p.p(f'{method} получил{fmt(s["ndcg_cut_10"])}nDCG@10 на323test queries. Вывод об улучшении относится к измеренной retrieval-задаче и сопровождается парным CI{interval(ci)}, полным dev comparison и отрицательными controls. Собственный порог+5% {"достигнут" if r["target_5pct_met"] else "не достигнут"}; внешние агрегаты и ограничения сравнимости приведены отдельно.')
    p.references(['Kovalev et al.2025. Building Russian Benchmark for Evaluation of Information Retrieval Models. https://arxiv.org/abs/2504.12879','Kovalev et al.2025. Wikipedia-based Datasets in Russian Information Retrieval Benchmark RusBEIR. https://arxiv.org/abs/2511.05079','Wang et al.2024. Multilingual E5 Text Embeddings: A Technical Report. https://arxiv.org/abs/2402.05672','Zhang et al.2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. https://arxiv.org/abs/2506.05176','Giga-Embeddings-instruct model card and released architecture. https://huggingface.co/ai-sage/Giga-Embeddings-instruct','RusBEIR result snapshots,16Nov2026. Official: https://huggingface.co/spaces/kaengreg/rusBEIR ; independent Lite: https://huggingface.co/spaces/sergeyzh/rusBEIR-lite'])
    p.p('Код, полный запуск и численные проверки: experiments/phase3/, PROTOCOL.md и README.md.')
    for block in p.blocks:
        if 'text' in block:block['text']=typography(block['text'])
        if 'caption' in block:block['caption']=typography(block['caption'])
        if block['type']=='table':block['headers']=[typography(x) for x in block['headers']];block['rows']=[[typography(x) for x in row] for row in block['rows']]
    dump(ROOT/'report/paper.json',dict(title=p.title,date=datetime.now().strftime('%d.%m.%Y'),blocks=p.blocks))
    print('Phase3 report source generated from completed locked results',flush=True)

if __name__=='__main__':main()

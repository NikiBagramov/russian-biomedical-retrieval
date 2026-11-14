"""Official TREC scores; avoid substituting a different DCG gain function."""
import argparse,ast,logging
from common import ROOT,dump,sha
import numpy as np
import pytrec_eval

def evaluate(qrels,run):
    assert set(qrels)<=set(run),'Every official query must have a ranking'
    selected={q:{d:float(v) for d,v in run[q].items() if d!=q} for q in qrels}
    per=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10','map_cut.10','recall.10,100','P.10'}).evaluate(selected)
    assert set(per)==set(qrels)
    for q in per:
        ranked=sorted(selected[q],key=lambda d:(-selected[q][d],d))[:10]
        per[q]['mrr_10']=next((1/(i+1) for i,d in enumerate(ranked) if qrels[q].get(d,0)>0),0.)
    summary={k:float(np.mean([r[k] for r in per.values()])) for k in next(iter(per.values()))}
    return summary,per

def conformance(reference_file=None):
    qrels={'q':{'a':2,'b':1,'c':1}};run={'q':{'b':3.,'a':2.,'z':1.}}
    result,_=evaluate(qrels,run)
    expected=(1+2/np.log2(3))/(2+1/np.log2(3)+1/np.log2(4))
    assert abs(result['ndcg_cut_10']-expected)<1e-10
    record=dict(passed=True,ndcg=result['ndcg_cut_10'],fixture='Graded linear-gain TREC DCG',engine='pytrec_eval (pytrec-eval-terrier 0.5.11)')
    if reference_file:
        # Evaluate the inspected upstream function without importing unrelated
        # retrievers. The unlicensed upstream file is not redistributed.
        tree=ast.parse(reference_file.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EvaluateRetrieval');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        fn.decorator_list=[];fn.returns=None
        for arg in fn.args.args:arg.annotation=None
        module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace=dict(pytrec_eval=pytrec_eval,logger=logging.getLogger('reference'))
        exec(compile(module,'upstream_reference','exec'),namespace)
        official=namespace['evaluate'](qrels,run,[10,100])
        assert round(result['ndcg_cut_10'],5)==official[0]['NDCG@10']
        assert round(result['map_cut_10'],5)==official[1]['MAP@10']
        record.update(upstream_function_agreement=True,upstream_sha256=sha(reference_file))
    return record
if __name__=='__main__':
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--reference-file',type=Path);a=ap.parse_args();dump(ROOT/'artifacts/metric_conformance.json',conformance(a.reference_file));print('Official RusBEIR metric fixture passed')

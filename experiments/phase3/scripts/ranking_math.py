"""Pure phase-three ranking rules shared by selection and frozen inference."""
import numpy as np

def order(values):return sorted(values,key=lambda d:(-values[d],d))

def rerank(pool,scores,query,config):
    original=order(pool);depth=config['depth']
    if not depth:return {d:float(len(original)-i) for i,d in enumerate(original)}
    head=original[:depth];prior={d:i for i,d in enumerate(original)}
    values={d:scores[query+'\t'+d] for d in head};tie=config['tie']
    ranked=sorted(head,key=lambda d:(-values[d],prior[d] if tie=='retrieval' else d))
    weight=config['weight']
    if config['mode']=='rank_rrf':combined={d:1/(60+i+1)+weight/(60+prior[d]+1) for i,d in enumerate(ranked)}
    elif config['mode']=='zlogit':
        p=np.clip(np.array([values[d] for d in ranked],dtype=np.float64),1e-7,1-1e-7)
        x=np.log(p/(1-p));x=(x-x.mean())/max(x.std(),1e-12)
        y=np.array([-prior[d] for d in ranked],dtype=float);y=(y-y.mean())/max(y.std(),1e-12)
        combined={d:float(a+weight*b) for d,a,b in zip(ranked,x,y)}
    else:raise ValueError(config['mode'])
    head=sorted(head,key=lambda d:(-combined[d],prior[d] if tie=='retrieval' else d))
    return {d:float(len(original)-i) for i,d in enumerate(head+original[depth:])}

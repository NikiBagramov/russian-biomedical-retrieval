"""Single entry point; full reproduction creates a clean isolated experiment directory."""
import argparse,subprocess,sys,shutil,os
from datetime import datetime
from common import ROOT,read

def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],check=True,cwd=ROOT)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reproduce',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--download',action='store_true');ap.add_argument('--verify',action='store_true');ap.add_argument('--report',action='store_true');a=ap.parse_args()
    if a.reproduce:
        destination=ROOT/'runs'/('reproduce_'+datetime.now().strftime('%Y%m%d_%H%M%S'));destination.mkdir(parents=True)
        for folder in ['scripts','third_party']:
            if (ROOT/folder).exists():shutil.copytree(ROOT/folder,destination/folder,ignore=shutil.ignore_patterns('__pycache__'))
        for filename in ['project.json','PROTOCOL.md','requirements.txt']:shutil.copyfile(ROOT/filename,destination/filename)
        (destination/'artifacts').mkdir()
        for filename in ['data_provenance.json','model_provenance.json','spacy_provenance.json','upstream_code_provenance.json']:
            if (ROOT/'artifacts'/filename).exists():shutil.copyfile(ROOT/'artifacts'/filename,destination/'artifacts'/filename)
        if (ROOT/'data/raw').exists():shutil.copytree(ROOT/'data/raw',destination/'data/raw')
        (destination/'cache').mkdir();env=os.environ.copy();env.setdefault('NLP_SHARED_MODELS',str(ROOT/'cache/models'))
        command=[sys.executable,str(destination/'scripts/run_pipeline.py'),'--run']
        if a.download:command.append('--download')
        if a.report:command.append('--report')
        subprocess.run(command,check=True,env=env,cwd=destination);print('Reproduction artifacts:',destination);return
    if a.download:call('download.py','--data','--models')
    if a.run:
        if (ROOT/'artifacts/evaluation_lock.json').exists():raise RuntimeError('A frozen experiment already exists. Use --reproduce for a clean independent run.')
        slug=read(ROOT/'project.json')['slug'];(ROOT/'cache').mkdir(exist_ok=True);(ROOT/'tmp').mkdir(exist_ok=True)
        call('prepare.py');call('environment.py')
        if slug=='russian-paraphrase-semantics':
            call('baselines.py','--stage','lexical');call('baselines.py','--stage','cross');call('baselines.py','--stage','gpu');call('train_lora.py');call('choose.py');call('evaluate.py')
        elif slug=='russian-conservative-spellcheck':
            call('setup_metric.py');call('official_metric.py');call('spell_metric.py');call('train.py','--no-identity');call('train.py','--no-identity','--matched')
            for s in [7,8,9]:call('train.py','--seed',str(s))
            names=['identity','sage95','sage_large','ordinary_seed7','ordinary_matched_seed7','identity_seed7','identity_seed8','identity_seed9']
            for n in names:call('predict.py','--model',n)
            call('choose.py');call('evaluate.py')
            for n in names[1:]:call('predict.py','--model',n,'--clean-control')
            call('evaluate.py','--clean-only')
        else:
            call('metrics.py');call('sparse.py');call('neural.py');call('context_audit.py');call('choose.py');call('evaluate.py')
        call('verify_results.py');call('error_analysis.py')
    if a.verify:call('verify_results.py')
    if a.report:
        call('figures.py');call('make_paper.py');call('build_report.py')
    if not any(vars(a).values()):ap.print_help()
if __name__=='__main__':main()

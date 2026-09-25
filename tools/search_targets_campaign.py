"""Receipt-gated overnight target experiment, full tiny rehearsal then real games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from match_evidence import atomic_json,file_hash
from analyze_cpu_gpu import analyze_match,paired_difference
from generate_search_targets import BASE,VALUE,TEACHER

B2=ROOT/'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'
ARMS=('raw','backed','ranked')

def recovery_changes(previous,current):
    """Only receipt I/O and this orchestration may differ during recovery."""
    if set(previous)!=set(current):raise ValueError('Recovery source set changed')
    changed={p:dict(before=previous[p],after=current[p]) for p in current if previous[p]!=current[p]}
    allowed={str(ROOT/'src/match_evidence.py'),str(Path(__file__))}
    if not set(changed)<=allowed:raise ValueError('Recovery engine/data/training sources changed')
    return changed


def reuse_artifacts(folder,hashes,rehearsal):
    """Preserve trained weights/book, never import partial or selected games."""
    old=json.loads((folder/'manifest.json').read_text())
    if old['rehearsal']!=rehearsal:raise ValueError('Recovery run size differs')
    changes=recovery_changes(old['hashes'],hashes)
    prepared=folder/'prepared';prep=json.loads((prepared/'complete.json').read_text())
    if not prep['complete'] or prep['implementation']!=file_hash(ROOT/'tools/prepare_search_backed.py'):
        raise ValueError('Prepared recovery receipt invalid')
    if file_hash(prepared/'provenance.json')!=prep['provenance_hash']:raise ValueError('Recovery provenance changed')
    artifacts=[folder/'manifest.json',prepared/'complete.json',prepared/'provenance.json']
    for split in ('train','val'):
        p=prepared/(split+'.npz')
        if file_hash(p)!=prep['splits'][split]['hash']:raise ValueError('Recovery targets changed')
        artifacts.append(p)
    values={'unchanged':VALUE}
    for arm in ARMS:
        model=ROOT/'models/candidates'/('search_backed_'+('rehearsal_' if rehearsal else '')+arm+'_001')
        manifest=json.loads((model/'manifest.json').read_text());receipt=json.loads((model/'complete.json').read_text())
        expected=dict(arm=arm,epochs=2 if rehearsal else 12,steps_override=2 if rehearsal else None,
            implementation=file_hash(ROOT/'tools/train_search_backed.py'),data=file_hash(prepared/'complete.json'),
            runtime=file_hash(ROOT/'native/monster_native.pyd'),initial_binary=file_hash(VALUE),
            initialization=file_hash(VALUE.with_suffix('.pt')))
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('Recovered training recipe changed')
        value=model/f"epoch_{receipt['best_epoch']:03}.bin"
        if not receipt['complete'] or receipt['parity_max_error']>1e-5 or file_hash(value)!=receipt['model_sha256']:
            raise ValueError('Recovered checkpoint invalid')
        values[arm]=value;artifacts.extend([model/'manifest.json',model/'complete.json',value])
    book=folder/'opening_book.json'
    evidence=json.loads((folder/'development_unchanged/manifest.json').read_text())
    evidence_hashes={str(Path(p).resolve()):h for p,h in evidence['hashes'].items()}
    if file_hash(book)!=evidence_hashes[str(book)]:raise ValueError('Recovered opening book changed')
    artifacts.extend([book,folder/'development_unchanged/manifest.json'])
    return values,book,dict(source=str(folder),changes=changes,
        artifacts={str(p):file_hash(p) for p in artifacts},
        note='All game stages restarted; no interrupted-run games count toward final comparisons')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=BASE/'campaign');ap.add_argument('--rehearsal',action='store_true')
    ap.add_argument('--pilot',type=Path,default=BASE/'pilot_v2')
    ap.add_argument('--rehearsal-receipt',type=Path,default=BASE/'rehearsal/summary.json')
    ap.add_argument('--reuse-from',type=Path,help='Validated completed preparation/models/book; never partial games')
    args=ap.parse_args();out=args.out
    pilot=json.loads((args.pilot/'complete.json').read_text())
    pmanifest=json.loads((args.pilot/'manifest.json').read_text())
    if not pilot['complete'] or any(file_hash(p)!=h for p,h in pmanifest['hashes'].items()):
        raise ValueError('Pilot missing, failed or stale')
    parity=json.loads((BASE/'default_parity.json').read_text())
    if not parity['complete'] or parity['runtime']!=file_hash(ROOT/'native/monster_native.pyd'):
        raise ValueError('Default search parity stale or failed')
    files=list((ROOT/'native/src').glob('*.rs'))+list((ROOT/'src').glob('*.py'))+[
        ROOT/'native/monster_native.pyd',VALUE,VALUE.with_suffix('.pt'),TEACHER,B2,Path(__file__),
        ROOT/'tools/generate_search_targets.py',ROOT/'tools/prepare_search_backed.py',
        ROOT/'tools/train_search_backed.py',ROOT/'tools/train_search_value.py',ROOT/'tools/train_search_leaves.py',
        ROOT/'tools/search_cpu_gpu_match.py',ROOT/'tools/analyze_cpu_gpu.py',ROOT/'tools/analyze_search_first.py',
        ROOT/'tools/audit_search_first_games.py',ROOT/'tools/make_book.py',ROOT/'tools/search_first_match.py',
        ROOT/'tools/stateful_generation.py',ROOT/'tools/search_leaf_audit.py',ROOT/'tools/match.py',
        args.pilot/'complete.json',args.pilot/'manifest.json',BASE/'default_parity.json']
    hashes={str(p):file_hash(p) for p in files}
    if not args.rehearsal:
        rehearsal=json.loads(args.rehearsal_receipt.read_text())
        if not rehearsal['complete'] or not rehearsal['rehearsal'] or rehearsal['hashes']!=hashes:
            raise ValueError('Full chain rehearsal missing/failed/stale')
    out.mkdir(parents=True,exist_ok=False);started=time.time();reports={};stage_times={}
    projected=pilot['seconds']*1280/pilot['roots'];train_roots,val_roots=(512,128) if projected>7200 else (1024,256)
    if projected>14400:raise ValueError('Pilot too slow even at reduced predeclared size')
    atomic_json(out/'manifest.json',dict(hashes=hashes,rehearsal=args.rehearsal,started=started,
        reuse_from=str(args.reuse_from) if args.reuse_from else None,
        train_roots=train_roots,val_roots=val_roots,pilot_projection_seconds=projected,
        hypothesis='same architecture/search/data, change raw versus searched/ranked targets',promotion=False,
        development='24 games each at300ms, book0..11',confirmation='32games each at2s, book12..27',
        nomination='best trained arm by development overall, tie raw then backed; unchanged also compared',
        secondary='24B2games@2s book28..39;12selfgames each@1s book40..51',
        book_seed=2143000000 if not args.rehearsal else 2142000000))
    def stage(name,command,hours=4):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Pinned campaign input changed')
        t=time.time();print('STAGE '+name,flush=True)
        with subprocess.Popen([sys.executable,*command],cwd=ROOT) as child:
            try:
                while True:
                    atomic_json(out/'status.json',dict(stage=name,complete=False,child_pid=child.pid,
                        heartbeat=time.time(),stage_started=t,elapsed=time.time()-started))
                    try:
                        code=child.wait(timeout=60)
                        if code:raise subprocess.CalledProcessError(code,command)
                        break
                    except subprocess.TimeoutExpired:
                        if time.time()-t>hours*3600:raise TimeoutError('Stage safety deadline: '+name)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=30)
        stage_times[name]=time.time()-t
    def audit(name):
        stage(name,['tools/audit_search_first_games.py','--root',str(out),'--out',str(out/(name+'.json'))])
        if not json.loads((out/(name+'.json')).read_text())['complete']:raise ValueError('Replay audit incomplete')
    try:
        stage('tests',['-m','pytest','tests','-q'])
        if args.reuse_from:
            values,book,recovery=reuse_artifacts(args.reuse_from.resolve(),hashes,args.rehearsal)
            atomic_json(out/'recovery.json',recovery)
            hashes.update(recovery['artifacts'])
            print('REUSED completed preparation, three trained checkpoints and frozen book',flush=True)
        elif args.rehearsal:corpus=args.pilot
        else:
            corpus=out/'corpus'
            stage('generate',['tools/generate_search_targets.py','--out',str(corpus),'--train-roots',str(train_roots),
                '--val-roots',str(val_roots),'--seed','913206'],hours=3)
        if not args.reuse_from:
            prepared=out/'prepared'
            stage('prepare',['tools/prepare_search_backed.py','--corpus',str(corpus),'--out',str(prepared)])
            values={'unchanged':VALUE}
            for arm in ARMS:
                folder=ROOT/'models/candidates'/('search_backed_'+('rehearsal_' if args.rehearsal else '')+arm+'_001')
                command=['tools/train_search_backed.py','--data',str(prepared),'--out',str(folder),'--arm',arm]
                if args.rehearsal:command+=['--epochs','2','--steps','2']
                stage('train_'+arm,command)
                receipt=json.loads((folder/'complete.json').read_text())
                value=folder/f"epoch_{receipt['best_epoch']:03}.bin"
                if not receipt['complete'] or file_hash(value)!=receipt['model_sha256']:raise ValueError('Training receipt invalid')
                if receipt['parity_max_error']>1e-5:raise ValueError('Export failed')
                values[arm]=value
        profiles={}
        for arm,value in values.items():
            profile=out/(arm+'_options.json');profiles[arm]=profile
            atomic_json(profile,dict(complete=True,runtime=file_hash(ROOT/'native/monster_native.pyd'),
                value=file_hash(value),options={},validation='native export parity and unchanged playing search'))
            hashes[str(value)]=file_hash(value);hashes[str(profile)]=file_hash(profile)
        # Keep the initial hash set separate: real-run authorization compares
        # the rehearsal's source inputs, not its disposable model output hashes.
        if not args.reuse_from:
            book=out/'opening_book.json'
            stage('opening_book',['tools/make_book.py','--model',str(TEACHER),'--model',str(B2),
                '--entries','8' if args.rehearsal else '64','--plies','16','--sims','128','--temperature','0.8',
                '--seed','2142000000' if args.rehearsal else '2143000000','--workers','1','--oversample','4',
                '--out',str(book)])
        hashes[str(book)]=file_hash(book)
        def play(name,arm,pairs,offset,seconds,reference=B2,selfplay=False):
            if args.rehearsal:
                pairs=1;offset={'development':0,'confirmation':2,'b2':4,'self':6}[name.split('_')[0]];seconds=.02
            command=['tools/search_cpu_gpu_match.py','--options',str(profiles[arm]),'--mode',
                'puct' if arm=='unchanged' and name=='self_gen47' else 'cpu','--out',str(out/name),
                '--value',str(values[arm]),'--reference',str(reference),'--book',str(book),
                '--pairs',str(pairs),'--offset',str(offset),'--seconds',str(seconds),'--node-limit','100000000',
                '--seed','2146000000']
            if selfplay:command+=['--selfplay']
            stage(name,command,hours=6)
            result=analyze_match(out/name)
            if not result['complete'] or result['proof_contradictions']:raise ValueError('Match evidence invalid')
            if result['games']!=pairs*(1 if selfplay else 2):raise ValueError('Wrong game count')
            if any(v['node_limit_interruptions'] for p in result['players'].values() for v in p.values()):
                raise ValueError('Node ceiling confounded clock')
            reports[name]=result;atomic_json(out/name/'analysis.json',result)
            atomic_json(out/'summary.json',dict(complete=False,rehearsal=args.rehearsal,stages=reports))
            return result['scores']['overall']['score']
        scores={arm:play('development_'+arm,arm,12,0,.3,TEACHER) for arm in values}
        nominee=max(ARMS,key=lambda arm:scores[arm]);overall=max(scores,key=scores.get)
        atomic_json(out/'decision.json',dict(best_trained=nominee,best_overall=overall,scores=scores,promotion=False))
        play('confirmation_unchanged','unchanged',16,12,2,TEACHER)
        play('confirmation_trained',nominee,16,12,2,TEACHER)
        delta=paired_difference(out/'confirmation_unchanged',out/'confirmation_trained')
        atomic_json(out/'paired_confirmation.json',delta)
        play('b2_trained',nominee,12,28,2,B2)
        play('self_trained',nominee,12,40,1,TEACHER,True)
        play('self_gen47','unchanged',12,40,1,TEACHER,True)
        audit('replay_audit')
        initial_hashes={str(p):hashes[str(p)] for p in files}
        atomic_json(out/'summary.json',dict(complete=True,rehearsal=args.rehearsal,hashes=initial_hashes,
            nominee=nominee,best_overall=overall,stages=reports,paired_confirmation=delta,
            stage_seconds=stage_times,seconds=time.time()-started,promotion=False,
            note='Teacher targets are bounded search estimates, not perfect play; no automatic release'))
        atomic_json(out/'status.json',dict(stage='complete',complete=True,heartbeat=time.time()))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),time=time.time()))
        atomic_json(out/'status.json',dict(stage='failed',complete=False,error=repr(exc),heartbeat=time.time()))
        raise

if __name__=='__main__':main()

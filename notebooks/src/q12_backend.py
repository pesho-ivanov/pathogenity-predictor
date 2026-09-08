"""GPU implementation of Q12; reuse the pinned Q11 backbone without modifying it."""

import hashlib
import io
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from . import q1, q1_full, q11, q11_backend as base, q12


def features_from_pooled(reference, difference):
    """Preserve mutation magnitude alongside Q11's direction features."""
    def rms(value):
        return value.square().mean(-1).clamp_min(q12.CONFIG['rms_floor']**2).sqrt()
    ref_rms, delta_rms = rms(reference), rms(difference)
    extra = torch.stack([delta_rms.log(), (delta_rms/ref_rms).clamp_min(q12.CONFIG['rms_floor']).log()], -1)
    return torch.cat([base.normalize_pair(reference, difference), extra], -1)


def features(model, tokenizer, rows, gradients=False):
    sequences = [s for pair in base.pairs_from(rows) for s in q11.crop_pair(*pair)]
    value = base.encode_batch(model, tokenizer, sequences, gradients)
    value = features_from_pooled(value[0::2], value[1::2]-value[0::2])
    q12.require(value.shape == (len(rows), q12.CONFIG['feature_dimension']) and torch.isfinite(value).all(), 'Invalid Q12 features')
    return value


def save_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO(); torch.save(base.cpu_state(state), stream)
    payload = stream.getvalue()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.partial')
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(hashlib.sha256(payload).hexdigest().encode()+b'\n'+payload)
            out.flush(); os.fsync(out.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_state(path, identity):
    with Path(path).open('rb') as stream:
        sha, payload = stream.readline(65).strip().decode(), stream.read()
    q12.require(hashlib.sha256(payload).hexdigest() == sha, 'Q12 checkpoint checksum mismatch')
    value = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    q12.require(value['format'] == 'q12-magnitude-v1' and value['identity'] == identity, 'Q12 checkpoint identity mismatch')
    def check(v):
        if isinstance(v, torch.Tensor):
            q12.require(torch.isfinite(v).all(), 'Non-finite checkpoint tensor')
        elif isinstance(v, dict):
            for item in v.values(): check(item)
        elif isinstance(v, (list, tuple)):
            for item in v: check(item)
    check(value)
    return value


def adapter_state(adapters):
    return base.cpu_state(dict(adapters.named_parameters()))


def restore_adapters(adapters, saved):
    q12.require(set(dict(adapters.named_parameters())) == set(saved), 'Adapter parameter names changed')
    with torch.no_grad():
        for name, p in adapters.named_parameters():
            p.copy_(saved[name]); p.grad = None


def preflight(model, tokenizer, rows):
    before = features(model, tokenizer, rows.iloc[:2])
    features(model, tokenizer, rows.iloc[2:4])
    repeated = features(model, tokenizer, rows.iloc[:2])
    q12.require(torch.equal(before, repeated), 'Q12 A/B/A repeatability failed')
    reverse = rows.iloc[:2].copy()
    for name in ['ref_sequence', 'alt_sequence']:
        reverse[name] = reverse[name].map(q1.reverse_complement)
    q12.require(torch.equal(before, features(model, tokenizer, reverse)), 'Q12 strand invariance failed')
    serial = torch.cat([features(model, tokenizer, rows.iloc[i:i+1]) for i in range(2)])
    q12.require(torch.allclose(before, serial, atol=2e-4, rtol=2e-4), 'Q12 batch equivalence failed')
    adapters = base.attach_lora(model)
    q12.require(torch.equal(before, features(model, tokenizer, rows.iloc[:2])), 'Zero adapters changed features')
    original, rng = adapter_state(adapters), base.rng_state()
    head = torch.nn.Linear(q12.CONFIG['feature_dimension'], 1, device='cuda', dtype=torch.float32)
    frozen = base.parameter_hash(model, frozen_only=True)
    optim = base.MasterAdamW(adapters, head, q12.CONFIG)
    optim.optimizer.param_groups[1]['lr'] = 0.
    optim.optimizer.param_groups[0]['weight_decay'] = 0.
    maxima = {name: 0. for name in original}
    initial = head(before).detach()
    try:
        for _ in range(8):
            value = features(model, tokenizer, rows.iloc[:2], gradients=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(value).flatten(), torch.ones(2,device='cuda'), reduction='sum')
            loss.backward()
            for name, p in adapters.named_parameters():
                q12.require(p.grad is not None and torch.isfinite(p.grad).all(), 'Invalid adapter gradient')
                maxima[name] = max(maxima[name], float(p.grad.abs().max()))
            optim.accumulate(2); optim.step()
        changed = head(features(model, tokenizer, rows.iloc[:2])).detach()
        q12.require(not torch.equal(initial, changed) and all(maxima.values()), 'Adapters did not affect a fixed-head prediction')
        old_head = base.cpu_state(head.state_dict())
        optim.optimizer.param_groups[1]['lr'] = q12.CONFIG['head_lr']
        value = features(model, tokenizer, rows.iloc[:2], gradients=True)
        torch.nn.functional.binary_cross_entropy_with_logits(head(value).flatten(), torch.ones(2,device='cuda'), reduction='sum').backward()
        optim.accumulate(2); optim.step()
        q12.require(any(not torch.equal(p.cpu(), old_head[n]) for n,p in head.named_parameters()), 'Head did not update')
    finally:
        restore_adapters(adapters, original); base.restore_rng(rng)
        del optim, head
    q12.require(torch.equal(before, features(model, tokenizer, rows.iloc[:2])), 'Probe adapters did not restore exactly')
    record = {'probe_keys': rows.iloc[:4].variant_key.tolist(), 'probe_split': 'train', 'A_B_A_exact': True,
        'strand_invariance_exact': True, 'batch_equivalence': True, 'zero_adapters_exact': True,
        'probe_restored': True, 'gradient_maxima': maxima, 'fixed_head_max_score_change': float((changed-initial).abs().max()),
        'frozen_sha256': frozen, 'conversion_audit': 'Verified Q11 source tensor audit and unchanged converted checkpoint; no upstream logit parity claim.'}
    q1.write_json(q12.OUTPUT/'preflight.json', record)
    print('Q12 training-only preflight passed.', flush=True)
    return adapters, original, frozen


def extract(model, tokenizer, rows, identity, budget):
    directory = q12.OUTPUT/'features'; directory.mkdir(exist_ok=True)
    path = q12.OUTPUT/'feature_manifest.json'
    record = q1.read_json(path) if path.exists() else {'identity': identity, 'batches': {}, 'seconds': 0.}
    q12.require(record['identity'] == identity, 'Frozen feature cache belongs to another experiment')
    arrays = {}
    for split, frame in rows.items():
        blocks = []
        for start in range(0, len(frame), 512):
            batch_rows = frame.iloc[start:start+512]
            name = f'{split}-{start:05d}.npz'; target = directory/name
            if name in record['batches']:
                q1.verify_file(target, record['batches'][name]['sha256'])
                with np.load(target, allow_pickle=False) as saved:
                    q12.require(saved['keys'].tolist() == batch_rows.variant_key.tolist(), 'Feature cache keys changed')
                    block = saved['features'].copy()
            else:
                q12.require(budget.can_train(30.), 'One-hour budget exhausted before frozen feature extraction completed')
                tick = time.perf_counter(); values = []
                for offset in range(0, len(batch_rows), q12.CONFIG['microbatch_variants']):
                    values.append(features(model, tokenizer, batch_rows.iloc[offset:offset+q12.CONFIG['microbatch_variants']]).detach().cpu().numpy())
                block = np.concatenate(values)
                temporary = target.with_suffix('.partial')
                with temporary.open('wb') as out:
                    np.savez(out, keys=batch_rows.variant_key.to_numpy(dtype=str), features=block)
                temporary.replace(target)
                elapsed = time.perf_counter()-tick
                record['batches'][name] = {'sha256': q1.digest_file(target), 'variants': len(block), 'seconds': elapsed}
                record['seconds'] += elapsed
                q1.write_json(path, record)
            q12.require(block.shape == (len(batch_rows), q12.CONFIG['feature_dimension']) and np.isfinite(block).all(), 'Invalid cached features')
            blocks.append(block)
            print(f'Frozen features {split}: {start+len(block)}/{len(frame)}; elapsed {budget.elapsed()/60:.1f} min', flush=True)
        arrays[split] = np.concatenate(blocks)
    return arrays


def new_head(dim):
    return torch.nn.Linear(dim, 1, device='cuda', dtype=torch.float32)


def score_head(head, values):
    with torch.no_grad():
        return head(values).flatten().cpu().numpy()


def fit_heads(raw, labels, identity, budget):
    path = q12.OUTPUT/'heads.pt'
    if path.exists():
        state = load_state(path, identity)
    else:
        mean, scale = q12.fit_scaler(raw['train'])
        standardized = {k: torch.from_numpy((x-mean)/scale).cuda() for k,x in raw.items()}
        targets = torch.tensor(labels['train'], device='cuda', dtype=torch.float32)
        weights = len(targets)/(2*np.bincount(labels['train'], minlength=2))
        sample_weights = torch.tensor(weights[labels['train']], dtype=torch.float32, device='cuda')
        states, scores, histories = {}, {}, {}
        orders = [np.random.default_rng(q12.CONFIG['seed']+epoch).permutation(len(targets)) for epoch in range(q12.CONFIG['head_epochs'])]
        for form in q12.CONFIG['forms']:
            torch.manual_seed(q12.CONFIG['seed'])
            indexes = torch.tensor(q12.columns(form), device='cuda')
            x = standardized['train'].index_select(1, indexes)
            v = standardized['validation'].index_select(1, indexes)
            head = new_head(len(indexes))
            optimizer = torch.optim.AdamW(head.parameters(), lr=q12.CONFIG['head_initial_lr'],
                betas=tuple(q12.CONFIG['betas']), eps=q12.CONFIG['epsilon'], weight_decay=q12.CONFIG['weight_decay'])
            history = []
            for epoch, order in enumerate(orders):
                q12.require(budget.can_train(5.), 'Budget exhausted during fixed head fitting')
                loss_sum = 0.
                for start in range(0, len(order), q12.CONFIG['head_batch']):
                    batch = torch.tensor(order[start:start+q12.CONFIG['head_batch']], device='cuda')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[batch]).flatten(), targets[batch], reduction='none')
                    loss = (loss*sample_weights[batch]).mean()
                    q12.require(torch.isfinite(loss), 'Non-finite head loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), q12.CONFIG['clip_grad'], error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    loss_sum += float(loss.detach())*len(batch)
                history.append(loss_sum/len(order))
            states[form] = base.cpu_state(head.state_dict())
            scores[form] = q12.metric(labels['validation'], score_head(head,v))
            histories[form] = history
            print(f'Head {form}: {scores[form]}; final training loss {history[-1]:.4f}', flush=True)
            del x,v,head,optimizer
        form = q12.select_form(scores)
        state = {'format': 'q12-magnitude-v1', 'identity': identity, 'mean': torch.from_numpy(mean),
            'scale': torch.from_numpy(scale), 'feature_form': form, 'heads': states,
            'scores': scores, 'loss_histories': histories, 'class_weights': weights.tolist(),
            'training_variants': len(labels['train']), 'head_epochs': q12.CONFIG['head_epochs']}
        save_state(path, state)
        del standardized
    q1.write_json(q12.OUTPUT/'head_results.json', {k: state[k] for k in
        ['feature_form','scores','loss_histories','class_weights','training_variants','head_epochs']}, frozen=True)
    print('Selected feature form:', state['feature_form'], flush=True)
    return state


def predict(model, tokenizer, head, mean, scale, indexes, rows, label='LoRA validation'):
    out = []
    for offset in range(0,len(rows),q12.CONFIG['microbatch_variants']):
        value = features(model,tokenizer,rows.iloc[offset:offset+q12.CONFIG['microbatch_variants']])
        out.append(score_head(head, ((value-mean)/scale).index_select(1,indexes)))
        if offset % 512 == 0 or offset+q12.CONFIG['microbatch_variants'] >= len(rows):
            print(f'{label}: {min(offset+q12.CONFIG["microbatch_variants"],len(rows))}/{len(rows)}',flush=True)
    return np.concatenate(out)


def make_checkpoint(identity, form, mean, scale, adapters, head, control, progress, optimizer=None, control_optimizer=None):
    state = {'format':'q12-magnitude-v1','identity':identity,'feature_form':form,'mean':mean,'scale':scale,
        'adapters':adapter_state(adapters),'head':base.cpu_state(head.state_dict()),
        'control':base.cpu_state(control.state_dict()),'progress':progress,'rng':base.rng_state()}
    if optimizer is not None:
        q12.require(optimizer.pending == 0, 'Cannot checkpoint inside a matched batch')
        state.update(masters=optimizer.masters, optimizer=optimizer.optimizer.state_dict(),
                     control_optimizer=control_optimizer.state_dict())
    return base.cpu_state(state)


def restore_training(state, adapters, head, control, optimizer, control_optimizer):
    restore_adapters(adapters,state['adapters'])
    head.load_state_dict(state['head']); control.load_state_dict(state['control'])
    q12.require(len(optimizer.masters)==len(state['masters']), 'Optimizer master count changed')
    with torch.no_grad():
        for p,master,saved in zip(optimizer.parameters,optimizer.masters,state['masters']):
            q12.require(torch.equal(p.cpu(),saved.to(p.dtype)), 'Master/deployed adapter mismatch')
            master.copy_(saved)
    optimizer.optimizer.load_state_dict(state['optimizer'])
    control_optimizer.load_state_dict(state['control_optimizer'])
    base.restore_rng(state['rng'])


def validate_cursor(progress, count):
    expected=np.random.default_rng(q12.CONFIG['seed']).permutation(count)
    q12.require(progress['order']==expected.tolist(), 'Continuation batch order changed')
    q12.require(progress['steps']==progress['control_steps'] and progress['offset']==progress['steps']*q12.CONFIG['microbatch_variants']
        and 0<=progress['offset']<=count, 'Matched continuation cursors differ')


def train(model,tokenizer,adapters,initial_adapters,frozen,rows,labels,raw,heads,identity,budget):
    form=heads['feature_form']; indexes=torch.tensor(q12.columns(form),device='cuda')
    mean,scale=heads['mean'].cuda(),heads['scale'].cuda()
    cached={k: ((torch.from_numpy(v).cuda()-mean)/scale).index_select(1,indexes) for k,v in raw.items()}
    head,control=new_head(len(indexes)),new_head(len(indexes))
    head.load_state_dict(heads['heads'][form]); control.load_state_dict(heads['heads'][form])
    optimizer=base.MasterAdamW(adapters,head,q12.CONFIG)
    control_optimizer=torch.optim.AdamW(control.parameters(),lr=q12.CONFIG['head_lr'],
        betas=tuple(q12.CONFIG['betas']),eps=q12.CONFIG['epsilon'],weight_decay=q12.CONFIG['weight_decay'])
    progress={'steps':0,'control_steps':0,'offset':0,'order':np.random.default_rng(q12.CONFIG['seed']).permutation(len(rows['train'])).tolist(),
        'evaluations':[],'loss_history':[],'loss_sum':0.,'control_loss_sum':0.,'bad_checks':0,
        'monitor_auroc':heads['scores'][form]['auroc'],'frozen_sha256':frozen,
        'best_lora':None,'best_control':None,'last_evaluated':0,'stop_reason':None}
    last=q12.OUTPUT/'last_checkpoint.pt'
    best_pair=q12.OUTPUT/'best_lora.pt'; best_control=q12.OUTPUT/'best_control.pt'
    if last.exists():
        state=load_state(last,identity); progress=state['progress']
        q12.require(state['feature_form']==form and torch.equal(state['mean'],heads['mean']) and torch.equal(state['scale'],heads['scale'])
                    and progress['frozen_sha256']==frozen, 'Continuation preprocessing or backbone changed')
        restore_training(state,adapters,head,control,optimizer,control_optimizer)
    else:
        scores=score_head(head,cached['validation']); measured=q12.metric(labels['validation'],scores)
        progress['best_lora']=measured; progress['best_control']=measured
        progress['evaluations'].append({'steps':0,'elapsed_seconds':budget.elapsed(),'lora':measured,'control':measured})
        snapshot=make_checkpoint(identity,form,mean,scale,adapters,head,control,progress)
        snapshot.update(lora_scores=torch.from_numpy(scores),control_scores=torch.from_numpy(scores))
        save_state(best_pair,snapshot); save_state(best_control,snapshot)
        save_state(last,make_checkpoint(identity,form,mean,scale,adapters,head,control,progress,optimizer,control_optimizer))
    validate_cursor(progress,len(rows['train']))
    weights=np.asarray(heads['class_weights'])
    targets=torch.tensor(labels['train'],dtype=torch.float32,device='cuda')
    sample_weights=torch.tensor(weights[labels['train']],dtype=torch.float32,device='cuda')
    next_seconds=5.

    def evaluate():
        score=predict(model,tokenizer,head,mean,scale,indexes,rows['validation'])
        control_score=score_head(control,cached['validation'])
        ma,mb=q12.metric(labels['validation'],score),q12.metric(labels['validation'],control_score)
        progress['evaluations'].append({'steps':progress['steps'],'elapsed_seconds':budget.elapsed(),'lora':ma,'control':mb})
        progress['last_evaluated']=progress['steps']
        stopped=q12.monitor(progress,ma['auroc'])
        a_new=q12.better(ma,progress['best_lora']); b_new=q12.better(mb,progress['best_control'])
        if a_new: progress['best_lora']=ma
        if b_new: progress['best_control']=mb
        snapshot=make_checkpoint(identity,form,mean,scale,adapters,head,control,progress)
        snapshot.update(lora_scores=torch.from_numpy(score),control_scores=torch.from_numpy(control_score))
        if a_new: save_state(best_pair,snapshot)
        if b_new: save_state(best_control,snapshot)
        print(f"Matched evaluation at {progress['steps']}: LoRA {ma}, control {mb}; elapsed {budget.elapsed()/60:.1f} min",flush=True)
        return stopped

    while progress['steps'] < min(q12.CONFIG['max_steps'],len(rows['train'])//q12.CONFIG['microbatch_variants']):
        if progress.get('stop_reason')=='early_stopping': break
        if not budget.can_train(next_seconds):
            progress['stop_reason']='time_budget';break
        tick=time.perf_counter();start=progress['offset'];size=q12.CONFIG['microbatch_variants']
        chosen=progress['order'][start:start+size]
        batch=torch.tensor(chosen,device='cuda')
        value=features(model,tokenizer,rows['train'].iloc[chosen],gradients=True)
        value=((value-mean)/scale).index_select(1,indexes)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(head(value).flatten(),targets[batch],reduction='none')
        loss=(loss*sample_weights[batch]).sum()
        q12.require(torch.isfinite(loss), 'Non-finite LoRA continuation loss')
        loss.backward();optimizer.accumulate(len(chosen));norm=optimizer.step()
        loss_c=torch.nn.functional.binary_cross_entropy_with_logits(control(cached['train'][batch]).flatten(),targets[batch],reduction='none')
        loss_c=(loss_c*sample_weights[batch]).mean()
        q12.require(torch.isfinite(loss_c),'Non-finite control loss')
        loss_c.backward()
        norm_c=torch.nn.utils.clip_grad_norm_(control.parameters(),q12.CONFIG['clip_grad'],error_if_nonfinite=True)
        control_optimizer.step();control_optimizer.zero_grad(set_to_none=True)
        progress['steps']+=1;progress['control_steps']+=1;progress['offset']+=len(chosen)
        progress['loss_sum']+=float(loss.detach());progress['control_loss_sum']+=float(loss_c.detach())*len(chosen)
        next_seconds=max(5.,(time.perf_counter()-tick)*1.5)
        if progress['steps']%q12.CONFIG['checkpoint_steps']==0:
            progress['loss_history'].append({'steps':progress['steps'],'mean_loss':progress['loss_sum']/progress['offset'],
                'control_mean_loss':progress['control_loss_sum']/progress['offset'],'gradient_norm':norm,'control_gradient_norm':float(norm_c),
                'elapsed_seconds':budget.elapsed()})
            print(f"Matched train: {progress['steps']}/{q12.CONFIG['max_steps']} updates; elapsed {budget.elapsed()/60:.1f} min",flush=True)
        if progress['steps']%q12.CONFIG['evaluation_steps']==0:
            if evaluate(): progress['stop_reason']='early_stopping'
        if progress['steps']%q12.CONFIG['checkpoint_steps']==0 or progress['stop_reason']:
            save_state(last,make_checkpoint(identity,form,mean,scale,adapters,head,control,progress,optimizer,control_optimizer))
    if progress['stop_reason'] is None: progress['stop_reason']='max_steps'
    q12.require(progress['steps']>0,'No budget remained for any matched LoRA updates')
    if progress['last_evaluated'] != progress['steps']: evaluate()
    q12.require(any(not torch.equal(p.cpu(),initial_adapters[name]) for name,p in adapters.named_parameters()),'Continuation did not update adapters')
    validate_cursor(progress,len(rows['train']))
    q12.require(base.parameter_hash(model,frozen_only=True)==frozen,'Original backbone changed during Q12')
    save_state(last,make_checkpoint(identity,form,mean,scale,adapters,head,control,progress,optimizer,control_optimizer))
    q1.write_json(q12.OUTPUT/'training_history.json',progress)
    del optimizer,control_optimizer,cached
    return progress,head,control,mean,scale,indexes


def finish(model,tokenizer,adapters,initial_adapters,rows,labels,groups,heads,progress,head,control,mean,scale,indexes,identity,budget):
    a=load_state(q12.OUTPUT/'best_lora.pt',identity)
    b=load_state(q12.OUTPUT/'best_control.pt',identity)
    scores={'lora':a['lora_scores'].numpy(),'matched_control':a['control_scores'].numpy(),'best_control':b['control_scores'].numpy()}
    promote=q12.promote(q12.metric(labels['validation'],scores['lora']),q12.metric(labels['validation'],scores['best_control']))
    selected_model='lora' if promote else 'frozen_control'
    selected=dict(a if promote else b)
    if not promote:
        selected['adapters']=initial_adapters;selected['head']=b['control']
    selected['selected_model']=selected_model
    save_state(q12.OUTPUT/'selected_model.pt',selected)
    # Erase learned tensors, then reload the chosen checkpoint for a fresh score check.
    with torch.no_grad():
        for p in list(adapters.parameters())+list(head.parameters()): p.zero_()
    selected=load_state(q12.OUTPUT/'selected_model.pt',identity)
    restore_adapters(adapters,selected['adapters']);head.load_state_dict(selected['head'])
    scores['selected']=scores['lora'] if promote else scores['best_control']
    n=q12.CONFIG['reload_variants']
    reloaded=predict(model,tokenizer,head,mean,scale,indexes,rows['validation'].iloc[:n],label='Selected checkpoint reload')
    q12.require(np.allclose(reloaded,scores['selected'][:n],atol=2e-4,rtol=2e-4),'Selected checkpoint does not reproduce predictions')
    frame=pd.DataFrame({'variant_key':rows['validation'].variant_key,'label':labels['validation'],'component':groups['validation'],**scores})
    frame.to_csv(q12.OUTPUT/'validation_predictions.csv',index=False)
    rows['train'][['variant_key']].to_csv(q12.OUTPUT/'training_membership.csv',index=False)
    metrics,paired=q12.intervals(labels['validation'],scores,groups['validation'])
    files=['protocol.json','input_checks.json','preflight.json','calibration.json','feature_manifest.json','heads.pt','head_results.json',
           'last_checkpoint.pt','best_lora.pt','best_control.pt','selected_model.pt','training_history.json','training_membership.csv','validation_predictions.csv','execution.json']
    result={'status':'complete','identity':identity,'scope':'sampled_validation','metrics':metrics,
        'paired_lora_minus_matched_control':paired,'feature_form':heads['feature_form'],'head_comparison':heads['scores'],
        'selected_model':selected_model,'validation_variants':len(frame),'training_variants':len(rows['train']),
        'full_validation_variants':17927,'training_pool_variants':46888,'optimizer_steps':progress['steps'],
        'best_lora_steps':a['progress']['steps'],'best_control_steps':b['progress']['steps'],
        'stop_reason':progress['stop_reason'],'seconds':budget.elapsed(),'within_one_hour':budget.elapsed()<=q12.CONFIG['total_seconds'],
        'frozen_unchanged':True,'matched_training_verified':True,'reloaded_predictions_verified':True,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'limitations':q12.LIMITATIONS,
        'artifacts':{name:q1.digest_file(q12.OUTPUT/name) for name in files}}
    q1.write_json(q12.OUTPUT/'metrics.json',result,frozen=True)
    print(f"Completed Q12: selected {selected_model}, {metrics['selected']}; {budget.elapsed()/60:.1f} min.",flush=True)


def explore():
    identity=q12.verify_protocol();budget=q12.Budget(q1.read_json(q12.OUTPUT/'execution.json')['started_utc'])
    rows,labels,groups,_=q12.inputs()
    model,tokenizer=base.load_backbone()
    try:
        adapters,initial,frozen=preflight(model,tokenizer,rows['train'])
        probe=rows['train'].iloc[:q12.CONFIG['microbatch_variants']]
        features(model,tokenizer,probe);torch.cuda.synchronize();tick=time.perf_counter()
        for _ in range(3):features(model,tokenizer,probe)
        torch.cuda.synchronize();budget.seconds_per_variant=(time.perf_counter()-tick)/(3*len(probe))
        q1.write_json(q12.OUTPUT/'calibration.json',{'probe_keys':probe.variant_key.tolist(),'probe_split':'train',
            'seconds_per_variant':budget.seconds_per_variant,'final_validation_and_report_reserve':budget.reserve()})
        print(f'Q12 calibration: {budget.seconds_per_variant:.4f} seconds/variant; reserve {budget.reserve()/60:.1f} min.',flush=True)
        raw=extract(model,tokenizer,rows,identity,budget)
        heads=fit_heads(raw,labels,identity,budget)
        progress,head,control,mean,scale,indexes=train(model,tokenizer,adapters,initial,frozen,rows,labels,raw,heads,identity,budget)
        finish(model,tokenizer,adapters,initial,rows,labels,groups,heads,progress,head,control,mean,scale,indexes,identity,budget)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def full_validation():
    identity=q12.verify_protocol();q12.verified_results()
    directory=q12.OUTPUT/'full';directory.mkdir(exist_ok=True)
    start=time.perf_counter()
    _,manifest,dna,indexes,labels,_=q11.verify_inputs()
    rows=dna.iloc[indexes['validation']]
    groups=manifest.loc[manifest.split.eq('validation'),'component'].to_numpy()
    state=load_state(q12.OUTPUT/'selected_model.pt',identity)
    model,tokenizer=base.load_backbone()
    try:
        adapters=base.attach_lora(model);restore_adapters(adapters,state['adapters'])
        frozen=base.parameter_hash(model,frozen_only=True)
        cols=torch.tensor(q12.columns(state['feature_form']),device='cuda')
        head=new_head(len(cols));head.load_state_dict(state['head'])
        mean,scale=state['mean'].cuda(),state['scale'].cuda()
        scores=predict(model,tokenizer,head,mean,scale,cols,rows,label='Full validation')
        with torch.no_grad():
            for p in list(adapters.parameters())+list(head.parameters()):p.zero_()
        state=load_state(q12.OUTPUT/'selected_model.pt',identity)
        restore_adapters(adapters,state['adapters']);head.load_state_dict(state['head'])
        n=q12.CONFIG['reload_variants']
        reload=predict(model,tokenizer,head,mean,scale,cols,rows.iloc[:n],label='Full checkpoint reload')
        q12.require(np.allclose(reload,scores[:n],atol=2e-4,rtol=2e-4),'Full checkpoint reload disagrees')
        q12.require(base.parameter_hash(model,frozen_only=True)==frozen,'Full inference changed backbone')
        pd.DataFrame({'variant_key':rows.variant_key,'label':labels['validation'],'component':groups,'selected':scores}).to_csv(directory/'validation_predictions.csv',index=False)
        metrics,_=q12.intervals(labels['validation'],{'selected':scores},groups)
        result={'status':'complete','identity':identity,'scope':'full_validation','metrics':metrics,
            'validation_variants':len(rows),'feature_form':state['feature_form'],'selected_model':state['selected_model'],
            'reloaded_predictions_verified':True,'seconds':time.perf_counter()-start,
            'selected_model_sha256':q1.digest_file(q12.OUTPUT/'selected_model.pt'),
            'exploration_sha256':q1.digest_file(q12.OUTPUT/'metrics.json'),
            'artifacts':{'validation_predictions.csv':q1.digest_file(directory/'validation_predictions.csv')},'limitations':q12.LIMITATIONS}
        q1.write_json(directory/'metrics.json',result,frozen=True)
        print('Completed separate Q12 full validation.',flush=True)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['explore','full-validation'])
    action=parser.parse_args().action
    {'explore':explore,'full-validation':full_validation}[action]()

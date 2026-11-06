"""Reproduce the listening summaries without changing or augmenting the export.
Usage: python analyse_listening.py /path/to/mushra-test-....json
A session is the analysis unit; repetitions are averaged within session.
"""
import argparse, hashlib, json, re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

ap=argparse.ArgumentParser(); ap.add_argument('export',type=Path)
ap.add_argument('--previous-export',type=Path)
ap.add_argument('--replacement-study',action='store_true',
                help='Analyse the author-identified replacement study, including its corrected N64 conditions.')
ap.add_argument('--corrections',type=Path,
                help='Explicit author-provided corrections; the original export and raw score output remain unchanged.')
ap.add_argument('--exclude-created-at',action='append',default=[],
                help='Exact createdAt timestamp of an author-identified non-study submission; repeat if needed.')
ap.add_argument('--exclusion-reason',default='Author-identified test submission')
args=ap.parse_args()
raw=args.export.read_bytes(); sessions=json.loads(raw); out=Path(__file__).resolve().parent
timestamps=[s['createdAt']['$date'] for s in sessions]
assert all(timestamps.count(t)==1 for t in args.exclude_created_at), 'Each explicit exclusion must identify exactly one record.'
corrections=json.loads(args.corrections.read_text())['corrections'] if args.corrections else []
assert all(timestamps.count(c['created_at'])==1 for c in corrections), 'Each correction must identify exactly one record.'
correction_by_time={c['created_at']:c for c in corrections}
assert len(correction_by_time)==len(corrections), 'Multiple corrections for the same record.'
rows=[]; corrected_rows=[]; info=[]; data={}; raw_data={}; patterns=defaultdict(list); reference_anchor_flags=[]
for i,s in enumerate(sessions,1):
    sid=f'S{i:02d}'; cells=defaultdict(list); raw_cells=defaultdict(list); refs=[]; page_counts=Counter(); filenames=defaultdict(set)
    correction=correction_by_time.get(s['createdAt']['$date'])
    for page in s['items']:
        if page['type']!=2: continue
        n=int(re.search(r'\d+',page['title']).group()); page_counts[n]+=1
        for m in page['example']['medias']:
            name=m['filename']; condition=re.search(r'N\d+_(.+)\.wav',name).group(1)
            original_value=float(m['value']); assert 0<=original_value<=100
            value=original_value
            if correction:
                assert original_value==correction['expected_original_value'], 'Original score differs from the correction specification.'
                value=float(correction['corrected_value']); assert 0<=value<=100
            raw_cells[n,condition].append(original_value)
            cells[n,condition].append(value); filenames[n,condition].add(name)
            row=dict(session=sid,N=n,condition=condition,repeat=page_counts[n],score=original_value,filename=name)
            rows.append(row)
            corrected_rows.append(dict(row,score=value,original_score=original_value,author_correction_applied=bool(correction)))
            if condition=='ref' and (args.replacement_study or n != 64): refs.append(value)
    values=[v for vs in cells.values() for v in vs]
    patterns[tuple((n,c,tuple(v)) for (n,c),v in sorted(cells.items()))].append(sid)
    for (n,c),vs in cells.items():
        if c == 'ref':
            anchors=cells.get((n,'anchor35'),[])
            for repeat,(reference,anchor) in enumerate(zip(vs,anchors),1):
                if reference < anchor:
                    reference_anchor_flags.append(dict(session=sid,N=n,repeat=repeat,reference=reference,anchor=anchor))
    all_zero=all(v==0 for v in values)
    original_all_zero=all(v==0 for vs in raw_cells.values() for v in vs)
    author_excluded=s['createdAt']['$date'] in args.exclude_created_at
    low_ref=sum(v<90 for v in refs)/len(refs)
    data[sid]={k:float(np.mean(v)) for k,v in cells.items()}
    raw_data[sid]={k:float(np.mean(v)) for k,v in raw_cells.items()}
    info.append(dict(session=sid,all_zero=all_zero,hidden_reference_below_90_fraction=low_ref,
                     original_all_zero=original_all_zero,author_correction_applied=bool(correction),
                     correction_reason=correction['reason'] if correction else None,
                     author_excluded=author_excluded,
                     author_exclusion_reason=args.exclusion_reason if author_excluded else None,
                     pages=dict(page_counts),scores=len(values),primary_included=not all_zero and not author_excluded,
                     reference_screen_included=not all_zero and not author_excluded and low_ref<=.15))
assert len(rows)==len(sessions)*35
assert all(x['pages']=={16:2,32:1,64:2,128:1} for x in info)
primary=[x['session'] for x in info if x['primary_included']]
all_sessions=list(data); screened=[x['session'] for x in info if x['reference_screen_included']]
without_corrections=[x['session'] for x in info if x['primary_included'] and not x['author_correction_applied']]
def array(ids,n,c,source=None): return np.array([(data if source is None else source)[s][n,c] for s in ids])
def ci_t(x):
    mean=float(np.mean(x)); half=float(stats.t.ppf(.975,len(x)-1)*stats.sem(x))
    return [mean-half,mean+half]
def exact_signflip(x):
    # Zero differences repeat every sign sum equally and can be omitted exactly.
    nonzero=np.asarray(x)[np.asarray(x)!=0]
    threshold=abs(np.sum(x))-len(x)*1e-12
    if threshold<=0: return 1.0
    # Meet in the middle: count the same 2^n assignments without storing them.
    def sums(values):
        result=np.array([0.0])
        for value in values: result=np.concatenate((result-value,result+value))
        return result
    split=len(nonzero)//2
    left=sums(nonzero[:split]); right=np.sort(sums(nonzero[split:]))
    lower=np.searchsorted(right,-threshold-left,side='right')
    upper=len(right)-np.searchsorted(right,threshold-left,side='left')
    return float(np.sum(lower+upper,dtype=np.int64)/(len(left)*len(right)))
def describe(ids,source=None):
    return [dict(N=n,condition=c,n=len(ids),mean=float(np.mean(x:=array(ids,n,c,source))),
                 sd=float(np.std(x,ddof=1)),median=float(np.median(x)),min=float(min(x)),max=float(max(x)),
                 mean_ci95_t=ci_t(x)) for n,c in sorted(data[ids[0]])]
selected={16:'r4L2',32:'r4L4',64:'r4L2',128:'L3'}
def contrasts(ids,source=None):
    result=[]
    for n in [16,32,64]:
        x=array(ids,n,selected[n],source)-array(ids,n,'ref',source)
        result.append(dict(N=n,contrast=selected[n]+' - ref',n=len(ids),mean=float(x.mean()),
                    ci95_t=ci_t(x),p_exact_two_sided_signflip=exact_signflip(x),differences=x.tolist()))
    ps=np.array([r['p_exact_two_sided_signflip'] for r in result]); order=np.argsort(ps)
    adjusted=np.maximum.accumulate([(len(ps)-j)*ps[idx] for j,idx in enumerate(order)])
    for j,idx in enumerate(order):result[idx]['p_holm_3']=float(min(1,adjusted[j]))
    return result
report=dict(input_sha256=hashlib.sha256(raw).hexdigest(),sessions_received=len(sessions),
 analysis_unit='session, with within-session repetitions averaged',
 assumption='Each exported session is treated as one response. The shared userId identifies the survey owner and does not verify listener identity. Any applied author-provided correction is listed explicitly below.',
 screening='Apply explicitly documented author corrections, then exclude any remaining all-zero sessions and explicitly excluded non-study submissions. Report raw scores and a sensitivity excluding all author-corrected submissions.',
 author_corrections=corrections,
 author_excluded_sessions=[x['session'] for x in info if x['author_excluded']],
 primary_n=len(primary),reference_screen_n=len(screened),session_checks=info,
 primary=describe(primary),all_sessions=describe(all_sessions,raw_data),reference_screen=describe(screened),
 without_author_corrections_n=len(without_corrections),without_author_corrections=describe(without_corrections),
 selected_reference_contrasts=contrasts(primary),all_sessions_contrasts=contrasts(all_sessions,raw_data),
 reference_screen_contrasts=contrasts(screened),
 method='Unadjusted 95% Student t intervals on session means or paired differences. Exact two-sided sign-flip tests enumerate all 2^n assignments under sign symmetry, with Holm adjustment across three selected-reference contrasts. Exploratory; no equivalence margin was prespecified.',
 issues=['The N32 anchor has an N64 filename and the same media source as N64; actual stimulus identity versus export error is unresolved.',
         'N128 has an explicit reference but no scored hidden reference.',
         'N16 and N64 occur twice; N32 and N128 once.',
         'No consent response or numerical age is present in this export.'])
report['author_reported_stimulus_error'] = {'N':64, 'treatment':'Exclude all N64 conditions from intended-design interpretation after the author reported a stimulus-file error. Preserve all original scores in this audit.', 'specific_file_mapping':'Unknown; no relabelling inferred.'}
report['interpretable_selected_reference_contrasts'] = report['selected_reference_contrasts'][:2]
report['manuscript_method'] = 'Descriptive means and standard deviations for selected operating points, with a stricter hidden-reference-screen sensitivity. Paired-t intervals and sign-flip tests remain in the analysis record; neither establishes equivalence.'
report['input_filename'] = args.export.name
report['identical_rating_patterns'] = [v for v in patterns.values() if len(v)>1]
report['identical_pattern_interpretation'] = 'Matching ratings do not establish duplicate respondents. Each submitted record is retained under the same screening rule; participant independence follows the author-reported count.'
report['reference_below_anchor_flags'] = reference_anchor_flags
if args.previous_export:
    prior_raw=args.previous_export.read_bytes(); prior=json.loads(prior_raw)
    record_key=lambda s: json.dumps(s['_id'],sort_keys=True)
    prior_by_id={record_key(s):s for s in prior}
    report['previous_export_check'] = {
        'sha256':hashlib.sha256(prior_raw).hexdigest(),
        'sessions':len(prior),
        'retained_record_ids':sum(record_key(s) in prior_by_id for s in sessions),
        'unchanged_records':sum(s==prior_by_id.get(record_key(s)) for s in sessions),
        'new_records':sum(record_key(s) not in prior_by_id for s in sessions)
    }
report['manuscript_scope'] = {
    'selection': 'During manuscript revision, restrict perceptual interpretation to the selected operating points in the benchmark table, with the reported N64 stimulus error excluded.',
    'operating_points': [{'N':16, 'b':4, 'L':2, 'condition':'r4L2'},
                         {'N':32, 'b':4, 'L':4, 'condition':'r4L4'},
                         {'N':128, 'b':2, 'L':3, 'condition':'L3'}],
    'other_administered_candidates': 'Retained in the full audit summaries, but excluded from manuscript comparisons and conclusions regardless of score. This includes legacy pairwise N16 L4 and N32 L10 and the unselected radix-4 depths.'
}
report['manuscript_operating_point_scores'] = [
    r for r in report['primary']
    if r['N'] in (16,32,128) and r['condition'] in (selected[r['N']], 'ref', 'anchor35')
]
if args.replacement_study:
    report['study_version'] = 'Author-identified replacement export from the corrected listening study'
    report['author_reported_stimulus_error'] = {'N':64,'treatment':'Prior exclusion superseded by the author identifying this export as the corrected study. Analyse as supplied; no scores replaced by this script.'}
    report['interpretable_selected_reference_contrasts'] = report['selected_reference_contrasts']
    report['screening'] = 'Exclude the all-zero record; average repeats within each response. No score replacement. Secondary hidden-reference screen uses all N16, N32, and N64 reference pages.'
    report['manuscript_scope'] = {'selection':'Report every administered configuration by page and condition. Benchmark-selected points remain identified separately.','operating_points':[{'N':n,'b':4 if n<128 else 2,'L':{16:2,32:4,64:2,128:3}[n],'condition':c} for n,c in selected.items()]}
    report['manuscript_operating_point_scores'] = [r for r in report['primary'] if r['condition'] in (selected[r['N']], 'ref', 'anchor35')]
    report['issues'] = ['The N32-page anchor carries an N64 filename; it is grouped by presentation page, not pooled into the N64 anchor mean.','N128 has an explicit reference but no scored hidden reference.','N16 and N64 each occur twice. All condition summaries still use one mean per response.','Participant count and corrected-stimulus status are author-reported; the export alone does not authenticate participant identity or audio provenance.']
(out/'listening_summary.json').write_text(json.dumps(report,indent=2)+'\n')
# No account IDs, session database IDs, names, contact details, or demographics are exported.
(out/'listening_scores_anonymised.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'listening_scores_corrected.json').write_text(json.dumps(corrected_rows,indent=2)+'\n')
print('N received / primary with documented corrections / stricter reference screen:',len(sessions),len(primary),len(screened))
for r in report['primary']:print(r['N'],r['condition'],f"{r['mean']:.2f}",f"SD {r['sd']:.2f}")
print('PAIRED',json.dumps(report['selected_reference_contrasts'],indent=2))
print('SCREENED',json.dumps(report['reference_screen_contrasts'],indent=2))

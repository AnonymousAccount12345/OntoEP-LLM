import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import torch
import copy
from collections import defaultdict
from collections import deque
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score

def parse_atomic(expr: str) -> list[str]:

    expr = expr.strip()
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        return [p.strip() for p in inner.split(" or ")]
    else:
        return [expr]

def is_subconcept_expr(child_expr: str,
                       parent_expr: str,
                       subclass_map: dict[str, set[str]]
                      ) -> bool:

    child_atoms  = parse_atomic(child_expr)
    parent_atoms = parse_atomic(parent_expr)

    for c in child_atoms:
        
        if not any(c == p or p in subclass_map.get(c, set())
                   for p in parent_atoms):
            return False
    return True

def log_counts(step_name, df_before, df_after):
    print(f"{step_name}: {len(df_before)} -> {len(df_after)} (dropped {len(df_before)-len(df_after)})")
    return df_after

# def filter_top_k_by_score(df, score_col='sim_to_orig', K=5):
#     parts = []
#     for name, grp in df.groupby(['subject','predicate','rtype']):
#         parts.append(grp.nlargest(K, score_col))
#     return pd.concat(parts)

from collections import deque

def compute_depth_and_height(children_map):

    parent_map = {}
    for p,ch in children_map.items():
        for c in ch:
            parent_map.setdefault(c,set()).add(p)
    all_nodes = set(children_map) | set(parent_map)

    roots = list(all_nodes - set(parent_map.keys()))
    if not roots: roots = list(all_nodes)

    # 1) depth via BFS
    depth = {r:0 for r in roots}
    q = deque(roots)
    while q:
        u = q.popleft()
        for v in children_map.get(u,[]):
            if v not in depth or depth[v] > depth[u]+1:
                depth[v] = depth[u]+1
                q.append(v)

    # 2) height via post-order DFS
    height = {n:0 for n in all_nodes}
    visited = set()
    def dfs(u):
        if u in visited: return height[u]
        visited.add(u)
        h = 0
        for v in children_map.get(u,[]):
            h = max(h, 1 + dfs(v))
        height[u] = h
        return h
    for n in all_nodes:
        dfs(n)

    # 3) relative depth = depth/(depth+height)
    rel_depth = {n: depth.get(n,0)/(depth.get(n,0)+height.get(n,0)) 
                 if depth.get(n,0)+height.get(n,0)>0 else 0
                 for n in all_nodes}
    return depth, height, rel_depth

#---------------------------------------------------

def load_data(dataset_path,model_type):
    with open(f'../{model_type}-Data/{dataset_path}/{dataset_path}_counter_reasoning.json','r',encoding='utf-8') as f:
        variants = json.load(f)


    ori_support = pd.read_csv(f'../{model_type}-Data/{dataset_path}/{dataset_path}_pre_counter_reasoning.csv')
    original_support = {
        ','.join([r.subject, r.predicate, r.rtype, r.object]): r.support_sentence
        for _, r in ori_support.iterrows()
    }
    with open(f'../{model_type}-Data/{dataset_path}/{dataset_path}_description.json','r',encoding='utf-8') as f:
        original_defs = json.load(f)
    with open(f'../{model_type}-Data/{dataset_path}/{dataset_path}_hierarchy.json','r',encoding='utf-8') as f:
        children_map = json.load(f)
    return variants, original_support, original_defs, children_map


# ---------------------------------------------------
def build_direct_parent_map(children_map):
    direct_parents = {}
    for parent, children in children_map.items():
        for child in children:
            direct_parents.setdefault(child, set()).add(parent)
    return direct_parents

def build_subclass_map(children_map):
    direct_parents = build_direct_parent_map(children_map)
    subclass_map = {}
    for cls in set(children_map) | set(direct_parents):
        ancestors = set()
        stack = list(direct_parents.get(cls, []))
        while stack:
            p = stack.pop()
            if p not in ancestors:
                ancestors.add(p)
                stack.extend(direct_parents.get(p, []))
        subclass_map[cls] = ancestors
    return direct_parents, subclass_map


# ---------------------------------------------------
def flatten_variants(variants, original_support, original_defs):
    records = []
    for key, varlist in variants.items():
        subj, pred, rtype, obj = key.split(',',3)
        combined = original_defs.get(subj,'') + ' ' + original_support.get(key,'')
        records.append({
            'constraint_key': key,
            'subject': subj, 'predicate': pred, 'rtype': rtype, 'object': obj,
            'operation': 'original', 'description': combined, 'combined': combined
        })
        for v in varlist:
            op, val = v['operation'], v.get('value','')
            new = {'subject': subj, 'predicate': pred, 'rtype': rtype, 'object': obj}
            if op == 'variant_rtype': new['rtype'] = val
            if op == 'prop_variant':    
                # continue
                new['predicate'] = val
            if op in ('generalize','specialize'): new['object'] = val
            records.append({
                'constraint_key': key,
                **new,
                'operation': op, 'description': v['description'], 'combined': combined
            })
    return pd.DataFrame(records)


MODEL = SentenceTransformer('all-mpnet-base-v2')
static_embeddings = {}
def add_embeddings(df):
    # compute only once for each unique text
    for col in ['description','combined']:
        for text in df[col].unique():
            if text not in static_embeddings:
                static_embeddings[text] = MODEL.encode(text, convert_to_numpy=True)
    df['desc_emb'] = df['description'].map(lambda x: static_embeddings[x])
    df['orig_emb'] = df['combined'].map(lambda x: static_embeddings[x])
    df['sim_to_orig'] = df.apply(lambda r: float(cosine_similarity([r['desc_emb']], [r['orig_emb']])[0,0]), axis=1)
    return df

def filter_parent_relevance(df, original_defs, direct_parents, threshold=0.6):

    keep = set(df.index)
    # parent_rows
    parent_rows = df[df.operation=='original']
    for idx, prow in parent_rows.iterrows():
        subj, pred, rtype = prow['subject'], prow['predicate'], prow['rtype']
        parent_emb = prow['desc_emb']
        # gather child definitions for this constraint
        child_sims = []
        for child in direct_parents.get(subj, []):
            child_def = original_defs.get(child, '')
            if not child_def:
                continue
            # encode child definition if not exists
            if child_def not in static_embeddings:
                static_embeddings[child_def] = MODEL.encode(child_def, convert_to_numpy=True)
            child_emb = static_embeddings[child_def]
            sim = float(cosine_similarity([parent_emb], [child_emb])[0,0])
            child_sims.append(sim)
        # average similarity
        if child_sims and np.mean(child_sims) < threshold:
            keep.discard(idx)
    return df.loc[sorted(keep)].reset_index(drop=True)


def compute_thresholds_grouped(df, direct_parents, quantile_strength=0.5, quantile_diff=0.5):
    thresholds = {}
    for name, grp in df.groupby(['subject','predicate','rtype']):

        min_strength = grp['sim_to_orig'].quantile(quantile_strength)
    # for name, grp in df.groupby(['subject','predicate','rtype']):
        subj, pred, rtype = name
        # min_strength = grp['sim_to_orig'].quantile(quantile_strength)
        diffs = []
        for _, row in grp.iterrows():
            sim_c = row['sim_to_orig']
            for p in direct_parents.get(subj, []):
                peers = df[(df.subject==p)&(df.predicate==pred)&(df.rtype==rtype)]
                for _, pr in peers.iterrows():
                    diffs.append(abs(pr['sim_to_orig'] - sim_c))
        diff_thresh = np.quantile(diffs, quantile_diff) if diffs else 0.2
        thresholds[name] = (min_strength, diff_thresh)
    return thresholds



def filter_remove_negative_rel(df, quantile_diff=0.75):
    keep, diffs = [], {}

    for key, grp in df.groupby('constraint_key'):
        sims = grp['sim_to_orig']
        sorted_sims = sims.sort_values(ascending=False).values
 
        if len(sorted_sims) > 1:
            diffs[key] = sorted_sims[0] - sorted_sims[1]

    global_diff = np.quantile(list(diffs.values()), quantile_diff) if diffs else 0.1

    for key, grp in df.groupby('constraint_key'):
        max_sim = grp['sim_to_orig'].max()
        diff_th = global_diff
        for _, row in grp.iterrows():
            op = row.operation
            sim = row.sim_to_orig
            if op not in ('remove','negative'):
                keep.append(row.name)
            else:

                if not (sim >= max_sim and (max_sim - sim) >= diff_th):
                    keep.append(row.name)
    return df.loc[sorted(set(keep))].reset_index(drop=True)


@lru_cache(None)
def is_subconcept_set(child_set, parent_set):
    return all(any(c==p or p in SUBCLASS_MAP.get(c,[]) for p in parent_set) for c in child_set)

def parse_obj(obj_str):
    return set(obj_str.strip('()').split(' or ')) if obj_str else set()


def filter_structural_looser(df, direct_parents):
    keep_idxs = []
    dfp = df.copy()
    dfp['obj_set'] = dfp['object'].apply(parse_obj)

    for idx, row in dfp.iterrows():
 
        if row['operation'] == 'original':
            keep_idxs.append(idx)
            continue

        subject = row['subject']
        child_set = frozenset(row['obj_set'])
        pred, rtype = row['predicate'], row['rtype']

        conflict_all = True
        for parent_cls in direct_parents.get(subject, []):
            peers = dfp[
                (dfp.subject == parent_cls) &
                (dfp.predicate == pred) &
                (dfp.rtype == rtype)
            ]
            
            if peers.empty:
                conflict_all = False
                break

            parent_ok = False
            for _, pr in peers.iterrows():
                parent_set = frozenset(pr['obj_set'])
                if is_subconcept_set(child_set, parent_set):
                    parent_ok = True
                    break


            if parent_ok:
                conflict_all = False
                break

        if not conflict_all:
            keep_idxs.append(idx)

    return df.loc[sorted(keep_idxs)].reset_index(drop=True)

def filter_parent_support_and_strength(df, direct_parents, thresholds):
    keep = set(df.index)
    dfp = df.copy(); dfp['obj_set'] = dfp['object'].apply(parse_obj)
    for name, (min_str, diff_th) in thresholds.items():
        subj, pred, rtype = name
        grp = dfp[(dfp.subject==subj)&(dfp.predicate==pred)&(dfp.rtype==rtype)]
        for p in direct_parents.get(subj, []):
            parents = grp[grp.subject==p]
            children = grp[grp.subject==subj]
            supported = any(is_subconcept_set(frozenset(cs['obj_set']), frozenset(pr['obj_set']))
                            for _, pr in parents.iterrows() for _, cs in children.iterrows())
            if not supported:
                for _, pr in parents.iterrows(): keep.discard(pr.name)
                continue
            for _, pr in parents.iterrows():
                for _, cs in children.iterrows():
                    if is_subconcept_set(frozenset(cs['obj_set']), frozenset(pr['obj_set'])) and \
                       pr['sim_to_orig'] - cs['sim_to_orig'] > diff_th and pr['sim_to_orig'] >= min_str:
                        keep.discard(pr.name)
    return df.loc[sorted(keep)].reset_index(drop=True)


def filter_by_group_threshold(df, thresholds, strength_scale=0.9,
                              min_per_group=1, max_per_group=5):

    out = []
    for name, grp in df.groupby(['subject','predicate','rtype']):
        min_str, _ = thresholds.get(name, (0.0,0.0))
        threshold = min_str * strength_scale

        
        passed = grp[ grp['sim_to_orig'] >= threshold ].copy()

       
        if len(passed) < min_per_group:
            needed = grp.nlargest(min_per_group, 'sim_to_orig')
          
            passed = pd.concat([passed, needed]) \
                    .drop_duplicates(subset=['constraint_key']) \
                    .reset_index(drop=True)

      
        if len(passed) > max_per_group:
            passed = passed.nlargest(max_per_group, 'sim_to_orig')

        out.append(passed)

    return pd.concat(out).reset_index(drop=True)

def filter_by_group_median(df, min_per_group=1, max_per_group=6):
    out = []
    for (subj,pred,rtype), grp in df.groupby(['subject','predicate','rtype']):

        med = grp['sim_to_orig'].median()

        passed = grp[ grp.sim_to_orig >= med ].copy()


        if len(passed) < min_per_group:
            needs = grp.nlargest(min_per_group, 'sim_to_orig')
            passed = pd.concat([passed, needs]) \
                       .drop_duplicates(subset=['constraint_key']) \
                       .reset_index(drop=True)

     
        if len(passed) > max_per_group:
            passed = passed.nlargest(max_per_group, 'sim_to_orig')

        out.append(passed)

    return pd.concat(out).reset_index(drop=True)

def adjust_parent_constraints(df, direct_parents):
    keep = set(df.index)
    df_copy = df.copy(); df_copy['obj_set'] = df_copy['object'].apply(parse_obj)
    for idx, row in df_copy[df_copy.operation=='original'].iterrows():
        subj, pred, rtype, parent_set = row['subject'], row['predicate'], row['rtype'], row['obj_set']
        child_sims = []
        for child in direct_parents.get(subj, []):
     
            crow = df_copy[(df_copy.subject==child)&(df_copy.operation=='original')]
            if crow.empty: continue
            cemb = crow.iloc[0]['desc_emb']
            pemb = row['desc_emb']
            child_sims.append(float(cosine_similarity([pemb],[cemb])[0,0]))

        if child_sims and np.mean(child_sims) < 0.6:
            keep.discard(idx)
    return df.loc[sorted(keep)].reset_index(drop=True)


def merge_constraints(df):
    def flatten_objs(objs):
        atoms = []
        for o in objs:
            if o.startswith('(') and o.endswith(')'):
                atoms += [p.strip() for p in o[1:-1].split(' or ')]
            else:
                atoms.append(o)
        return sorted(set(atoms))

    merged = []
    for (s, p, r), grp in df.groupby(['subject','predicate','rtype']):
        raw = [o for o in grp['object'].unique() if o]
        if not raw: continue

        if r in ('only','exactly 1','value'):
            flat = flatten_objs(raw)
            obj_str = f"({' or '.join(flat)})" if len(flat)>1 else flat[0]
            merged.append({'subject':s,'predicate':p,'rtype':r,'object':obj_str})
        else:
            for o in sorted(raw):
                merged.append({'subject':s,'predicate':p,'rtype':r,'object':o})
    return pd.DataFrame(merged)


def unify_predicates_strict(df_final,
                            direct_parents,
                            relation_emb_dict,
                            subclass_map,
                            sim_thresh=0.75):
    df = df_final.copy()
    # 1) parent_info: subj → [(parent_pred, parent_obj), …]
    parent_info = {}
    for subj in df['subject'].unique():
        cands = []
        for par in direct_parents.get(subj, []):
            prs = df[(df.subject==par)][['predicate','object']].drop_duplicates()
            for p,o in zip(prs.predicate, prs.object):
                cands.append((p,o))
        parent_info[subj] = cands

  
    for idx, row in df.iterrows():
        subj, pred, obj = row.subject, row.predicate, row.object
        cands = parent_info.get(subj, [])

   
        range_ok = [(p,o) for p,o in cands if is_subconcept_expr(obj, o, subclass_map)]
        if not range_ok:
            continue


        uniq_p = {p for p,_ in range_ok}

        if len(uniq_p) != 1:
            continue
        parent_pred = next(iter(uniq_p))


        emb_child  = relation_emb_dict.get(pred)
        emb_parent = relation_emb_dict.get(parent_pred)
        if emb_child is None or emb_parent is None:
            continue
        sim = cosine_similarity(
            emb_child.reshape(1,-1),
            emb_parent.reshape(1,-1)
        )[0,0]
        if sim < sim_thresh:
            continue


        df.at[idx, 'predicate'] = parent_pred

    return merge_constraints(df)

def prune_ancestor_objects(df, subclass_map):
    out = []
    for (s,p,r), grp in df.groupby(['subject','predicate','rtype']):
        objs = list(grp['object'])
        keep = []
        for o in objs:

            if any(o != o2 and o in subclass_map.get(o2, set()) for o2 in objs):
                continue
            keep.append(o)
        out.append(grp[grp['object'].isin(keep)])
    return pd.concat(out).reset_index(drop=True)


def dedupe_group(df_grp, emb_col='desc_emb', sim_thresh=0.9):

    df_sorted = df_grp.sort_values('sim_to_orig', ascending=False)
    kept = []           
    kept_embs = []       

    for idx, row in df_sorted.iterrows():
        emb = row[emb_col].reshape(1, -1)

        if any(cosine_similarity(emb, ke)[0,0] >= sim_thresh for ke in kept_embs):
            continue

        kept.append(idx)
        kept_embs.append(emb)

    return df_grp.loc[kept]


import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

def cal_score_with_inheritance(gold_data,pr_df,dataset_path,model_type,hierarchy_dict):


    # 1) Load gold data

    gold_df = pd.json_normalize(gold_data)
    gold_df = gold_df.rename(columns={
        'subject': 'subject',
        'restriction.property': 'property',
        'restriction.type': 'type',
        'restriction.target': 'target'
    })[['subject','property','type','target']]

    # 2) Prepare pred_df
    pred_df = pr_df.rename(columns={'predicate':'property','rtype':'type','object':'target'})
    pred_df = pred_df[['subject','property','type','target']]


    mask = (pred_df['type'] == 'only') & (pred_df['target'].str.contains(r'\sor\s'))
    pred_or = pred_df[mask].copy()
    pred_or = pred_or.assign(target=pred_or['target'].str.split(r'\sor\s')).explode('target')
    pred_rest = pred_df[~mask]
    pred_df = pd.concat([pred_rest, pred_or], ignore_index=True)

    descendants = defaultdict(set)
    def collect_descendants(parent):
        for child in hierarchy_dict.get(parent, []):
            if child not in descendants[parent]:
                descendants[parent].add(child)
                collect_descendants(child)
                descendants[parent].update(descendants[child])
    for cls in hierarchy_dict:
        collect_descendants(cls)


    def expand_df(df):
        rows = []
        for _, row in df.iterrows():
            subj = row['subject']
            rows.append(tuple(row)) 
            for desc in descendants.get(subj, []):
                rows.append((desc, row['property'], row['type'], row['target']))
        expanded = pd.DataFrame(rows, columns=['subject','property','type','target'])
        return expanded.drop_duplicates().reset_index(drop=True)

    gold_exp = expand_df(gold_df)
    pred_exp = expand_df(pred_df)

    
    all_tuples = set(map(tuple, gold_exp.values)) | set(map(tuple, pred_exp.values))
    tuple_list = sorted(all_tuples)
    tuple_to_idx = {t:i for i,t in enumerate(tuple_list)}

    y_true = [1 if t in set(map(tuple, gold_exp.values)) else 0 for t in tuple_list]
    y_pred = [1 if t in set(map(tuple, pred_exp.values)) else 0 for t in tuple_list]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)


    results = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-score'],
        'Value': [precision, recall, f1]
    })

    print('--------------------------------')
    print(f'{model_type}')
    print(f'{dataset_path}')
    print(f'{dataset_path} Evaluation with Inheritance')
    print(results.to_string(index=False))
    print('-------------------------------')

    return results


def cal_score(df_final,dataset_path,model_type,write_path=''):
    # Load ground-truth constraints from JSON (swap roles)
    with open(f'./gold/{dataset_path}_class.json', 'r') as f:
        gold_data = json.load(f)
    gold_df = pd.json_normalize(gold_data)
    gold_df = gold_df.rename(columns={
        'subject': 'subject',
        'restriction.property': 'property',
        'restriction.type': 'type',
        'restriction.target': 'target'
    })
    gold_df = gold_df[['subject', 'property', 'type', 'target']]

    # Load predicted constraints from CSV
    # pred_df = pd.read_csv(f'output/{dataset_path}/pizza_latest_results (1).csv')
    pred_df = df_final
    pred_df = pred_df.rename(columns={'predicate':'property','rtype':'type','object':'target'})
    pred_df = pred_df[['subject','property','type','target']]


    mask = (pred_df['type'] == 'only') & (pred_df['target'].str.contains(r'\sor\s'))
    pred_or = pred_df[mask].copy()

    pred_or = pred_or.assign(target=pred_or['target'].str.split(r'\sor\s')).explode('target')

    pred_rest = pred_df[~mask]

    pred_expanded = pd.concat([pred_rest,pred_or], ignore_index=True)


    pred_df = pred_expanded

    # Create a universe of all unique tuples
    all_tuples = set([tuple(x) for x in gold_df.values]) | set([tuple(x) for x in pred_df.values])
    # Map each tuple to an index
    tuple_to_idx = {t: i for i, t in enumerate(sorted(all_tuples))}

    # Build binary vectors: y_true from gold, y_pred from pred
    y_true = [1 if t in set([tuple(x) for x in gold_df.values]) else 0 for t in tuple_to_idx]
    y_pred = [1 if t in set([tuple(x) for x in pred_df.values]) else 0 for t in tuple_to_idx]

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Display results
    results = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-score'],
        'Value': [precision, recall, f1]
    })
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Evaluation Results", dataframe=results)

    # Also print to console
    print('--------------------------------')
    print(f'{model_type}')
    print(f'{dataset_path}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f1:.4f}')
    print('-------------------------------')

def ensure_subject_presence(df_filtered, df_initial):

    subjects_all = set(df_initial['subject'])
    subjects_kept = set(df_filtered['subject'])
    missing = subjects_all - subjects_kept

    to_add = []
    for subj in missing:

        cand = df_initial[
            (df_initial['subject'] == subj) &
            (df_initial['operation'] == 'original')
        ]
        if not cand.empty:
      
            top = cand.nlargest(1, 'sim_to_orig')
            to_add.append(top)
    if to_add:
        df_filtered = pd.concat([df_filtered, *to_add], ignore_index=True)
    # return df_filtered[['subject','predicate','rtype','object']]
    return df_filtered


def compute_max_depth_dfs(children_map):

    parent_map = {}
    all_nodes = set(children_map.keys())
    for parent, kids in children_map.items():
        for c in kids:
            parent_map.setdefault(c, set()).add(parent)
            all_nodes.add(c)
    
   
    roots = [n for n in all_nodes if n not in parent_map]
    if not roots:
        roots = list(all_nodes)

    @lru_cache(maxsize=None)
    def dfs(node):
        
        if node not in children_map or not children_map[node]:
            return 0
      
        return 1 + max(dfs(child) for child in children_map[node])

    #
    return max(dfs(root) for root in roots)

def compute_avg_branching(children_map):
    """
    비말단 노드(자식이 하나 이상 있는 노드)의 평균 자식 수를 계산합니다.
    """
    branch_counts = [len(childs) for childs in children_map.values() if childs]
    if not branch_counts:
        return 0.0
    return sum(branch_counts) / len(branch_counts)



# ---------------------------------------------------
def main():
    # model_types = ['4o-mini','o3-mini']
    model_types = ['o3-mini']
    # datasets = ['pizza','AfricanWild','ssn']
    datasets = ['pizza','AfricanWild','ssn']
    
    for model_type in model_types:
        print(model_type)
        for dataset in datasets:
            print(dataset)
     
            embedding_dict = torch.load(f'../{model_type}-Data/{dataset}/des_retrictions_{dataset}_embedding.pt',map_location='cpu')

            with open(f'../{model_type}-Data/{dataset}/all_processed_restrictions.json','r') as f:
                all_restrictions = json.load(f)

            object_dict = {}
            for i in all_restrictions:
                for num,j in enumerate(all_restrictions[i]):
                    object_dict[j['o']] = copy.deepcopy(embedding_dict[i][num][2])

            property_dict = {}
            for i in all_restrictions:
                for num,j in enumerate(all_restrictions[i]):
                    property_dict[j['p']] = copy.deepcopy(embedding_dict[i][num][1])

            print('Pipeline start')
            variants, orig_sup, orig_defs, children_map = load_data(dataset,model_type)
            direct_parents, subclass_map = build_subclass_map(children_map)
            globals()['SUBCLASS_MAP'] = subclass_map


            max_depth = compute_max_depth_dfs(children_map)
            avg_branching = compute_avg_branching(children_map)

            flat = (max_depth < 3)
            print(max_depth)
            # breakpoint()

            df = flatten_variants(variants, orig_sup, orig_defs)
            df = log_counts("after flatten", df, df)

            df.to_csv(f'{model_type}-output/{dataset}/preserve_df_initial.csv', index=False)
            df[['subject','predicate','rtype','object']].to_csv(f'{model_type}-output/{dataset}/wo_hierarchical_df_initial.csv', index=False)

            df = add_embeddings(df)
            df = log_counts("after embeddings", df, df)

            df_initial = copy.deepcopy(df)

            def compute_depth_and_height(children_map):
                # 1) parent_map
                parent_map = {}
                for p, childs in children_map.items():
                    for c in childs:
                        parent_map.setdefault(c, set()).add(p)
                # 2) all nodes
                all_nodes = set(children_map.keys()) | set(parent_map.keys())
                # 3) find roots
                roots = list(all_nodes - set(parent_map.keys())) or list(all_nodes)
                # BFS → depth
                depth = {r:0 for r in roots}
                q = deque(roots)
                while q:
                    u = q.popleft()
                    for v in children_map.get(u, []):
                        d = depth[u] + 1
                        if v not in depth or d < depth[v]:
                            depth[v] = d
                            q.append(v)
                # DFS → height
                height = {n:0 for n in all_nodes}
                visited = set()
                def dfs(u):
                    if u in visited: return height[u]
                    visited.add(u)
                    h_max = 0
                    for v in children_map.get(u, []):
                        h_max = max(h_max, 1 + dfs(v))
                    height[u] = h_max
                    return h_max
                for n in all_nodes:
                    dfs(n)

                # relative depth
                rel = {}
                for n in all_nodes:
                    tot = depth.get(n,0) + height.get(n,0)
                    rel[n] = depth.get(n,0)/tot if tot>0 else 0.0
                return depth, height, rel

            def get_descendants(children_map, node):
                out, q = set(), deque([node])
                while q:
                    u = q.popleft()
                    for v in children_map.get(u, []):
                        if v not in out:
                            out.add(v); q.append(v)
                return out

            def get_parent_raw_sim(row):
                subs, pred, rtype = row.subject, row.predicate, row.rtype
                sims = []
                for par in direct_parents.get(subs, []):
                    sims += df.loc[
                        (df.subject == par)
                    & (df.predicate == pred)
                    & (df.rtype     == rtype),
                        'sim_to_orig'
                    ].tolist()
                return float(np.mean(sims)) if sims else 0.0
            

            df['P_raw'] = df.apply(get_parent_raw_sim, axis=1)

            min_P, max_P = df['P_raw'].min(), df['P_raw'].max()
            range_P    = (max_P - min_P) if max_P > min_P else 1.0


            parent_score_norm = {
                subj: (score - min_P) / range_P
                for subj, score in df.groupby('subject')['P_raw'].first().items()
            }

            depth_map, height_map, rel_depth = compute_depth_and_height(children_map)

       
            roots = [c for c, ps in direct_parents.items() if not ps]
            df = df[~df['subject'].isin(roots)]

   
            children_count = {n: len(get_descendants(children_map, n)) 
                            for n in rel_depth}
            print('check')

            parent_score = {}
            for subj in df['subject'].unique():
                sims = []
                for par in direct_parents.get(subj, []):
                    sims += df.loc[
                        (df.subject==par)&(df.operation=='original'),
                        'sim_to_orig'
                    ].tolist()
                parent_score[subj] = float(np.mean(sims)) if sims else 0.0

     
            max_C = max(children_count.values()) or 1
            max_d = max(depth_map.values()) or 1
            min_P, max_P = min(parent_score.values()), max(parent_score.values())
            P_range = max_P - min_P or 1

     
            df['C_norm']     = df['subject'].map(
            lambda s: 1 - (children_count.get(s, 0) / max_C)
            )
            df['d_norm']     = df['subject'].map(lambda s: depth_map.get(s,0)/max_d)
            df['o_norm']     = df['object'].map(lambda o: depth_map.get(o,0)/max_d)
            df['Δ_depth']    = (df['o_norm'] - df['d_norm']).abs()


            # 7-f) 가중합 H 및 최종 score
            w1,w2,w3,w4 = 0.33,0.33,0.33

            df['final_score']           = w1*df['C_norm'] + w2*df['d_norm'] \
                            + w3*df['Δ_depth'] + w4*df['P_norm']

            
            # 7-g)
            if flat:
                thr = df['final_score'].quantile(0.05)
            else:
                thr = df['final_score'].quantile(0.1)
            # breakpoint()

            df = df[df['final_score'] >= thr].reset_index(drop=True)
            # ────────────────────────────────────────────────────────────
            


            all_parent_sims = []
            for idx, row in df[df.operation=='original'].iterrows():
                parent_emb = row['desc_emb']
                subj = row['subject']
                for child in direct_parents.get(subj, []):
                    cdef = orig_defs.get(child, '')
                    if not cdef: continue
                    cemb = MODEL.encode(cdef, convert_to_numpy=True)
                    all_parent_sims.append(float(cosine_similarity([parent_emb],[cemb])[0,0]))
            if flat:
                # dynamic_thresh = np.quantile(all_parent_sims,0.1) if all_parent_sims else 0.6
                pass
            else:
                dynamic_thresh = np.quantile(all_parent_sims,0.4) if all_parent_sims else 0.6
                df2 = filter_parent_relevance(df, orig_defs, direct_parents, threshold=dynamic_thresh)
                df = log_counts("after parent_relevance", df, df2)




            df2 = filter_structural_looser(df, direct_parents)
            # df2 = filter_structural(df, direct_parents)
            df = log_counts("after structural", df, df2)


            

            def filter_group_median_with_relaxed_original(
                df: pd.DataFrame,
                min_per_group: int = 0
            ) -> pd.DataFrame:
                out = []


                for key, grp in df.groupby('constraint_key'):
                    # 1)  median, 25% quantile 
                    med = grp['sim_to_orig'].median()
                    q25 = grp['sim_to_orig'].quantile(0.2)

                    # 2) non-original: median 
                    non_orig = grp[grp.operation != 'original']
                    keep_non_orig = non_orig[ non_orig.sim_to_orig >= med ]

                    # 3) original: 25% quantile  (relaxed)
                    orig = grp[ grp.operation == 'original' ]
                    keep_orig = orig[ orig.sim_to_orig >= q25 ]

                  
                    merged = pd.concat([keep_orig, keep_non_orig])
                    if len(merged) < min_per_group:
                        needed = grp.nlargest(min_per_group, 'sim_to_orig')
                        merged = pd.concat([merged, needed])
                    
                    merged = merged.loc[~merged.index.duplicated(keep='first')]

                    out.append(merged)

                return pd.concat(out).reset_index(drop=True)

            # 사용 예시
            
            if flat:
                pass
            else:
                df2 = filter_group_median_with_relaxed_original(df, min_per_group=1)
            # print(f"after relaxed_original_filter: {len(df)} -> {len(df2)}")
            df = log_counts("after median", df, df2)


            df2 = filter_remove_negative_rel(df)
            df = log_counts("after remove_negative", df, df2)


            if flat:
                thresholds = compute_thresholds_grouped(df, direct_parents,quantile_strength=0.4, quantile_diff=0.4)
                pass
            else:
                thresholds = compute_thresholds_grouped(df, direct_parents,quantile_strength=0.4, quantile_diff=0.4)
                df2 = filter_parent_support_and_strength(df, direct_parents, thresholds)
                df = log_counts("after parent_support_strength", df, df2)

            if flat:
                df2 = filter_by_group_threshold(df, thresholds, strength_scale=0.7,min_per_group=1,   
        max_per_group=6)
            else:
                df2 = filter_by_group_threshold(df, thresholds, strength_scale=0.7,min_per_group=1,   
        max_per_group=6)    
            # df2 = filter_by_group_threshold(
            #     df,
            #     min_per_group=1,
            #     max_per_group=6
            # )
            # df = log_counts("after group_threshold", df, df2)

            # df2 = filter_by_group_median(
            #     df,
            #     min_per_group=1,
            #     max_per_group=6
            # )
            df = log_counts("after group_threshold", df, df2)

            if flat:
                pass
            else:
                df2 = adjust_parent_constraints(df, direct_parents)
                df = log_counts("after adjust_parent", df, df2)

 
            df_final = merge_constraints(df)

 
            df_final = unify_predicates_strict(
                df_final,
                direct_parents,
                property_dict,    # predicate embedding dict
                subclass_map,
                sim_thresh=0.5
            )

            df_final = prune_ancestor_objects(df_final, subclass_map)
            df_final = ensure_subject_presence(df_final, df_initial)  



            cal_score(df_final,dataset,model_type)
            with open(f'../{model_type}-Data/{dataset}/{dataset}_class.json', 'r') as f:
                gold_data = json.load(f)
            cal_score_with_inheritance(gold_data,df_final,dataset,model_type,children_map)
            df_final.to_csv(f'{model_type}-output/{dataset}/preserve_df_final.csv', index=False)
            print('Pipeline end')

if __name__=='__main__':
    main()

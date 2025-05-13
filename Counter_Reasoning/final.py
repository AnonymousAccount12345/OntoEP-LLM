import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import torch
import copy
from collections import deque

dataset_path = 'AfricanWild'
model_id = '4o-mini'
print(dataset_path)
print(model_id)
print('check_1')
print("new")

with open(f'../{model_id}-Data/{dataset_path}/all_processed_restrictions.json','r') as f:
    all_restrictions = json.load(f)

with open(f'../{model_id}-Data/{dataset_path}/{dataset_path}_object_property.json','r') as f:
    all_property = json.load(f)
with open(f'../{model_id}-Data/{dataset_path}/{dataset_path}_class.json','r') as f:
    all_class = json.load(f)
with open(f'../{model_id}-Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    all_hierarchy = json.load(f)

embedding_dict = torch.load(f'../{model_id}-Data/{dataset_path}/des_retrictions_{dataset_path}_embedding.pt',map_location='cpu')


all_dict = {}
for i in embedding_dict:
    tmp = []
    for j in embedding_dict[i]:
        k = torch.matmul(torch.Tensor([0.1,0.3,0.25,0.25,0.1]),embedding_dict[i][j])
        tmp.append(copy.deepcopy(k))
    all_dict[i] = copy.deepcopy(tmp)


new_all_dict = {}
for i in all_restrictions:
    for num,j in enumerate(all_restrictions[i]):
        k = torch.matmul(torch.Tensor([0.1,0.3,0.25,0.25,0.1]),embedding_dict[i][num])
        new_all_dict[','.join([j['s'],j['p'],j['rtype'],j['o']])] = copy.deepcopy(k)
    
object_dict = {}
for i in all_restrictions:
    for num,j in enumerate(all_restrictions[i]):
        object_dict[j['o']] = copy.deepcopy(embedding_dict[i][num][2])

property_dict = {}
for i in all_restrictions:
    for num,j in enumerate(all_restrictions[i]):
        property_dict[j['p']] = copy.deepcopy(embedding_dict[i][num][1])


k = [1,2,3]
if type(all_property) == type(k):
    relation_list = all_property
else:
    relation_list = list(all_property.keys())




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
        if not any(
            c == p or p in subclass_map.get(c, set())
            for p in parent_atoms
        ):
            return False
    return True

def compute_depth_and_height(concept_children: dict):
  

    parent_map = {}
    for p, childs in concept_children.items():
        for c in childs:
            parent_map.setdefault(c, set()).add(p)

    all_nodes = set(concept_children.keys()) | set(parent_map.keys())
 
    roots = list(all_nodes - set(parent_map.keys()))
    if not roots:
        roots = list(all_nodes)

    # 1) depth_map via BFS from all roots
    depth_map = {}
    q = deque(roots)
    for r in roots:
        depth_map[r] = 0
    while q:
        u = q.popleft()
        for v in concept_children.get(u, []):
            if v not in depth_map or depth_map[v] > depth_map[u] + 1:
                depth_map[v] = depth_map[u] + 1
                q.append(v)

    # 2) height_map via post-order DFS
    height_map = {n: 0 for n in all_nodes}
    visited = set()
    def dfs(u):
        if u in visited:
            return height_map[u]
        visited.add(u)
        h = 0
        for v in concept_children.get(u, []):
            h = max(h, 1 + dfs(v))
        height_map[u] = h
        return h
    for n in all_nodes:
        dfs(n)

    return depth_map, height_map

def compute_subclass_map(concept_children: dict) -> dict:
    """
    Returns: { node: set(all ancestors), ... }
    """
    direct_parents = {}
    for parent, children in concept_children.items():
        for child in children:
            direct_parents.setdefault(child, set()).add(parent)
    all_nodes = set(concept_children.keys()) | set(direct_parents.keys())
    subclass_map = {}
    for node in all_nodes:
        ancestors = set()
        stack = list(direct_parents.get(node, []))
        while stack:
            p = stack.pop()
            if p not in ancestors:
                ancestors.add(p)
                stack.extend(direct_parents.get(p, []))
        subclass_map[node] = ancestors
    return subclass_map

def build_parents_from_children(concept_children: dict) -> dict:
    """
    { child: [parent1, parent2, ...], ... }
    """
    pm = {}
    for parent, children in concept_children.items():
        for child in children:
            pm.setdefault(child, []).append(parent)
    return pm

# --- Main prefilter function ---

def prefilter_transformations_depth_aware(
    candidates_df: pd.DataFrame,
    object_emb_dict: dict,
    relation_emb_dict: dict,
    concept_children: dict,
    all_restrictions: dict,
    depth_map: dict,
    height_map: dict,
    subclass_map: dict,
    alpha: float = 0.2,
    beta: float = 0.8,
    delta: float = 0.1,
    gen_quantile: float = 0.25,
    spec_quantile: float = 0.25,
    k_gen: int = 2,
    k_spec: int = 2,
    prop_sim_thresh=0.5
) -> pd.DataFrame:
    # 0) precompute relative depth
    relative_depth = {}
    for n, d in depth_map.items():
        h = height_map.get(n, 0)
        total = d + h
        relative_depth[n] = d/total if total>0 else 0.0

    parents_map = build_parents_from_children(concept_children)
    valid_classes = set(relative_depth.keys())


    anc_sims, desc_sims = [], []
    for o, pars in parents_map.items():
        o_emb = object_emb_dict.get(o)
        for p in pars:
            p_emb = object_emb_dict.get(p)
            if o_emb is not None and p_emb is not None:
                anc_sims.append(cosine_similarity(o_emb.reshape(1,-1),p_emb.reshape(1,-1))[0,0])
    for o, childs in concept_children.items():
        o_emb = object_emb_dict.get(o)
        for c in childs:
            c_emb = object_emb_dict.get(c)
            if o_emb is not None and c_emb is not None:
                desc_sims.append(cosine_similarity(o_emb.reshape(1,-1),c_emb.reshape(1,-1))[0,0])

    parent_sim_thresh = np.quantile(anc_sims, gen_quantile) if anc_sims else 0.0
    child_sim_thresh  = np.quantile(desc_sims, spec_quantile) if desc_sims else 0.0

    records = []
    for _, row in candidates_df.iterrows():
        subj, pred, obj, rtype, s_s = (
            row.subject, row.predicate,
            row.object,  row.rtype,
            row.support_sentence
        )
        ops = {'remove': True}

        prts = ['some', 'only']

        ops['variant_rtype'] = [rt for rt in prts if rt!=rtype] or [rtype]

        

        parent_props = {
            pr['p']
            for parent in parents_map.get(subj, [])
            for pr in all_restrictions.get(parent, [])
        }

        ops['prop_variant'] = []
        p_emb = relation_emb_dict.get(pred)
        for pp in parent_props:
            if pp == pred:
                continue
            pp_emb = relation_emb_dict.get(pp)
            if p_emb is None or pp_emb is None:
                continue
            sim_pp = cosine_similarity(
                p_emb.reshape(1,-1),
                pp_emb.reshape(1,-1)
            )[0,0]
            if sim_pp < global_prop_thresh:
                continue

    
            for parent in parents_map.get(subj, []):
                for pr in all_restrictions.get(parent, []):
                    if pr['p'] != pp:
                        continue
                    pr_obj = pr['o']
                    # 
                    if is_subconcept_expr(obj, pr_obj, subclass_map):
                        ops['prop_variant'].append(pp)
                        break
                else:
                    continue
                break

        # mbeddings & rel_depth
        o_emb   = object_emb_dict.get(obj)
        r_obj   = relative_depth.get(obj, 0.0)

   
        gen_cands = []
        for parent_obj in parents_map.get(obj, []):
            if parent_obj==obj: continue
      
            if parent_obj not in valid_classes: continue
            if not is_subconcept_expr(obj, parent_obj, subclass_map): continue
            p_emb = object_emb_dict.get(parent_obj)
            if o_emb is None or p_emb is None: continue
            sim = cosine_similarity(o_emb.reshape(1,-1), p_emb.reshape(1,-1))[0,0]
  
            r_par = relative_depth[parent_obj]
            if sim>=parent_sim_thresh and (r_obj - r_par)>=0.1:
                gen_cands.append((parent_obj, sim*(r_obj - r_par)))
        ops['generalize'] = [c for c,_ in sorted(gen_cands, key=lambda x:-x[1])[:k_gen]]


        spec_cands = []
        for child in concept_children.get(obj, []):
            if child==obj: continue
            if child not in subclass_map.get(obj, set()): continue
            c_emb = object_emb_dict.get(child)
            if o_emb is None or c_emb is None: continue
            sim = cosine_similarity(o_emb.reshape(1,-1), c_emb.reshape(1,-1))[0,0]
            r_ch = relative_depth[child]
            if sim>=child_sim_thresh and (r_ch - r_obj)>=0.1:
                spec_cands.append((child, sim*(r_ch - r_obj)))
        ops['specialize'] = [c for c,_ in sorted(spec_cands, key=lambda x:-x[1])[:k_spec]]

        records.append({
            'subject': subj,
            'predicate': pred,
            'rtype': rtype,
            'object': obj,
            'operations': ops,
            'support_sentence': s_s
        })

    return pd.DataFrame(records)


all_rels = list(property_dict.keys())
global_sims = []
for i in range(len(all_rels)):
    e1 = property_dict[all_rels[i]]
    for j in range(i+1, len(all_rels)):
        e2 = property_dict[all_rels[j]]
        sim = cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0,0]
        global_sims.append(sim)


global_prop_thresh = np.quantile(global_sims, 0.5)

# ── Usage ──
print('변형16')
# dataset_path = 'AfricanWild'
with open(f'../{model_id}-Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    all_hierarchy = json.load(f)
# prepare embeddings, all_restrictions, object_dict, property_dict as before

# candidates_df = pd.read_csv(f'../Clustering/output/{dataset_path}_semantic_clustered.csv')
candidates_df = pd.read_csv(f'../{model_id}-Data/{dataset_path}/{dataset_path}_semantic_clustered.csv')
depth_map, height_map = compute_depth_and_height(all_hierarchy)
relative_depth = {
    n: (d/(d+height_map[n])) if (d+height_map[n])>0 else 0.0
    for n, d in depth_map.items()
}
subclass_map  = compute_subclass_map(all_hierarchy)
max_depth     = max(depth_map.values()) if depth_map else 1

df_ops = prefilter_transformations_depth_aware(
    candidates_df,
    object_dict, property_dict,
    all_hierarchy, all_restrictions,
    depth_map, height_map, subclass_map,
    gen_quantile=0.25, spec_quantile=0.25
)

df_ops.to_csv(f'../{model_id}-Data/{dataset_path}/{dataset_path}_pre_counter_reasoning.csv')

print('')

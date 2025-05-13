import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import torch
import copy
import json
import ast
import re

dataset_path = 'AfricanWild'
model_id = '4o-mini'
print(model_id)
print(dataset_path)

with open(f'../{model_id}-Data/{dataset_path}/all_processed_restrictions.json','r') as f:
    all_restrictions = json.load(f)

with open(f'../{model_id}-Data/{dataset_path}/{dataset_path}_object_property.json','r') as f:
    all_property = json.load(f)
with open(f'../{model_id}-Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    all_class = json.load(f)

embedding_dict = torch.load(f'../{model_id}-Data/{dataset_path}/des_retrictions_{dataset_path}_embedding.pt',map_location='cpu')



# b = 0.1*embedding_dict['Mushroom'][1][0] + 0.2*embedding_dict['Mushroom'][1][1] + 0.3*embedding_dict['Mushroom'][1][2] + 0.4*embedding_dict['Mushroom'][1][3] + 0.5*embedding_dict['Mushroom'][1][4]


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

support_sentence_dict = dict()
for i in all_restrictions:
    for j in all_restrictions[i]:
        support_sentence_dict[','.join([j['s'],j['p'],j['rtype'],j['o']])] = copy.deepcopy(j['support_sentence'])

    
object_dict = {}
for i in all_restrictions:
    for num,j in enumerate(all_restrictions[i]):
        object_dict[j['o']] = copy.deepcopy(embedding_dict[i][num][2])

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def get_embedding_for_object(o: str, object_emb_dict: dict) -> np.ndarray:
    """Return embedding for object or union; None if atomic unknown and not in dict."""
    if o in object_emb_dict:
        return object_emb_dict[o]
    s = o.strip()
    if s.startswith("(") and s.endswith(")"):
        parts = [p.strip() for p in s[1:-1].split(" or ")]
        embs = [get_embedding_for_object(p, object_emb_dict) for p in parts]
        embs = [e for e in embs if e is not None]
        if not embs:
            return None
        emb = np.mean(embs, axis=0)
        object_emb_dict[o] = emb
        return emb
    return None

def map_to_ontology(value, allowed_set, emb_dict, threshold=0.7):
    """
    Map either predicate or object to nearest ontology term if not exact match.
    """
    if value in allowed_set:
        return value
    emb = get_embedding_for_object(value, emb_dict)
    if emb is None:
        return None
    candidates = [c for c in allowed_set if c in emb_dict]
    if not candidates:
        return None
    mats = np.vstack([emb_dict[c] for c in candidates])
    sims = cosine_similarity(emb.reshape(1,-1), mats).flatten()
    idx = np.argmax(sims)
    return candidates[idx] if sims[idx] >= threshold else None

def semantic_merge_filter_with_mapping(emb_dict, object_emb_dict,
                                       ontology_props, ontology_concepts,sup_sen_dict,
                                       knn_k=5, cluster_distance_threshold=0.6,
                                       distance_thresh=0.3, min_cluster_size=2,
                                       mapping_threshold=0.7):
    # 1. Compute X matrix and avg distances
    keys = list(emb_dict.keys())
    X = np.vstack([emb_dict[k] for k in keys])
    knn = NearestNeighbors(n_neighbors=knn_k, metric='cosine').fit(X)
    avg_dist = knn.kneighbors(X)[0].mean(axis=1)
    # 2. Semantic clustering
    clustering = AgglomerativeClustering(n_clusters=None, affinity='cosine',linkage='average', distance_threshold=cluster_distance_threshold)
    clusters = clustering.fit_predict(X)

    support_sen = []
    #prepare support sentence
    for i in keys:
        support_sen.append(sup_sen_dict[i])

    pd.set_option('display.max_colwidth',None)
    # 3. Prepare DataFrame
    df = pd.DataFrame({'key': keys, 'cluster': clusters, 'avg_dist': avg_dist})
    df = df[df['avg_dist'] <= distance_thresh]
    valid = df['cluster'].value_counts().loc[lambda s: s>=min_cluster_size].index
    df = df[df['cluster'].isin(valid)].copy()
    # 4. Parse into components (subject, predicate, rtype, object)
    parsed = df['key'].str.split(',', n=3, expand=True)
    df['subject']   = parsed[0]
    df['predicate'] = parsed[1]
    df['rtype']     = parsed[2]
    df['object']    = parsed[3]
    df['support_sentence'] = support_sen

    # 5. Map predicates and objects to ontology
    df['mapped_pred'] = df['predicate'].apply(
        lambda p: map_to_ontology(p, ontology_props, emb_dict, mapping_threshold))
    df['mapped_obj'] = df['object'].apply(
        lambda o: map_to_ontology(o, ontology_concepts, object_emb_dict, mapping_threshold))
    df = df[df['mapped_pred'].notnull() & df['mapped_obj'].notnull()].copy()
    # 6. Merge by (subject, mapped_pred, rtype) with correct semantics
    merged = []
    for (s, p, r), grp in df.groupby(['subject', 'mapped_pred', 'rtype']):
        objs = sorted(grp['mapped_obj'].unique())

        if r in ('only', 'exactly 1', 'value'):
            obj_str = f"({' or '.join(objs)})" if len(objs) > 1 else objs[0]
            merged.append({
                'subject': s,
                'predicate': p,
                'rtype': r,
                'object': obj_str,
                'support_sentence': sorted(grp['support_sentence'].unique())
            })
        else:
    
            for o in objs:
                merged.append({
                    'subject': s,
                    'predicate': p,
                    'rtype': r,
                    'object': o,
                    'support_sentence': sorted(grp[grp['mapped_obj'] == o]['support_sentence'].unique())
                })
    return pd.DataFrame(merged)

# breakpoint()
k = dict()
k['1'] = 1
if type(all_property) == type(k):
    all_p = set(all_property.keys())
else:
    all_p =set(all_property)

if type(all_class) == type(k):
    all_c = set(all_class.keys())
else:
    all_c = set(all_class)

df_result = semantic_merge_filter_with_mapping(new_all_dict, object_dict,all_p,all_c,support_sentence_dict)
df_result.to_csv(f'../{model_id}-Data/{dataset_path}/{dataset_path}_semantic_clustered.csv')


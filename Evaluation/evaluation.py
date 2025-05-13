import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pandas as pd
from collections import defaultdict


import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

def cal_score_with_inheritance(dataset_path, pr_df, hierarchy_dict):

    # 1) Load gold data
    with open(f'../4o-mini-Data/{dataset_path}/{dataset_path}_class.json', 'r') as f:
        gold_data = json.load(f)
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
    print(f'{dataset_path} Evaluation with Inheritance')
    print(results.to_string(index=False))
    return results

def _prepare_gold_pred(dataset_path, pr_df):

    with open(f'../4o-mini-Data/{dataset_path}/{dataset_path}_class.json', 'r') as f:
        gold_data = json.load(f)
    gold_df = pd.json_normalize(gold_data)
    gold_df = gold_df.rename(columns={
        'subject':'subject',
        'restriction.property':'property',
        'restriction.type':'type',
        'restriction.target':'target'
    })[['subject','property','type','target']]

    pred_df = pr_df.rename(columns={'predicate':'property','rtype':'type','object':'target'})
    pred_df = pred_df[['subject','property','type','target']]

    mask = (pred_df['type']=='only') & pred_df['target'].str.contains(r'\sor\s')
    pred_or = pred_df[mask].assign(target=pred_df.loc[mask,'target'].str.split(r'\sor\s')).explode('target')
    pred_df = pd.concat([pred_df[~mask], pred_or], ignore_index=True)
    return gold_df.drop_duplicates(), pred_df.drop_duplicates()

def error_type_metrics(dataset_path, pr_df):

    gold_df, pred_df = _prepare_gold_pred(dataset_path, pr_df)


    gold_set = set(map(tuple, gold_df.values))
    pred_set = set(map(tuple, pred_df.values))


    counts = {'correct':0, 'error1_property':0, 'error2_type':0, 'error3_object':0, 'other_errors':0}


    for subj, prop, rtype, obj in pred_set:
        tpl = (subj, prop, rtype, obj)
        if tpl in gold_set:
            counts['correct'] += 1
        else:

            same_subj = [g for g in gold_set if g[0]==subj]

            if any((g[0]==subj and g[2]==rtype and g[3]==obj and g[1]!=prop) for g in same_subj):
                counts['error1_property'] += 1

            elif any((g[0]==subj and g[1]==prop and g[3]==obj and g[2]!=rtype) for g in same_subj):
                counts['error2_type'] += 1

            elif any((g[0]==subj and g[1]==prop and g[2]==rtype and g[3]!=obj) for g in same_subj):
                counts['error3_object'] += 1
            else:
                counts['other_errors'] += 1


    total = len(pred_set)
    metrics = {k: (v, v/total if total>0 else 0.0) for k,v in counts.items()}
    return pd.DataFrame([
        {'ErrorType':k, 'Count':v[0], 'Rate':v[1]}
        for k,v in metrics.items()
    ])


def error_type_metrics_with_inheritance(dataset_path, pr_df, hierarchy_dict):

    gold_df, pred_df = _prepare_gold_pred(dataset_path, pr_df)


    descendants = defaultdict(set)
    def collect_desc(p):
        for c in hierarchy_dict.get(p, []):
            if c not in descendants[p]:
                descendants[p].add(c)
                collect_desc(c)
                descendants[p].update(descendants[c])
    for cls in hierarchy_dict:
        collect_desc(cls)

    def expand(df):
        rows = []
        for subj, prop, rtype, obj in df.values:
            rows.append((subj,prop,rtype,obj))
            for desc in descendants.get(subj, []):
                rows.append((desc,prop,rtype,obj))
        return pd.DataFrame(rows, columns=['subject','property','type','target']).drop_duplicates()

    gold_exp = expand(gold_df)
    pred_exp = expand(pred_df)

    gold_set = set(map(tuple, gold_exp.values))
    pred_set = set(map(tuple, pred_exp.values))


    counts = {'correct':0, 'error1_property':0, 'error2_type':0, 'error3_object':0, 'other_errors':0}
    for subj, prop, rtype, obj in pred_set:
        tpl = (subj, prop, rtype, obj)
        if tpl in gold_set:
            counts['correct'] += 1
        else:
            same_subj = [g for g in gold_set if g[0]==subj]
            if any((g[0]==subj and g[2]==rtype and g[3]==obj and g[1]!=prop) for g in same_subj):
                counts['error1_property'] += 1
            elif any((g[0]==subj and g[1]==prop and g[3]==obj and g[2]!=rtype) for g in same_subj):
                counts['error2_type'] += 1
            elif any((g[0]==subj and g[1]==prop and g[2]==rtype and g[3]!=obj) for g in same_subj):
                counts['error3_object'] += 1
            else:
                counts['other_errors'] += 1

    total = len(pred_set)
    metrics = {k: (v, v/total if total>0 else 0.0) for k,v in counts.items()}
    return pd.DataFrame([
        {'ErrorType':k, 'Count':v[0], 'Rate':v[1]}
        for k,v in metrics.items()
    ])


from collections import defaultdict
import pandas as pd

def find_structural_conflict(df, hierarchy):

    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('subject',):
            col_map[col] = 'subject'
        elif cl in ('predicate', 'property'):
            col_map[col] = 'predicate'
        elif cl in ('rtype', 'type', 'restrictiontype'):
            col_map[col] = 'rtype'
        elif cl in ('object', 'target', 'value'):
            col_map[col] = 'object'
    df = df.rename(columns=col_map)
    required = {'subject', 'predicate', 'rtype', 'object'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"aa: {missing}")


    parent_map = {}
    for parent, children in hierarchy.items():
        for child in children:
            parent_map[child] = parent

 
    constraints = defaultdict(lambda: defaultdict(lambda: {'only': set(), 'some': set()}))
    for _, row in df.iterrows():
        cls = row['subject']
        prop = row['predicate']
        rtype = row['rtype']
        obj = row['object']
        if pd.isna(rtype) or pd.isna(obj):
            continue
        obj_str = str(obj).strip()

        if ' or ' in obj_str:
            cleaned = obj_str.strip('()')
            vals = [x.strip() for x in cleaned.split(' or ') if x.strip()]
        else:
            vals = [obj_str]
        if rtype not in ('only', 'some'):
            continue
        constraints[cls][prop][rtype].update(vals)


    def gather_ancestor(cls, prop, rtype):
        vals = set()
        cur = cls
        while cur in parent_map:
            par = parent_map[cur]
            prop_dict = constraints.get(par, {})
            vals |= prop_dict.get(prop, {}).get(rtype, set())
            cur = par
        return vals


    for cls in list(constraints.keys()):
        for prop in list(constraints[cls].keys()):
            inh_only = gather_ancestor(cls, prop, 'only')
            inh_some = gather_ancestor(cls, prop, 'some')
            constraints[cls][prop]['only'] |= inh_only
            constraints[cls][prop]['some'] |= inh_some


    conflicts = []
    class_conflict_count = defaultdict(int)

    for cls, props in constraints.items():
        for prop, types in props.items():
            child_only = types['only']
            child_some = types['some']
            parent_only = gather_ancestor(cls, prop, 'only')


            if parent_only and child_only and not child_only.issubset(parent_only):
                conflicts.append({
                    'class': cls, 'property': prop,
                    'type': 'only–only',
                    'parent_vals': parent_only,
                    'child_vals': child_only
                })
                class_conflict_count[cls] += 1


            if parent_only:
                invalid = {v for v in child_some if v not in parent_only}
                if invalid:
                    conflicts.append({
                        'class': cls, 'property': prop,
                        'type': 'only–some (parent→child)',
                        'parent_vals': parent_only,
                        'child_vals': invalid
                    })
                    class_conflict_count[cls] += 1

  
            if child_only and child_some:
                invalid_internal = {v for v in child_some if v not in child_only}
                if invalid_internal:
                    conflicts.append({
                        'class': cls, 'property': prop,
                        'type': 'internal some–only',
                        'only_vals': child_only,
                        'some_vals': invalid_internal
                    })
                    class_conflict_count[cls] += 1

    total_conflicts = len(conflicts)
    print(f"total conflicts count: {total_conflicts}\n")

    if total_conflicts:
        print("contradiction:")
        for cls, cnt in class_conflict_count.items():
            print(f" - {cls}: {cnt}")
        print("\detail:")
        for c in conflicts:
            print(f"- [{c['type']}] class `{c['class']}` / property `{c['property']}`")
            if c['type'] == 'internal some–only':
                print(f"    only = {c['only_vals']}")
                print(f"    some  = {c['some_vals']}")
            else:
                print(f"    parent only = {c.get('parent_vals')}")
                print(f"    child      = {c.get('child_vals')}")
    else:
        print("no contradiction")



def find_only_some_count(df):
    constraints = defaultdict(lambda: defaultdict(lambda: {'only': set(), 'some': set()}))
    for _, row in df.iterrows():
        cls, prop, rtype, obj = row['subject'], row['predicate'], row['rtype'], row['object']
        if pd.isna(rtype) or pd.isna(obj): continue
        rtype = rtype.strip().lower()
        obj_str = str(obj).strip()
        if obj_str.startswith('(') and obj_str.endswith(')'):
            vals = [x.strip() for x in obj_str[1:-1].split(' or ') if x.strip()]
        else:
            vals = [obj_str]
        if rtype in ('only', 'some'):
            constraints[cls][prop][rtype].update(vals)


    total_only = sum(len(types['only'])
                    for props in constraints.values()
                    for types in props.values())
    total_some = sum(len(types['some'])
                    for props in constraints.values()
                    for types in props.values())
    print(f"# of only: {total_only}")
    print(f"# of some: {total_some}")






dataset_path = 'AfricanWild'
model_type = '4o-mini'

with open(f'../{model_type}-Data/{dataset_path}/{dataset_path}_hierarchy.json', 'r') as f:
    hierarchy = json.load(f)

with open(f'../{model_type}-Data/{dataset_path}/{dataset_path}_class.json', 'r') as f:
    gold_data = json.load(f)
gold_df = pd.json_normalize(gold_data)
gold_df = gold_df.rename(columns={
    'subject': 'subject',
    'restriction.property': 'property',
    'restriction.type': 'type',
    'restriction.target': 'target'
})[['subject','property','type','target']]

gold_df.to_csv(f'{model_type}/output/{dataset_path}/gold.csv', index=False)

print('baselines:')
df = pd.read_csv(f'{model_type}/output/{dataset_path}/my.csv') 
cal_score_with_inheritance(dataset_path,df,hierarchy)
df_errors = error_type_metrics(dataset_path, df)
print(df_errors)
print('------')
df_err_inh = error_type_metrics_with_inheritance(dataset_path, df, hierarchy)
print(df_err_inh)
find_structural_conflict(df,hierarchy)
find_only_some_count(df)

print('=============================================================')
print('our model:')
df = pd.read_csv(f'{model_type}/output/{dataset_path}/{dataset_path}_batch_results.csv') 
cal_score_with_inheritance(dataset_path,df,hierarchy)
df_errors = error_type_metrics(dataset_path, df)
print(df_errors)
print('------')
df_err_inh = error_type_metrics_with_inheritance(dataset_path, df, hierarchy)
print(df_err_inh)
find_structural_conflict(df,hierarchy)
find_only_some_count(df)

print('=============================================================')
print('w/o Final Restriction:')
df = pd.read_csv(f'{model_type}/output/{dataset_path}/preserve_df_final.csv') 
cal_score_with_inheritance(dataset_path,df,hierarchy)
df_errors = error_type_metrics(dataset_path, df)
print(df_errors)
print('------')
df_err_inh = error_type_metrics_with_inheritance(dataset_path, df, hierarchy)
print(df_err_inh)
find_structural_conflict(df,hierarchy)
find_only_some_count(df)



print('=============================================================')

print('without Counter:')
df = pd.read_csv(f'{model_type}/wo_Counter_output/{dataset_path}/{dataset_path}_batch_results.csv') 
cal_score_with_inheritance(dataset_path,df,hierarchy)
df_errors = error_type_metrics(dataset_path, df)
print(df_errors)
print('------')
df_err_inh = error_type_metrics_with_inheritance(dataset_path, df, hierarchy)
print(df_err_inh)
find_structural_conflict(df,hierarchy)
find_only_some_count(df)

print('=============================================================')

print('without Hierarchical:')
df = pd.read_csv(f'{model_type}/wo_Hierarchical_output/{dataset_path}/{dataset_path}_batch_results.csv') 
cal_score_with_inheritance(dataset_path,df,hierarchy)
df_errors = error_type_metrics(dataset_path, df)
print(df_errors)
print('------')
df_err_inh = error_type_metrics_with_inheritance(dataset_path, df, hierarchy)
print(df_err_inh)
find_structural_conflict(df,hierarchy)
find_only_some_count(df)


# df = pd.read_csv(f'output/{dataset_path}/{dataset_path}_batch_results.csv') 


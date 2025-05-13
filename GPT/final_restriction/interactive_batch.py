import json
import copy
import jsonlines
import pandas as pd
import json
import pandas as pd
import argparse
import os
import re
import io
import copy


dataset_path = 'pizza'
# model_name = 'Exemplar_Vanilla'
pt_template = 'generate_final_restriction'

with open(f'../../Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    hierarchy = json.load(f)
    
with open(f'../../Data/{dataset_path}/{dataset_path}_description.json','r') as f:
    descriptions = json.load(f)

with open(f'../../Data/{dataset_path}/{dataset_path}_object_property.json','r') as f:
    properties = json.load(f)
if dataset_path == 'pizza' or dataset_path == 'AfricanWild':
    properties = properties.keys()
initial_restrictions = pd.read_csv(f'./final_input/{dataset_path}/preserve_df_initial.csv')
all_restrictions = pd.read_csv(f'./final_input/{dataset_path}/preserve_df_final.csv')


all_restrictions = pd.read_csv(f'../../Data/{dataset_path}/{dataset_path}_pre_counter_reasoning.csv')
# with open(f'../Data/{dataset_path}/{dataset_path}_clas','r') as f:
    # test = f.readlines()


init_template = {
    "custom_id": None, 
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "o3-mini",  
             "messages": 
                 [],

            "reasoning_effort":"medium"  
             }
    }

with open(f'../../prompt_template/{pt_template}.txt','r') as file:
    pt_template = file.read()



all_dict = dict()
for i in all_restrictions['subject'].unique():
    class_definition = descriptions[i]
    num = 1
    all_list = []
    for j in all_restrictions[all_restrictions['subject']==i].iterrows():
        temp_subject = j[1]['subject']
        temp_property = j[1]['predicate']
        temp_rtype = j[1]['rtype']
        temp_object = j[1]['object']
        mask = (
            (initial_restrictions['subject']   == temp_subject)  &
            (initial_restrictions['predicate'] == temp_property) &
            (initial_restrictions['rtype']     == temp_rtype)    &
            (initial_restrictions['object']    == temp_object)
        )
        if mask.any():
            matched_row = initial_restrictions.loc[mask].iloc[0]   
            temp_sentence = matched_row['combined']  
        else:

            temp_sentence = class_definition
        all_str = f'{num}. {temp_subject}, {temp_property}, {temp_rtype}, {temp_object} \n   - Support: {temp_sentence}'
        all_list.append(all_str)
        num+=1
    all_dict[i] = '\n'.join(all_list)


batches = []
data_dict = {}
score = []
subquestions = []
subanswers = []


for id, query in enumerate(all_dict.keys()):
    temp = copy.deepcopy(init_template)
    temp['custom_id'] = f'{id}'
    prompt = pt_template.format(domain = dataset_path,class_name = query, class_definition = descriptions[query], property_restriction = all_dict[query],class_hierarchy = hierarchy,object_property = properties)

    temp['body']['messages'].append({"role": "user", "content": prompt})
    batches.append(temp)
    data_dict[f'{id}'] = query



task_type = f'output/{dataset_path}'

with open(f'{task_type}/{dataset_path}.json','w') as f:
    json.dump(data_dict,f)
    
with open(f'{task_type}/{dataset_path}_batchinput.jsonl', 'w') as file:
    for item in batches:
        json_string = json.dumps(item)
        file.write(json_string + '\n')




from openai import OpenAI
client = OpenAI(api_key="")

batch_input_file = client.files.create(
  file=open(f"{task_type}/{dataset_path}_batchinput.jsonl", "rb"),
  purpose="batch"
)

all_des = []
temp_dict = dict()

for i in batches:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": i['body']['messages'][0]['content']}],
        temperature=0.2,
        max_completion_tokens=1800
    )
   
    temp_dict[i['custom_id']] = copy.deepcopy(response)
  


all_dict = dict()

for i in temp_dict:
    all_dict[i] = temp_dict[i].choices[0].message.content
with open(f'{task_type}/{dataset_path}_batch_results.json', 'w') as file:
    json.dump(all_dict,file)







def parse_csv_blocks(data):

    dfs = []
    def extract_from_obj(obj):
        if isinstance(obj, str):
            for match in re.finditer(r'```csv\n(.*?)```', obj, re.DOTALL):
                csv_str = match.group(1)
                dfs.append(pd.read_csv(io.StringIO(csv_str)))
        elif isinstance(obj, dict):
            for v in obj.values():
                extract_from_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_from_obj(item)
    extract_from_obj(data)
    return dfs

def json_to_csv(input_json_path, output_csv_path,
                record_path=None, meta=None):

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    dfs = parse_csv_blocks(data)
    if dfs:

        df = pd.concat(dfs, ignore_index=True)
    else:

        if record_path is None:
            df = pd.DataFrame(data)
        else:
            df = pd.json_normalize(
                data,
                record_path=record_path,
                meta=meta or [],
                errors='ignore'
            )


    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved CSV: {output_csv_path}")


datasets = ['pizza','AfricanWild','ssn']
for i in datasets:
    input_json = f'./output/{i}/{i}_batch_results.json'
    output_csv = f'./output/{i}/{i}_batch_results.csv'


    json_to_csv(
        input_json_path=input_json,
        output_csv_path=output_csv,
    )

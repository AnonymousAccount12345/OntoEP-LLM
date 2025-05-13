import json
import copy
import jsonlines
import pandas as pd
import json
import jsonlines
import copy

dataset_path = 'AfricanWild'
# model_name = 'Exemplar_Vanilla'
pt_template = 'Counter_Reasoning'


with open(f'../../Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    hierarchy = json.load(f)
with open(f'../../Data/{dataset_path}/all_processed_restrictions.json','r') as f:
    restrictions = json.load(f)
    
with open(f'../../Data/{dataset_path}/{dataset_path}_description.json','r') as f:
    descriptions = json.load(f)

with open(f'../../Data/{dataset_path}/{dataset_path}_object_property.json','r') as f:
    properties = json.load(f)

all_restrictions = pd.read_csv(f'../../Data/{dataset_path}/{dataset_path}_pre_counter_reasoning.csv')
# with open(f'../Data/{dataset_path}/{dataset_path}_clas','r') as f:
    # test = f.readlines()


    
init_template = {
    "custom_id": None, 
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4o-mini",  
             "messages": 
                 [],
             "temperature": 0.6,
             "max_tokens": 800 
             }
    }

with open(f'../../prompt_template/{pt_template}.txt','r') as file:
    pt_template = file.read()


batches = []
# task_type = 'Vanilla'
data_dict = {}
score = []
subquestions = []
subanswers = []


for id, query in enumerate(all_restrictions.iterrows()):
    temp = copy.deepcopy(init_template)
    temp['custom_id'] = f'{id}'
    prompt = pt_template.format(class_name = query[1]['subject'],class_def_text = descriptions[query[1]['subject']],support_text = query[1]['support_sentence'],ori_predicate=query[1]['predicate'],ori_rtype=query[1]['rtype'], ori_object=query[1]['object'],transformations=query[1]['operations'].replace('\'','').replace('\"',''))
    # d_h = '\n'.join(query['dialogue_history'])
    # exemplars = []
    # for i in query['exemplars']:
        # exemplars.append('usr: ' + i[0] + '\n' + '[' + i[1] + ']'+ 'sys: ' + i[2])
    # prompt = pt_template.format(concept=i.spilit('#')[-1])
    temp['body']['messages'].append({"role": "user", "content": prompt})
    batches.append(temp)
    data_dict[f'{id}'] = ','.join([query[1]['subject'],query[1]['predicate'],query[1]['rtype'],query[1]['object']])



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
import copy
for i in batches:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": i['body']['messages'][0]['content']}],
        temperature=0.6,
        max_completion_tokens=500
    )
   
    temp_dict[i['custom_id']] = copy.deepcopy(response)
    # temp_des = response
  


# json_all_des = [json.dumps(temp_dict[item]) for item in temp_dict]

all_dict = dict()

for i in temp_dict:
    all_dict[i] = temp_dict[i].choices[0].message.content
with open(f'{task_type}/{dataset_path}_batch_results.json', 'w') as file:
    json.dump(all_dict,file)



import jsonlines

# Open the jsonlines file for reading
all_output = []
with open(f'./output/{dataset_path}/{dataset_path}_batch_results.json', 'r') as reader:
    # Iterate over each line in the file
    all_output = json.load(reader)
# with jsonlines.open(f'./output/{dataset_path}/{dataset_path}_batch_results.jsonl') as reader:
#     # Iterate over each line in the file
#     for line in reader:
#         # Process each line as a JSON object
#         all_output.append(line)

with open(f'./output/{dataset_path}/{dataset_path}.json','r') as f:
    all_data = json.load(f)

outputs = dict()

for i in all_output:
    outputs[i] = all_output[i]

import json
import pandas as pd
import re

def organize_and_save_from_string(json_str: str):

    md_pattern = r"```json\s*(.*?)```"
    match = re.search(md_pattern, json_str, re.DOTALL)
    json_content = match.group(1) if match else json_str


    data = json.loads(json_content)



    return data

output_csv = "operations.csv"
output_json = "organized_operations.json"

df = dict()

# outpu
# df = pd.DataFrame()
for i in outputs:
    cur_data = organize_and_save_from_string(outputs[i])
    df[all_data[i]] = copy.deepcopy(cur_data)



with open(f'./output/{dataset_path}/{dataset_path}_counter_reasoning.json','w') as f:
    json.dump(df, f, ensure_ascii=False, indent=2)
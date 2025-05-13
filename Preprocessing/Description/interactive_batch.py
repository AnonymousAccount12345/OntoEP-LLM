import json
import copy
import jsonlines


dataset_path = 'ssn'
# model_name = 'Exemplar_Vanilla'
pt_template = 'generate_description'


# with open(f'../Data/{dataset_path}/{dataset_path}_class.txt','r') as f:
#     test = f.readlines()
# test = [line.strip() for line in test]
with open(f'../../Data/{dataset_path}/{dataset_path}_class.json','r') as f:
    test = json.load(f)
with open(f'../../Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    class_hierarchy = json.load(f)
# test = [line.strip() for line in test]



init_template = {
    "custom_id": None,
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4o-mini", 
             "messages": 
                 [],
             "temperature": 0.6,
             "max_tokens": 300 
             }
    }


with open(f'../../prompt_template/{pt_template}.txt','r') as file:
    pt_template = file.read()



batches = []
# task_type = 'Vanilla'
data_set = 'test'
data_dict = {}
score = []
subquestions = []
subanswers = []


for id, query in enumerate(class_hierarchy):
    temp = copy.deepcopy(init_template)
    temp['custom_id'] = f'{id}'
    # d_h = '\n'.join(query['dialogue_history'])
    all_info = []
    for j in test:
        if j['subject'] == query:
            all_info.append(j)
    exemplars = []
    # for i in query['exemplars']:
    #     exemplars.append('usr: ' + i[0] + '\n' + '[' + i[1] + ']'+ 'sys: ' + i[2])
    # for i in test:
        # pt.templatetest[i]
    prompt = pt_template.format(domain = dataset_path, class_information = all_info,class_hierarchy = class_hierarchy)
    temp['body']['messages'].append({"role": "user", "content": prompt})
    batches.append(temp)
    data_dict[f'{id}'] = query


task_type = f'output/{dataset_path}'


with open(f'{task_type}/{data_set}.json','w') as f:
    json.dump(data_dict,f)
    
with open(f'{task_type}/{data_set}_batchinput.jsonl', 'w') as file:
    for item in batches:
        json_string = json.dumps(item)
        file.write(json_string + '\n')


from openai import OpenAI

client = OpenAI(api_key="")

batch_input_file = client.files.create(
  file=open(f"{task_type}/{data_set}_batchinput.jsonl", "rb"),
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
  

all_dict = dict()

for i in temp_dict:
    all_dict[i] = temp_dict[i].choices[0].message.content
with open(f'{task_type}/{dataset_path}_batch_results.json', 'w') as file:
    json.dump(all_dict,file)


import json
import copy


with open(f'./output/{dataset_path}/{dataset_path}_batch_results.json','r') as f:
    results = json.load(f)
with open(f'./output/{dataset_path}/test.json','r') as f:
    dic = json.load(f)


all_des = dict()

for i in dic:
    all_des[dic[i]] = copy.deepcopy(results[i])


import json

with open(f'../../Data/{dataset_path}/{dataset_path}_description.json', 'w') as f:
    json.dump(all_des, f)
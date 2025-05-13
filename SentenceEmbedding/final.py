import torch
import json
from sentence_transformers import SentenceTransformer
import copy

st_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

dataset_path = 'ssn'
model_id = '4o-mini'
print(f"L5oading data for: {dataset_path}")


restriction_dict = {
    'some':    "At least one must exist",
    'only':    "If any exist, they must all match the defined criterion"
}

with open(f'../{model_id}-Data/{dataset_path}/all_processed_restrictions.json','r') as f:
    datas = json.load(f)

all_dict = {}

for i in datas:
    temp_dict = {}
    for num,j in enumerate(datas[i]):
        temp = [j['s'],j['p'],j['o'],restriction_dict[j['rtype'].split(' ')[0]]+' '.join(j['rtype'].split(' ')[1:]),j['support_sentence']]
        with torch.no_grad():
            embeddings = st_model.encode(temp, convert_to_tensor=True)
        temp_dict[num] = copy.deepcopy(embeddings)
    all_dict[i] = copy.deepcopy(temp_dict)
        

output_path = f'../{model_id}-Data/{dataset_path}/des_retrictions_{dataset_path}_embedding.pt'
torch.save(all_dict, output_path)
print(f"Saved embeddings to {output_path}")
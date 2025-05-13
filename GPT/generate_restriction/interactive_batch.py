import json
import copy
import jsonlines


dataset_path = 'AfricanWild'
# model_name = 'Exemplar_Vanilla'
pt_template = 'my_generate_restriction'


# with open(f'../Data/{dataset_path}/{dataset_path}_class.txt','r') as f:
#     test = f.readlines()
# test = [line.strip() for line in test]
with open(f'../../Data/{dataset_path}/{dataset_path}_description.json','r') as f:
    class_descriptions = json.load(f)
# with open(f'../../Data/{dataset_path}/{dataset_path}_class.json','r') as f:
    # class_description = json.load(f)
with open(f'../../Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    class_hierarchy = json.load(f)
with open(f'../../Data/{dataset_path}/{dataset_path}_object_property.json','r') as f:
    object_properties = json.load(f)
object_properties = object_properties.keys()


# test = [line.strip() for line in test]


import json
import os
from collections import deque
from openai import OpenAI

model_name = 'gpt-4o-mini'

client = OpenAI(api_key="")


# Load class descriptions
with open(f'../../Data/{dataset_path}/{dataset_path}_description.json','r') as f:
    class_descriptions = json.load(f)
# with open(f'../../Data/{dataset_path}/{dataset_path}_class.json','r') as f:
    # class_description = json.load(f)
with open(f'../../Data/{dataset_path}/{dataset_path}_hierarchy.json','r') as f:
    class_hierarchy = json.load(f)
with open(f'../../Data/{dataset_path}/{dataset_path}_object_property.json','r') as f:
    object_properties = json.load(f)


parent_map = {}
for parent, children in class_hierarchy.items():
    for child in children:
        parent_map[child] = parent

# Find root classes (those not appearing as a child)
root_classes = [cls for cls in class_hierarchy.keys() if cls not in parent_map]

# Storage for generated restrictions
class_restrictions = {}

# Breadth-first traversal
queue = deque(root_classes)
while queue:
    current = queue.popleft()
    description = class_descriptions.get(current, "")
    parent = parent_map.get(current)
    super_constraint = class_restrictions.get(parent, "")

    # Prepare lists for prompt
    if type(object_properties) == dict:

        props_list = list(object_properties.keys())
    else:
        props_list = object_properties
    concept_list = list(class_descriptions.keys())

    
    pt_template = 'my_generate_restriction'
    with open(f'../../prompt_template/{pt_template}.txt','r') as file:
        prompt = file.read()

    if pt_template == 'my_generate_restriction':
        prompt = prompt.format(class_name = current, description = description,class_hierarchy = class_hierarchy,object_properties = props_list)


#     # Call GPT to generate the restriction
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=512
    )
    constraint = response.choices[0].message.content.strip()
    class_restrictions[current] = constraint
    print('prompt:',prompt)
    print(current,constraint)

    print('======================')

    # Enqueue children
    for child in class_hierarchy.get(current, []):
        queue.append(child)

OUTPUT_PATH = f"{dataset_path}/{model_name}_{pt_template}.json"

# Save all restrictions to JSON
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(class_restrictions, f, ensure_ascii=False, indent=2)

print(f"Generated property restrictions saved to {OUTPUT_PATH}")


with open(f'./{dataset_path}/gpt-4o-mini_my_generate_restriction.json','r') as f:
    data = json.load(f)


import pprint
import re

def parse_support_data(input_data):
    """
    Parses input_data where each value is a string containing entries of the form:
    '(s,p,rtype,o) / support sentence: "..."'
    
    Returns a dict with the same keys, where each value is a list of dicts:
    {'s': ..., 'p': ..., 'rtype': ..., 'o': ..., 'support_sentence': ...}
    """
    output_data = {}

    for key, text in input_data.items():
        # Split entries by blank lines
        entries = [entry.strip() for entry in re.split(r'\n\s*\n', text) if entry.strip()]
        parsed_entries = []

        for entry in entries:
            # Split into the quadruple part and the support sentence
            if '/ support sentence:' in entry:
                quad_part, support_part = entry.split('/ support sentence:', 1)
                # Parse the quadruple (s, p, rtype, o)
                fields = [f.strip() for f in quad_part.split(',', 3)]
                if len(fields) == 4:
                    s, p, rtype, o = fields
                    # Clean up the support sentence text
                    support_sentence = support_part.strip().strip(' "')
                    parsed_entries.append({
                        's': s,
                        'p': p,
                        'rtype': rtype,
                        'o': o,
                        'support_sentence': support_sentence
                    })

        output_data[key] = parsed_entries

    return output_data

# Example usage:


parsed = parse_support_data(data)

with open(f'./{dataset_path}/all_processed_restrictions.json','w') as f:
    json.dump(parsed, f, indent=4)
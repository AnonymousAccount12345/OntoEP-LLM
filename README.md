

This repository is a repo that stores the source code, data, and prompts for OntoEP-LLM (LLM-Driven Property Restriction Generation via Variant Expansion and Semantic-Hiearchical Pruning).

The sequence of operations is as follows

1. run Preprocessing/Description/Interactive_batch.py. Generate a description for each class based on GPT-4o-mini.

2. next, generate an Initial restriction for each dataset and model in GPT/generate_restriction (api_key required)

3. For the generated Initial restrictions, go through the OntoEP-LLM process.

3.1. Create an embedding for the created restriction via SentenceEmbedding/final.py.
3.2. Mapping with ontology elements via Clustering/final.py.
3.3. Generate variant candidates via Counter_Reasoning/final.py.

4. run GPT/generate_reasoning/interactive_batch.py to generate trasnformed rational text for each variant.

5. perform pruning via Hierarchical/final.py.

6. Finally, run GPT/final_restriction/interactive_batch.py to generate the final restriction.


You can see the results by running evaluation.py in the Evaluation folder.

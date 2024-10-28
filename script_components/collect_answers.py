import os 
import pandas as pd 
import json
import re
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset_path", type=str, help="The path to InstructQA")
parser.add_argument("--output_path", type=str, help="The path to the output file")
args = parser.parse_args()


model_names = ['alpaca-7b', 'flan-t5-xxl', 'gpt-3.5-turbo', 'llama-2-7b-chat']
datasets = ['hotpot_qa', 'natural_questions', 'topiocqa']
 

data_collector = []

for dataset in datasets:
    for name in model_names: 
        
        
        files = os.listdir(os.path.join(args.dataset_path, 'results', dataset, 'response'))
        pattern = f"^{dataset}_validation_c.*{name}_r-gold-passage_prompt.*qa_p-0\.95_t-0\.95_s-0\.jsonl$"
        selected_file=None
        
        for file in files: 
            if re.match(pattern, file): 
                selected_file = file
                break
        
        path_answers = os.path.join(args.dataset_path, 'results', dataset, 'response',selected_file)
        path_evaluation = os.path.join(args.dataset_path, 'human_eval_annotations', 'correctness', dataset, f"{name}_human_eval_results.json")
        
        
        answers = []

        with open(path_answers, "r") as f:     
            for line in f: 
                answers.append(json.loads(line)) 

        with open(path_evaluation, "r") as f: 
            annotations = json.load(f) 


        for answer in answers: 

            id_ = answer['id_'] 
            question = answer['question'] 
            llm_answer = answer['response'] 
            ground_truth = '/'.join(a for a in answer['answer']) 
            prompt = answer['prompt'] 
            verdict = None 


            for evaluated_answer in annotations:
                if int(id_) == int(evaluated_answer['id_']):
                    
                    verdict=evaluated_answer['annotation'] 
                    

                    data_collector.append({
                        "id" : int(id_), 
                        "model_name" : name, 
                        "dataset": dataset,
                        "question": question, 
                        "answer": llm_answer, 
                        "ground_truth": ground_truth, 
                        "prompt": prompt, 
                        "human_verdict": verdict
                    })




df = pd.DataFrame(data_collector)


df['answer'] = df['answer'].fillna('No answer')


df.to_csv(os.path.join(args.output_path,'instruct_qa_full.csv'),index=False)
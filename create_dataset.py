import yaml 
import os 


with open("config.yaml", "r") as f: 
    config = yaml.load(f, Loader=yaml.FullLoader)



input_path = config['paths']['instruct_qa_path']
output_path = config['paths']['output_path']


print("Collect answers...")
os.system(f"python3 script_components/collect_answers.py --dataset_path {input_path} --output_path {output_path}")
print("Applying corrections...")
os.system(f"python3 script_components/apply_corrections.py --output_path {output_path}")



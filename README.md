# Towards_Lighter_And_Robust_Evaluation
The corrected dataset for ECIR 2025

This is the script for the reproduction of the dataset used for evaluating correctness

## Step 1 

Download the original Instruct_QA dataset: https://github.com/McGill-NLP/instruct-qa

## Step 2 

Configure the yaml file. There are two parameters in config.yaml: 

-- instruct_qa_path : the path where you saved the instruct_qa folder

-- output_path: the path where the corrected dataset will be saved 

## Step 3 

Run the script: 

```
python3 create_dataset.py 
```

The output will contain two files: 

- instruct_qa_full.csv: the original InstructQA dataset inside a csv file 
- corrected_dataset_natural_questions.csv: the corrected subsampled of NaturalQuestions that was used in the paper 



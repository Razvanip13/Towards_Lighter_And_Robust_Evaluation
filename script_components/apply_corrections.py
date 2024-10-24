import numpy as np 
import pandas as pd 
import argparse
import os 


parser = argparse.ArgumentParser(description="")
parser.add_argument("--output_path", type=str, help="The path to the output file")
args = parser.parse_args()


# First phase of changes 

negatives_to_positives = {
    "alpaca-7b": [200, 1258, 1376, 1764, 2934, 3187, 3485, 3547, 3579], 
    "flan-t5-xxl": [3074, 3366, 156, 2789], 
    "gpt-3.5-turbo": [1021, 1147, 3510],
    "llama-2-7b-chat": [701, 1258, 1937, 2096, 2895, 2934, 3485]
}

positives_to_negatives = {
    "alpaca-7b": [441, 1147, 1290, 3045, 3366], 
    "flan-t5-xxl": [640, 1932, 2087, 3407, 3566], 
    "gpt-3.5-turbo": [335, 441, 468, 513, 732, 1258, 1290, 2087, 2292, 2416, 2690, 3045, 3321, 3366, 3433, 3566, 3579],
    "llama-2-7b-chat": [335, 2532, 2988, 3045, 3566, 3579]
}


df = pd.read_csv(os.path.join(args.output_path, 'instruct_qa_full.csv'))

df = df[df['dataset'] == 'natural_questions']
df['human_verdict'] = df['human_verdict'].apply(lambda x: 'incorrect' if x =='refuse-to-answer' else x) 



for key, ids in negatives_to_positives.items(): 
    for value in ids:
        condition = (df['id'] == value) & (df['model_name'] == key)
        df.loc[condition, 'human_verdict'] = 'correct'



for key, ids in positives_to_negatives.items(): 
    for value in ids:
        condition = (df['id'] == value) & (df['model_name'] == key)
        df.loc[condition, 'human_verdict'] = 'incorrect'


# df.to_csv('instruct_qa_natural_questions_corrected.csv', index=False)


# Second phase of changes

possible_check = [640,2789, 3217]
strange = [441, 1675, 3121]
certain_enumerations = [137, 468, 738, 1652, 2066, 3271, 3280, 931, 200]


def change_enumeration(value): 
    
    words = value.strip().split('/') 
    my_string = words[0] 
    
    for word in words[1:-1]: 
        my_string = my_string +", "+ word
        
    my_string = my_string +" and " + words[-1] 
    
    return my_string

df_copy = df.copy()
df_copy.loc[df_copy['id'].isin(certain_enumerations), 'ground_truth'] = df_copy.loc[df_copy['id'].isin(certain_enumerations), 'ground_truth'].apply(change_enumeration)

df_copy.loc[df_copy['id'] == 640, 'ground_truth'] = df_copy.loc[df_copy['id'] == 640, 'ground_truth'].apply(lambda x: 'russia') #always lowercase
df_copy.loc[(df_copy['id'] == 2789) & (df_copy['model_name'] == 'alpaca-7b'), 'human_verdict'] = df_copy.loc[(df_copy['id'] == 2789) & (df_copy['model_name'] == 'alpaca-7b'), 'human_verdict'].apply(lambda x: 'correct')


mask = df_copy['id'] == 3121
df_test = df_copy[~mask]
df_test.loc[df_test['id'] == 1675, 'question'] = df_test.loc[df_test['id'] == 1675, 'question'].apply(lambda x: 'who plays georgia in angus thongs and perfect snogging')

print(f"{len(df_test)} samples after the changes")
df_test.to_csv(os.path.join(args.output_path,'corrected_dataset_natural_questions.csv'), index=False)


import pandas as pd
from utils import *
import ast


df = pd.read_csv('../../data/4_pos_resolved_truth_cleaned.csv')

list_cols = [
    'resolved_sentence_list', 'sentence_list', 'quot_truth_list', 'link_truth_list',
    'linkname_truth_list', 'pos_resolved_sentence_list', 'pos_sentence_list'
]

for col in list_cols:
    df[col] = df[col].apply(ast.literal_eval)

df_exploded = df.explode(list_cols)

df_exploded.to_csv('../../data/5_sentences_exploded.csv')

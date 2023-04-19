import sys

sys.path.append('/root/projects/jpthesis/keygens/masterthesis/codebase/')
import pandas as pd
from utilities.utils import *
import ast
import re


folder ='../../data_files/test_samples/7th_RNN25k/'
sample = 25000



labels = ['quot_label', 'link_label', 'namelink_label']
texts = [
    'nopos_resolved_text', 'nopos_nonresolved_text', 'pos_nonresolved_text', 'pos_resolved_text','masked_resolved_text','masked_nonresolved_text'
]
if_filter = ['filter', 'nofilter']
article_categories = ['protected', 'excellent', 'readworthy']



for category in article_categories:
    df = pd.DataFrame()
    df = pd.read_csv(
        f'../../data_files/pipeline_steps/{category}_articles/7.1_maskadded.csv')
    df_equal_sample = df.copy()
    df_equal_sample = df_equal_sample.sample(n=(sample * 2),random_state=2)

    for label in labels:
        df_random = pd.DataFrame()
        df_equal = pd.DataFrame()

        for text in texts:
            for to_filter in if_filter:
                name = f"{text[:-5]}_{label[:-6]}_{to_filter}"
                if to_filter == 'filter' and not text[:3] == 'pos':
                    pattern = r'_([^_]*)_'
                    match = re.search(pattern, text)
                    name = f"pos_{match.group(1)}_{label[:-6]}_filter"
                df_filter_random = df.copy()
                # df_filter_equal = df_equal_sample.copy()
                if to_filter == 'filter':
                    df_filter_random = df_filter_random.query(
                        f'not ({label} == False and {name} == True)')
                    # df_filter_equal = df_filter_equal.query(f'not ({label} == False and {name} == True)')
                    name = f"{text[:-5]}_{label[:-6]}_{to_filter}"
                df_random_grouped = df_filter_random.groupby(label)
                df_random_sample = df_random_grouped.apply(
                    lambda x: x.sample(n=sample,random_state = 2)).reset_index(drop=True)

                df_random['label'] = df_random_sample[label]
                # df_equal['label'] = df_equal_sample[label]

                df_random['text'] = ''
                # df_equal['text'] = ''

                df_random['text'] = df_random_sample[text]
                # df_equal['text'] = df_equal_sample[text]

                df_random = df_random.sample(frac=1)
                # df_equal = df_equal.sample(frac=1)

                df_random.to_csv(f'{folder}{category}_{name}.csv')
                # df_equal.to_csv(f'../../data_files/test_samples/4th_test_dataset_samples/{name}_equal.csv')

""" Filters overlapping pos-sequence from the non-claim class.
1. Loops through text & label
2. Groups by pos_tag
3. Adds True is pos-seq is present in both classes
4. for each pos-text <-> label combination adds one column:
    - {text}_{label}_filter 
    """


import pandas as pd

from back_classes.utils import *

df_read = pd.read_csv('../../data/5.1_sentences_exploded.csv')
df = df_read.copy()
pos_list=['pos_nonresolved_text', 'pos_resolved_text']

label_list = ['quot_label', 'link_label', 'namelink_label']

for pos_text in pos_list:

    for label in label_list:
        def pos_union_func(x):
            # count the number of unique labels for this pos_seq
            label_count = len(set(x[label]))
            # if there is only one unique label, return True
            if label_count == 1:
                return False
            if label_count == 2:
                return True

        name=f"{pos_text[:-5]}_{label[:-6]}_filter"
        # group by pos_seq and apply the custom aggregation function to calculate pos_union
        pos_union = df.groupby(pos_text).apply(pos_union_func).reset_index(name=name)
        
        # merge the pos_union back into the original dataframe
        df = df.merge(pos_union, on=pos_text, how='left')

df.to_csv('../../data/6.1_sentences_filtersadded.csv')


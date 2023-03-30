import sys
print(sys.path)

import pipe_1_fetch
import pipe_2_split
import pipe_3_resolve
import pipe_4_postag
import pipe_5_explode
import pipe_6_posfilter

article_list = '../../../data_files/pipeline_steps/0_all_protected_wiki_list.csv'
folder='../../../data_files/test_pipeline'



pipe_1_fetch.main(article_list, folder)
print("Pipeline Step 1 finished!! ")
pipe_2_split.main(folder)
print("Pipeline Step 2 finished!! ")
pipe_3_resolve.main(folder)
print("Pipeline Step 3 finished!! ")
pipe_4_postag.main(folder)
print("Pipeline Step 4 finished!! ")
pipe_5_explode.main(folder)
print("Pipeline Step 5 finished!! ")
pipe_6_posfilter.main(folder)
print("Pipeline Step 6 finished!! ")
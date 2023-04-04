import sys
import pipe_1_fetch
import pipe_2_split
import pipe_3_resolve
import pipe_4_postag
import pipe_5_explode
import pipe_6_posfilter

article_list = '../../../data_files/pipeline_steps/readworthy_articles/0_readworthy_titles.csv'
folder = '../../../data_files/pipeline_steps/readworthy_articles/'
suffix = '200'
nrows = 200

pipe_1_fetch.main(article_list, folder, suffix, nrows)
print("Pipeline Step 1 finished!! ")
pipe_2_split.main(folder, suffix)
print("Pipeline Step 2 finished!! ")
pipe_3_resolve.main(folder, suffix)
print("Pipeline Step 3 finished!! ")
pipe_4_postag.main(folder, suffix)
print("Pipeline Step 4 finished!! ")
pipe_5_explode.main(folder, suffix)
print("Pipeline Step 5 finished!! ")
pipe_6_posfilter.main(folder, suffix)
print("Pipeline Step 6 finished!! ")
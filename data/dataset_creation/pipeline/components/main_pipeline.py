import sys
sys.path.append('/root/projects/jpthesis/keygens/masterthesis/codebase/')
import pipe_1_fetch
import pipe_2_split
import pipe_3_resolve
import pipe_4_postag
import pipe_5_explode
import pipe_6_posfilter
import pipe_7_mask
folder = '../../../data_files/pipeline_steps_reresolve/random_articles/'
article_list = f'{folder}0_random_titles.csv'
suffix = ''
nrows = None
print("starting to process: "+ folder)
# pipe_1_fetch.main(article_list, folder, suffix, nrows)
# print("Pipeline Step 1 finished!! ")
# pipe_2_split.main(folder, suffix)
# print("Pipeline Step 2 finished!! ")
pipe_3_resolve.main(folder, suffix)
print("Pipeline Step 3 finished!! ")
pipe_4_postag.main(folder, suffix)
print("Pipeline Step 4 finished!! ")
pipe_5_explode.main(folder, suffix)
print("Pipeline Step 5 finished!! ")
pipe_6_posfilter.main(folder, suffix)
print("Pipeline Step 6 finished!! ")
pipe_7_mask.main(folder, suffix)
print("Pipeline Step 7 finished!! ")

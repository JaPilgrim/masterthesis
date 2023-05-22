import spacy
import pandas as pd

df = pd.read_csv('pipeline_steps/random_articles/6.1_sentences_filtersadded.csv')
print(str(len(df))+" long...___")
nlp = spacy.load('de_core_news_sm', disable=['tagger', 'parser'])


def replace_entities_with_types(text):
    modified_tokens = []
    for token in text:
        if token.ent_type_:
            modified_tokens.append(token.ent_type_)
        else:
            modified_tokens.append(token.text)
    modified_text = ' '.join(modified_tokens)
    return modified_text


nonresolved_sentences = df['nopos_nonresolved_text']
docs = nlp.pipe(nonresolved_sentences)
modified_nonresolved_sentences = []
count = 0
for doc in docs:
    modified_nonresolved_sentences.append(replace_entities_with_types(doc))
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} nonresolved_sentences.")
if len(modified_nonresolved_sentences) != len(nonresolved_sentences):
    raise KeyError
df['masked_nonresolved_text'] = modified_nonresolved_sentences

resolved_sentences = df['nopos_resolved_text']
docs = nlp.pipe(resolved_sentences)
modified_resolved_sentences = []
count = 0
for doc in docs:
    modified_resolved_sentences.append(replace_entities_with_types(doc))
    count += 1
    if count % 50000 == 0:
        print(f"Processed {count} resolved_sentences.")
if len(modified_resolved_sentences) != len(resolved_sentences):
    raise KeyError
df['masked_resolved_text'] = modified_resolved_sentences

df.to_csv('pipeline_steps/random_articles/7.1_maskadded.csv', index=False)

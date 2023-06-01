import spacy
import pandas as pd




def replace_entities_with_types(text):
    modified_tokens = []
    for token in text:
        if token.ent_type_:
            modified_tokens.append(token.ent_type_)
        else:
            modified_tokens.append(token.text)
    modified_text = ' '.join(modified_tokens)
    return modified_text

def main(folder:str,suffix=''):

    df = pd.read_csv(f'{folder}6.1_sentences_filtersadded{suffix}.csv')
    print(str(len(df))+" long...___")
    nlp = spacy.load('de_core_news_sm', disable=['tagger', 'parser'])
    nonresolved_sentences = df['nopos_nonresolved_text']

    modified_nonresolved_sentences = []
    count = 0
    for doc in nlp.pipe(nonresolved_sentences):
        modified_nonresolved_sentences.append(replace_entities_with_types(doc))
        count += 1
        if count % 50000 == 0:
            print(f"Processed {count} nonresolved_sentences.")
    if len(modified_nonresolved_sentences) != len(nonresolved_sentences):
        raise KeyError
    df['masked_nonresolved_text'] = modified_nonresolved_sentences

    resolved_sentences = df['nopos_resolved_text']

    modified_resolved_sentences = []
    count = 0
    for doc in nlp.pipe(resolved_sentences):
        modified_resolved_sentences.append(replace_entities_with_types(doc))
        count += 1
        if count % 50000 == 0:
            print(f"Processed {count} resolved_sentences.")
    if len(modified_resolved_sentences) != len(resolved_sentences):
        raise KeyError
    df['masked_resolved_text'] = modified_resolved_sentences

    df.to_csv(f'{folder}7.1_maskadded{suffix}.csv', index=False)

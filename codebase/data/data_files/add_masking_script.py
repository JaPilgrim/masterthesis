import spacy
import pandas as pd

df = pd.read_csv('pipeline_steps/protected_articles/6.1_sentences_filtersadded.csv')

nlp = spacy.load('de_core_news_sm',disable=['tagger','parser'])



def replace_entities_with_types(text):
    # Initialize an empty list to store the words (tokens) in the text
    modified_tokens = []

    # Iterate through the tokens in the parsed document
    for token in text:
        # Check if the token is a named entity
        if token.ent_type_:
            # Replace the token with its entity type
            modified_tokens.append(token.ent_type_)
        else:
            # If it's not an entity, keep the original token
            modified_tokens.append(token.text)

    # Join the tokens back into a single string
    modified_text = ' '.join(modified_tokens)
    return modified_text



nonresolved_sentences = df['nopos_nonresolved_text']
docs = nlp.pipe(nonresolved_sentences)
modified_nonresolved_sentences = [replace_entities_with_types(doc) for doc in docs]
if len(modified_nonresolved_sentences) != len(nonresolved_sentences):
    raise KeyError
df['masked_nonresolved_text'] = modified_nonresolved_sentences



resolved_sentences = df['nopos_resolved_text']
docs = nlp.pipe(resolved_sentences)
modified_resolved_sentences = [replace_entities_with_types(doc) for doc in docs]
if len(modified_resolved_sentences) != len(resolved_sentences):
    raise KeyError
df['masked_resolved_text'] = modified_resolved_sentences

df.to_csv('pipeline_steps/protected_articles/7.1_maskadded.csv', index=False)

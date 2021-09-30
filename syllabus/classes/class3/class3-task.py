#### QUESTIONS ####
#- Best way to use the md file? bc this is not it...
#- Why does "def corpus_loader(folder: str) -> List[str]:" not work for me? - it says "List not defined"
#pip install spacy
# install smaller model
#python -m spacy download en_core_web_sm
#cd NLP-E21/syllabus/classes/class3/

#doc:
# `lemma_` which denotes the lemma of the token, 
# `pos_` which denotes part-of-speech tag of the token and 
# `ent_type_` which denote the entity type of the token.

#**Exercise 1**:
#Inspect the `doc`-object using `dir` and `type` along with the [documentation](https://spacy.io/api/Doc). 
# You should before class have though 

# i) about what is intended use (or benefit) of the `doc`-object 
'''
it contains the raw text as well as the annotations which allows 
us to do cool analyses on texts without us having to do that ourselves 
- i.e. we can "move on" to the actual analyses instead of spending a lot of time on annotations
'''
# ii) What are the two ways in which I can create an `Doc` object?
'''
# Construction 1 - using nlp
doc = nlp("Some text")
print(doc)
-- the nlp pipeline takes in text and create component like; tokenizer, parser, NER, POS-tagger, etc. 
- tokenizer: split sentence into tokens
- parser: creates dependency tree --> you can recreate the sentences using this parser
- ner: named entity recognition - tag names as names
- POS-tagger: verb, noun, etc. 
- and possibly many more... (you can keep adding components to the doc class)

# Construction 2 - create words and spaces and collect in a doc object 
from spacy.tokens import Doc
words = ["hello", "world", "!"]
spaces = [True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces) - classes are typically upper case
'''

### Corpus loader
#Before the class, you should have a corpus loader ready. This should be able to read in each 
# file in the folder `syllabus/classes/data/train_corpus` as a list of strings (or similar object).

import os

#def corpus_loader(folder: str) -> List[str]:
def corpus_loader(folder):
    """
    A corpus loader function which takes in a path to a 
    folder and returns a list of strings.
    """
    corpus = []
    # - 1) Loading in the files as text.
    for file in os.listdir(folder):
        current_file = folder+str(file)
        with open(current_file) as f:
            contents = f.read()
            corpus.append(contents)
            f.close()
    return corpus

texts = corpus_loader("/work/NLP-E21/syllabus/classes/data/train_corpus/")

## Plan for class

#- 1) Talk about exercise 1
#- 2) Filter a text to keep only the lemma of nouns, adjectives and verbs
import spacy
nlp = spacy.load("en_core_web_sm")

# docs is a list of doc objects
docs = [nlp(t) for t in texts]

one_doc = docs[0]

def noun_adj_verb_filter(doc):
    lemmas = []
    for tok in doc:
        if tok.pos_ in ['NOUN','VERB','ADJ']:
            lemmas.append(tok.lemma_)
    return lemmas

all_lemmas = [noun_adj_verb_filter(doc) for doc in docs]

# 3) Calculate the ratio of pos-tags in texts. 
'''
from collections import Counter
pos_counts = Counter([token_pos_ for token in doc]) # return dictionary with how many 
list(zip([j for j in pos_counts.keys()],[i/len(doc) for i in pos_counts.values()]))

[(pos,count/len(doc)) for pos,count in pos_counts.items()]
'''
# 4 Calculate mean dependency distance (MDD)
'''
token: "reporter"
token.head: "admitted"
take indices from each and subtract them

for i,token in enumerate(doc):
    #gives us index and token for all indices in the doc

then subtract the two: abs(tokeni-token.head)

pos_counts = Counter([token.pos_ for token in doc])
[(pos, count/len(doc)) for pos, count in pos_counts.items()]
'''

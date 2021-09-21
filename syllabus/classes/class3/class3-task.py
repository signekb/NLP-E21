#pip install spacy

# install smaller model
#python -m spacy download en_core_web_sm

# install larger model
#python -m spacy download en_core_web_lg

#cd NLP-E21/syllabus/classes/class3/
'''
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is an English text")

# We can expect this quite easily using, where we see that the output 
# if the pipeline is a document class called `Doc`:
print(type(doc))

# And that if you index the `Doc` you get a `Token` object.
token = doc[1]
print(token)
# is
print(type(token))

#What does this token contain? Well we can (like any other object in python) inspect it using the `dir`:
print(dir(token))
#Which gives is a very long list of which should look something like this: 
# `['_', '__bytes__', (...), 'is_digit', 'is_punct', 'whitespace_']` of all the attributes of 
# the function. We could also just have looked at the [documentation](https://spacy.io/api/token), 
#but that wouldn't teach you how to inspect a class. 
# For example we can now check whether a token is a digit:

print(token.is_digit)
# False

#You might also find a couple of other interesting things in there especially the 
# `lemma_` which denotes the lemma of the token, 
# `pos_` which denotes part-of-speech tag of the token and 
# `ent_type_` which denote the entity type of the token.
'''

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

# Construction 2 - create words and spaces and collect in a doc object 
from spacy.tokens import Doc

words = ["hello", "world", "!"]
spaces = [True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
#We will talk about this exercise as the first thing in the class.
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
            corpus.append([file,contents])
            f.close()
    # - 2) Segment the sentences in the text.
    for idx,file in enumerate(corpus):
        current_txt = file[1].split(".")
        corpus[idx].append([current_txt])

    # - 3) Tokenize each sentence.
    for idx,file in enumerate(corpus):
        current_txt = file[1].split()
        corpus[idx].append([current_txt])

    return corpus

corpus = corpus_loader("/work/NLP-E21/syllabus/classes/data/train_corpus/")

## Plan for class

#- 1) Talk about exercise 1
#- 2) Filter a text to keep only the lemma of nouns, adjectives and verbs

'''
<details>
    <summary> Deconstruction of the task </summary>

The task can meaningfully be deconstructed into a series of functions on the token level:
- A filter function, which decided if a token should be kept.
- A function which extract the lemma

These function can then be combined and used iteratively over the tokens of a document.


</details>

<br /> 

- 3) Calculate the ratio of pos-tags in texts. The ratios of pos-tags on other linguistic feature have for example been [linked](https://www.nature.com/articles/s41537-021-00154-3) to scizophrenia which e.g. use less adverbs, adjectives, and determiners (e.g., “the,” “a,”).

<details>
    <summary> Deconstruction of the task </summary>

The task can meaningfully be deconstructed into a series of functions:
- A function (or list comprehension) which takes a list of tokens (Doc) and extracts the pos tag for each
- A function which counts these. *Hint* look up the `Counter` class.

</details>

<br /> 
'''
- 4) If you get the time calculate PMI (see last weeks class) using the tokenization and sentence segmentation of spaCy.
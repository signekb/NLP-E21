import gensim.downloader as api
import numpy as np

model = api.load("glove-wiki-gigaword-50")

# 1) Calculate the dot product between two word embedding which you believe are similar
np.dot(model["fork"],model["knife"])

# 2) Calculate the dot product between the two word and a word which you believe is dissimilar
np.dot(model["fork"],model["mud"])

''' 
Higher dot product = more similar words
'''

# 3) make the three words into a matrix $E$ and multiply it with its own transpose
# using matrix multiplication. So $E \cdot E^T$
'''
- what does the values in matric correspond to? What do you imagine the dot product is? 
*Hint*, similarity between vectors (cosine similarity) is exactly the same as the dot product assuming you normalize the lenghth of each vector to 1.
'''

E = np.array([model["fork"],model["knife"],model["mud"]])

np.dot(E,E.transpose())

'''
How similar are the vectors? All calculated at the same time. 
'''

# Examine the attention formula from Vaswani et al. (2017), 
# you have now implemented $Q\cdot K^T$
'''
$$ Attention(Q, K, V) = softmax(\frac{Q\cdot K^T}{\sqrt{d}}) \cdot V $$ Where $d$ is the dimension of of the embedding and Q, K, V stands for queries, keys and values.
Where $d$ is the dimension of of the embedding and Q, K, V stands for queries, keys and values.
'''
d = len(model["fork"])
Q = E
K = E
V = np.dot(E,E.transpose())

attention = np.dot(nn.Softmax(np.dot(Q,K.transpose()/np.sqrt(d))),V)



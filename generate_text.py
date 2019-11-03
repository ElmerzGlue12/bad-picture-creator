from cfgen import GrammarModel # https://github.com/williamgilpin/cfgen
import nltk
from random import seed

seed(100)
PATH = 'wah.txt' # path of text file to create model with
model = GrammarModel(PATH , 2) # create model with 2 layers

for ii in range(1):
    print(model.make_sentence(do_markov=True)) # print generated sentence

###
##Clean the data
    #Split tokens on white space
    #Remove all puncutation
    #Removal all words that are not purly comprised of alpha
    #Remove all words that are known stop words
    #Remove all words that have len <= 1

#################################################################
#we need to create our vocabulary for our BoW model
import string
import os 
from collections import Counter
from nltk.corpus import stopwords

def load_data(filepath):
    file = open(filepath, 'r')
    text = file.read()
    file = file.close()
    return text

def clean_doc(text):
    tokens = text.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

#iterate through all the movie reviews and add the vocabs 
def process_docs(filepath, vocab, is_train):
    for file in os.listdir(filepath):
        if is_train and file.startswith('cv9'):
            continue
        if not is_train and not file.startswith('cv9'):
            continue
        path = filepath +"/"+file
        text = load_data(path)
        tokens = clean_doc(text)
        ##update vocab
        vocab.update(tokens)

vocab = Counter()

process_docs(filename_pos, vocab, True)
process_docs(filename_neg, vocab, True)

def remove_infrequent_words(vocab, value):
    vocab = [k for k,c in vocab.items() if c>= value]
    return vocab

tokens = remove_infrequent_words(vocab, 2)
print(len(tokens))

#save our vocab to use in our actual model
def save_vocab(vocab, filepath):
    data = '\n'.join(vocab)
    file = open(filepath, 'w')
    file.write(data)
    file.close()

filepath = 'token_list.txt'
save_vocab(tokens, filepath)


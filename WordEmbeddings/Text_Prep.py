import string

#load tokens
def load_token(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#extract descriptions from images
def load_descriptions(tokens):
    mapping = dict()
    for line in tokens.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        #see if image is in the mapping dictionary
        if image_id not in mapping:
            mapping[image_id] = list()
        #add to the oject with key as image id but value being a list of the description
        mapping[image_id].append(image_desc)
    return mapping 

#clean the descriptions
def clean_descriptions(mapping):
    table = str.maketrans('','',string.punctuation)
    for key, desc_list in mapping.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

#conver loaded descriptions into a vocabulary of words
def to_vocab(mapping):
    all_desc = set()
    for key in mapping.keys():
        [all_desc.update(d.split()) for d in mapping[key]]
    return all_desc

#save descriptiosn
def save_description(mapping, output):
    lines = list()
    for key, desc_list in mapping.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(output, 'w')
    file.write(data)
    file.close()

filename = '/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/photo_captioning/FlickerText/Flickr8k.token.txt'
text = load_token(filename)
mapping = load_descriptions(text)
clean_descriptions(mapping)
vocab = to_vocab(mapping)
print(len(vocab))

#savefile
save_description(mapping, 'mapping.txt')
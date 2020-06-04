##Using cnn database to go over the stories and to summarize the texts provided
##first step is to do data clealing, such as:
    #normalice lower case
    #Remove punctuation
    #Remove low-frequency words

#########################################
import os
import string
from pickle import dump, load

#Function to load all the data at once
def load_storie(filepath):
    stories = list()
    for file in os.listdir(filepath):
        path = filepath + "/" + file
        openFile = open(path, encoding="utf-8")
        text = openFile.read()
        openFile.close()
        story, highlights = split_story(text)
        stories.append({"story":story,"highlight":highlights})
    return stories


#Function to perform some cleaning on each individual text
def split_story(text):
    index = text.find('@highlight')
    story,highlights = text[:index], text[index:].split('@highlight')
    highlights = [h.strip() for h in highlights if len(h) > 0]
    story = clean_story(story.split('\n'))
    highlights = clean_story(highlights)
    return story, highlights

def clean_story(story):
    cleaned_story = list()
    table  = str.maketrans('','',string.punctuation)
    for line in story:
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN'):]
        line = line.split()
        line = [word.lower() for word in line]
        line = [word.translate(table) for word in line]
        line = [word for word in line if word.isalpha()]
        cleaned_story.append(' '.join(line))
    cleaned_story = [c for c in cleaned_story if len(c) > 0]
    return cleaned_story


filepath = "/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/cnn/stories"
stories = load_storie(filepath)

#Save cleaned data
dump(stories, open('cnn_dataset.pkl','wb'))


poop = load(open('cnn_dataset.pkl','rb'))



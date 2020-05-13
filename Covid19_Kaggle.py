import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

dirs = ["biorxiv_medrxiv", "comm_use_subset","noncomm_use_subset"]

docs = list()

#go through files in folders & attempt to structure it 
for d in dirs:
    for file in tqdm(os.listdir(f"{d}/{d}")):
        for new_file in os.listdir(f"{d}/{d}/{file}"):
            file_path = f"{d}/{d}/{file}/{new_file}"
            j = json.load(open(file_path, "rb"))

            try:
                #print(j['abstract'][0]['text'])
                abstract = j['abstract'][0]['text']
            except: 
                abstract =[]

            text =j['body_text']

            full_text = ""

            for text in text:
                full_text += text['text'] + '\n\n'


            #print(full_text)
            docs.append([full_text, abstract])

            

df = pd.DataFrame(docs, columns=['full text', "abstract"])
#print(df.head())

#Search documentation that contains incubation
incubation = df[df['full text'].str.contains('incubation')]
print(incubation.head())

#Only save information from full text
texts = incubation['full text'].values

incubation_times = list()

#How do we process days, hours, minutes? There is a lot of poor information here as well
for t in texts:
    for sentence in t.split(". "):
        if "incubation" in sentence:
            single_day = re.findall(r" \d{1,2}\.?\d{1,2} day", sentence)
            if len(single_day) == 1 :
                #print(single_day[0].split(' ')[1])
                incubation_times.append(float(single_day[0].split(' ')[1]))

print(incubation_times)
print(len(incubation_times))

print(f"the mean projected incubation time is {np.mean(incubation_times )}")

plt.hist(incubation_times, bins=10)
plt.ylabel("Bin counts")
plt.xlabel("incubation days ")
plt.show()








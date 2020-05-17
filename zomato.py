

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting=3)

import re
import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

"Replacing puncuations and numbers using re library"
data=[]
for i in range(0,1000):
    review=dataset["Review"][i]
    review= re.sub('[^a-zA-Z]',' ',review)
    #stemming    
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
i+=1
data
df = pd.DataFrame(data)
dataset.to_csv("Restaurant_Reviews.csv", index=False)

pf=pd.read_csv('Restaurant_Reviews.csv')





#convert lower case.
d=[]
for i in range(0,1000):
    lower_case=data[i]
    lower_case=lower_case.lower()
    
    d.append(lower_case)
data=d


#


dataset["Review"]=data


import pickle




from  sklearn.feature_extraction.text import CountVectorizer
#removing less repeated words
cv =CountVectorizer(max_features=1500)
X=cv.fit_transform(d).toarray()
y= dataset.iloc[:,-1].values

#split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#save the model.
pickle.dump(cv.vocabulary_,open("resturent.pkl","wb"))











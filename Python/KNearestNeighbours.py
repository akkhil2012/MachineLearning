### K nearest neighbours
# why do svm scales better than k nearest
#https://www.youtube.com/watch?v=1i0zu9jHN6U&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=14

# where i=1.......n (dimensions)

import numpy as np

from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

import pandas as pd

import random

style.use('fivethirtyeight')


dataset = {'k':[[1,3],[2,3],[3,3]],'r':[[6,3],[7,3],[5,3]]}

new_features=[4,3]
#euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2) ## Two dimensions Distances	
def k_nearest_neighbours(data,predict,k=3):
    if len(data)>=k:
	     warnings.warn('K is set to a value less than total voting groups!!!!')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])	
    votes = [i[1] for i in sorted(distances)[:k]] 	
    print(Counter(votes).most_common(1))	
    vote_result=Counter(votes).most_common(1)[0][0]
    confidence=Counter(votes).most_common(1)[0][1]/k  #### How does confidence level affects performance
    return vote_result
	
######## To read the cancer dataSet 
#df = pd.read_csv('cancerDataSet.csv')## worked when specified return character and endoding
with open('C:\\PMLExecises\\breast-cancer-wisconsin.data.txt', 'r', encoding='UTF-8') as f:
    df = pd.read_csv(f)
   # print(df.head())

#print(df.head())
   
df.replace('?',-99999, inplace=True)

df.drop(['id'],1,inplace=True)


full_data = df.astype(float).values.tolist() 

#print(full_data[:1])

random.shuffle(full_data)

test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}


train_data = full_data[:-int(test_size*len(full_data))]

test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    train_set[i[-1]].append(i[:-1]) 


for i in test_data:
    test_set[i[-1]].append(i[:-1])


correct = 0
total = 0 


for group in test_set:
    for data in test_set[group]:
       vote,confidence = k_nearest_neighbours(train_set,data,k=200)	 ## Increasing K value doenot make much difference
       if group==vote:
	       correct += 1
	   else:
           print(confidence) 	  
       total += 1

print('Accuracy: ', correct/total)
	


#print(20*'#')

#print(full_data[:1])

#print(full_data[:10])
	
#print(df.head())	




	
############### input features are just numerical data points ###########	
#result = k_nearest_neighbours(dataset,new_features,k=3)

#print(result)
	
	
#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1],color=result)
#plt.show()	
	
#plot1=[1,3]
#plot2=[2,5]


#euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2) ## Two dimensions Distances

#print(euclidean_distance)

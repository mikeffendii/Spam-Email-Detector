import os
import nltk
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import joblib

X_data = pickle.load(open('X_data.pkl','rb'))
Y_data = pickle.load(open('Y_data.pkl','rb'))

train_features,test_features,train_lables,test_lables = train_test_split(X_data,Y_data,test_size=0.3,random_state=30)

RFC = RandomForestClassifier(n_jobs=-1, n_estimators=1000, oob_score=True)
RFC.fit(train_features,train_lables)
print('=============Saving Model=============')
joblib.dump(RFC,'Detector_Model.m')

Out_lables = RFC.predict(test_features)
# print(Out_lables)
# print(test_lables)
train_num = len(train_features)
test_num = len(test_features)
total_num = train_num+test_num
print('train_num :{} test_num :{} total_num :{}'.format(train_num,test_num,total_num))


right = 0

TP = 0
FP = 0
TN = 0
FN = 0
for i_la,o_la in zip(test_lables,Out_lables):
    if i_la == o_la:
        right+=1
    if i_la =='SPAM' and i_la ==o_la:
        TP += 1
    elif i_la =='SPAM' and i_la !=o_la:
        FN +=1
    elif i_la =='LEGAL' and i_la ==o_la:
        TN +=1
    elif i_la =='LEGAL' and i_la !=o_la:
        FP +=1
    
print('====================================')
print('Accuracy：{}'.format(right/len(Out_lables)))

print('Precision：{}'.format(TP/(TP+FP)))
print('Recall:{}'.format(TP/(TP+FN)))

test_list = []
out_list = []
for lable in test_lables:
    if lable =='SPAM':
        test_list.append(1)
    else:
        test_list.append(0)

for lable in Out_lables:
    if lable =='SPAM':
        out_list.append(1)
    else:
        out_list.append(0)

precision_list , recall_list ,thresholds= precision_recall_curve(test_list,out_list,pos_label=1)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
plt.plot(recall_list, precision_list, marker='.', label='Random forest',linewidth=3)
plt.xlabel('Recall',fontdict={'weight':'normal','size': 30})
plt.ylabel('Precision',fontdict={'weight':'normal','size': 30})
plt.show()


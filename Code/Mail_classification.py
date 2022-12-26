import os
import nltk
import pickle
from sklearn.ensemble import RandomForestClassifier

base_path=os.path.dirname(os.path.abspath(__file__))

in_path = os.path.abspath(os.path.join(os.path.join(base_path,'Cleaned_email'),'cleaned_email_text.txt'))
#----------------------------------
train_num = 10000
#----------------------------------
temp = input('renew data_dict?(y/n) THE FITST TIME INPUT Y :')
data_dict = []

X_data = []
Y_data = []

if temp in ['y','Y']:

    data_dict ={
        'train_features':[],
        'train_lables':[],
        'test_features':[],
        'test_lables':[],
    }

    print('==================Read WORD_DICTIONARY===================')
    word_dict = pickle.load(open('WORD_DICTIONARY.pkl','rb'))
    print('The length of WORD_DICTIONARY is {}'.format(len(word_dict)))

    print('==========Read Email=========')
    with open(in_path,'r') as emails:
        coun_ = 0
        Spam_num = 0
        Lega_num = 0
        Spam_train_num = 0
        Spam_test_num = 0
        Lega_train_num = 0
        Lega_test_num = 0
        for email in emails.readlines():

            word_list = nltk.tokenize.word_tokenize(email) 
            # print(word_list)
           
            print('==========Get features=========')
            feature = [word_list.count(word) for word in word_dict]

            X_data.append(feature)
            Y_data.append(word_list[-1])

            if word_list[-1] == 'SPAM':
            
                if Spam_num < train_num:
                    data_dict['train_features'].append(feature)
                    data_dict['train_lables'].append(word_list[-1])
                    Spam_train_num +=1
                else:
                    data_dict['test_features'].append(feature)
                    data_dict['test_lables'].append(word_list[-1])
                    Spam_test_num +=1
                Spam_num +=1

                
            if word_list[-1] == 'LEGAL':

                if Lega_num < train_num:
                    data_dict['train_features'].append(feature)
                    data_dict['train_lables'].append(word_list[-1])
                    Lega_train_num += 1
                else:
                    data_dict['test_features'].append(feature)
                    data_dict['test_lables'].append(word_list[-1])
                    Lega_test_num += 1
                Lega_num +=1
        
            print('Spam training num,testing num,totalnum = {} : {} : {}'.format(Spam_train_num,Spam_test_num,Spam_num))
            print('Lega training num,testing num,totalnum = {} : {} : {}'.format(Lega_train_num,Lega_test_num,Lega_num))
    
    print('==========Save training samples and test samples as well as corresponding labels=========')
    pickle.dump(data_dict,open('data_dict.pkl','wb'))
    print('==========Save all samples and corresponding labels=========')
    pickle.dump(X_data,open('X_data.pkl','wb'))
    pickle.dump(Y_data,open('Y_data.pkl','wb'))
else:
    print('==========Read training samples and test samples as well as corresponding labels=========')
    data_dict = pickle.load(open('data_dict.pkl','rb'))

print('==========Training model=========')
RFC = RandomForestClassifier(n_jobs=-1, n_estimators=1000, oob_score=True)
RFC.fit(data_dict['train_features'],data_dict['train_lables'])

Out_lables = RFC.predict(data_dict['test_features'])
# print(Out_lables)
# print(data_dict['test_lables'])

right = 0
TP = 0
FP = 0
TN = 0
FN = 0
for i_la,o_la in zip(data_dict['test_lables'],Out_lables):
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

print('\nAccuracy:{}'.format(right/len(Out_lables)))
print('Precision:{}'.format(TP/(TP+FP)))
print('Recall:{}'.format(TP/(TP+FN)))
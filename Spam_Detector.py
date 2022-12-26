import os 
import nltk
import joblib
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
WNL = WordNetLemmatizer()

base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path,'Email_list') 

word_dict = pickle.load(open('WORD_DICTIONARY.pkl','rb'))

RFC = joblib.load('Detector_Model.m')

email_vector_list = []
for email_name in os.listdir(input_path):
    with open(os.path.join(input_path,email_name),'r',encoding='utf-8',errors='ignore') as email_text:
        email = email_text.read()
        word_list = word_tokenize(email)
        stop_words = stopwords.words('english')
        word_list_1 = [WNL.lemmatize(WNL.lemmatize(word.lower(),'v'),'n') for word in word_list if word.isalnum() and not word in stop_words]
        
        feature = [word_list_1.count(word) for word in word_dict]
        #print(feature)
        lable = RFC.predict([feature])
        if lable[0] == 'SPAM':
            print('{} is a junk mail'.format(email_name))
        elif lable[0] =='LEGAL':
            print('{} is a legitimate email'.format(email_name))

            



        
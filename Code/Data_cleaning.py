import os 
import nltk
import shutil
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

WNL = WordNetLemmatizer()

base_path = os.path.dirname(os.path.abspath(__file__))

out_folder_path = os.path.abspath(os.path.join(base_path,'Cleaned_email'))
in_folder_path = os.path.join(base_path,'Samples')

if not os.path.exists(out_folder_path):
    os.mkdir(out_folder_path)
    print('Create Formated_sample')
else:
    shutil.rmtree(out_folder_path)
    os.mkdir(out_folder_path)
    print('Create Formated_sample')

sample_name_list = os.listdir(in_folder_path)
L_num = 0
S_num = 0
with open(os.path.join(out_folder_path,'cleaned_email_text.txt'),'w',encoding='utf-8') as F_emai:

    for sample_name in sample_name_list:
        # print(sample_name)
        with open(os.path.join(in_folder_path,sample_name),'r',encoding='gb18030',errors='ignore') as email:
            text = email.read()
            word_list = nltk.tokenize.word_tokenize(text)             
            stop_words = stopwords.words('english')
            word_list_orinal = [word.lower() for word in word_list if word.isalnum() and not word in stop_words]
            word_list_1 = [WNL.lemmatize(WNL.lemmatize(word.lower(),'v'),'n') for word in word_list if word.isalnum() and not word in stop_words]
           
            for word in word_list_1:
                F_emai.write(word+' ')
            if sample_name[0] =='L':
                F_emai.write('LEGAL\n')  
                L_num += 1
                print('{} IS not SPAM'.format(sample_name))
            elif sample_name[0] == 'S':
                F_emai.write('SPAM\n')
                S_num += 1
                print('{} IS  SPAM'.format(sample_name))

print('Spam num {} Legal num {}'.format(S_num,L_num))

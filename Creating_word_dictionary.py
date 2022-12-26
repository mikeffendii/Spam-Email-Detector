import os 
import pickle
import nltk
#----------------------------------
word_frequency = 100 
#----------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

in_path = os.path.abspath(os.path.join(os.path.join(base_path,'Cleaned_email'),'cleaned_email_text.txt'))

with open(in_path,'r') as email:
    text=email.read()
    text=nltk.tokenize.word_tokenize(text)
    word_dict1 = dict(nltk.FreqDist(text))
    
    dict_word = []
    for key,value in word_dict1.items():
        if value > word_frequency:
            dict_word.append(key)
            # print('({} {})'.format(key,value))
            print(len(dict_word))

    dict_word.remove('SPAM') 
    dict_word.remove('LEGAL')
    dict_word.remove('subject')
    pickle.dump(dict_word,open('WORD_DICTIONARY.pkl','wb'))
    print('==========================================')
    print('The length of WORD_DICTIONARY is {}'.format(len(dict_word)))



    
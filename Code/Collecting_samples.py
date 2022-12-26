import os
import shutil

Sample_folder_name = 'Samples'

base_name = os.path.dirname(os.path.abspath(__file__))

Spam_folder_name = 'Spam'
Legi_folder_name = 'Legitimate'
Folder_list = [Spam_folder_name,Legi_folder_name]

Out_path = os.path.join(base_name,Sample_folder_name)

if not os.path.exists(Out_path):
    os.mkdir(Out_path)
else:
    shutil.rmtree(Out_path)
    os.mkdir(Out_path)

for folder_name in Folder_list:
    path_ = os.path.join(base_name,folder_name)
    file_name_list = os.listdir(path_)
    count_ = 0
    for file_name in file_name_list:
        shutil.copyfile(os.path.join(path_,file_name),os.path.join(Out_path,folder_name[0]+'-'+str(count_)+'.txt'))
        count_ +=1




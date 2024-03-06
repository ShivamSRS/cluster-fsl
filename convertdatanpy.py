from PIL import Image
import os
import numpy as np
import pandas as pd

expt_name = "NCT_newsplits"
path_to_files = "/data/srs/{}/".format(expt_name)  
array_of_images = []
# random_state = 0

train = pd.read_csv(path_to_files+'train_7cls.csv')
# val = pd.read_csv(path_to_files+'val.csv')
# test = pd.read_csv(path_to_files+'test.csv')
print(train['label'].value_counts())#,val['label'].value_counts(),test['label'].value_counts())



# print(train['label'].value_counts())
# print(train[train['label']=='ig'])
val_new = pd.DataFrame(columns=['filename','label'])
for i in train['label'].unique():
    val_1  = train[train['label']==i].take(np.random.permutation(len(train[train['label']==i]))[:int(0.2*len(train[train['label']==i]))])
    val_new = pd.concat([val_new,val_1])

# val_1 = train[train['label']=='ig'].take(np.random.permutation(len(train[train['label']=='ig']))[:int(0.2*len(train[train['label']=='ig']))])
# val_2 = train[train['label']=='eosinophil'].take(np.random.permutation(len(train[train['label']=='eosinophil']))[:int(0.2*len(train[train['label']=='eosinophil']))])
# val_new = pd.concat([val_1,val_2],axis=0)

full = set(range(0,len(train)))
indices_to_drop = set(val_new.index)
# print(indices_to_drop,len(indices_to_drop))
# exit()
indices_to_keep = full - indices_to_drop
# # full[indices_to_drop] = False
train_new = train.take(list(indices_to_keep))

print(len(train_new),len(val_new))

train_new = train_new.reset_index(drop=True)
val_new = val_new.reset_index(drop=True)

train_new.to_csv(path_to_files+'train_7cls.csv', encoding='utf-8', index=False)
val_new.to_csv(path_to_files+'val_7cls.csv', encoding='utf-8', index=False)

# print(val_new,set(val_new.index))
# print(train_new['label'].value_counts(),val_new['label'].value_counts())
# print(train['label'].value_counts(),val['label'].value_counts())
# print("##########")
# print(test['label'].value_counts())
exit()

"""

nv      150
mel     150
bkl     100
bcc     100
vasc    100
df      100

"""

"""

erythroblast    1200
lymphocyte      1200
basophil        1200
platelet        1200
monocyte        1200
neutrophil      1200

ig               800
eosinophil       800



"""
all_classes_nct = {'TUM':0,'NORM':1,'BACK':2,'MUS':3, 'MUC':4 ,'ADI':5 ,'STR':6,'DEB':7 , 'LYM':8}
all_classes_skin = {'nv':0,'mel':1,'bkl':2,'bcc':3, 'vasc':4 ,'df':5 }
all_classes_blood = {'ig':0,'eosinophil':1,'neutrophil':2,'monocyte':3,'platelet':4,'basophil':5,'lymphocyte':6,'erythroblast':7}

if expt_name=='Skin':
    all_classes = all_classes_skin
elif expt_name=='Blood':
    all_classes = all_classes_blood
else:
    all_classes=all_classes_nct

print(test['label'].unique(),train['label'].unique(),val['label'].unique(),sep="\n")

train_dict = {'images':[],'class_dict':[]}
val_dict = {'images':[],'class_dict':[]}
test_dict = {'images':[],'class_dict':[]}

print(train['label'].unique)
import pylab
for idx,row in enumerate(train.itertuples()):
    train_dict['images'].append(np.array(Image.open(path_to_files+"images/"+row.filename).convert('RGB')))
    train_dict['class_dict'].append(all_classes[row.label])

for idx,row in enumerate(val.itertuples()):
    val_dict['images'].append(np.array(Image.open(path_to_files+"images/"+row.filename).convert('RGB')))
    val_dict['class_dict'].append(all_classes[row.label])

for idx,row in enumerate(test.itertuples()):
    test_dict['images'].append(np.array(Image.open(path_to_files+"images/"+row.filename).convert('RGB')))
    test_dict['class_dict'].append(all_classes[row.label])

test_dict['class_dict'] = np.array(test_dict['class_dict'])
val_dict['class_dict'] = np.array(val_dict['class_dict'])
train_dict['class_dict'] = np.array(train_dict['class_dict']) 

test_dict['images'] = np.array(test_dict['images'])
val_dict['images'] = np.array(val_dict['images'])
train_dict['images'] = np.array(train_dict['images']) 

print(train_dict['class_dict'],val_dict['class_dict'],test_dict['class_dict'])
print(set(test_dict['class_dict']))
# exit()
import pickle


# Store data (serialize)
with open(path_to_files+'{}_train.pickle'.format(expt_name), 'wb') as handle:
    pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store data (serialize)
with open(path_to_files+'{}_val.pickle'.format(expt_name), 'wb') as handle:
    pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store data (serialize)
with open(path_to_files+'{}_test.pickle'.format(expt_name), 'wb') as handle:
    pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# exit()

# for _, file in enumerate(os.listdir(path_to_files+"images/")):
#     if "direction.jpg" in file: # to check if file has a certain name   
#         single_im = Image.open(file)
#         single_array = np.array(im)
#         array_of_images.append(single_array)            
# np.savez("all_images.npz",array_of_images) # save all in one file
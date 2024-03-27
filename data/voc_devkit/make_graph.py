from xml.dom.minidom import parse
import numpy as np
import xml.dom.minidom
import os
category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

img_dir = './VOC2007/JPEGImages'
anno_path = './VOC2007/ImageSets/Main/trainval.txt' 
labels_path = './VOC2007/Annotations'

labels = []  
train_graph = np.zeros((20,20)) 
with open(anno_path) as f:
    img_names=f.readlines()
for name in img_names:
    label_file = os.path.join(labels_path,name[:-1]+'.xml')
    label_vector = np.zeros(20)
    DOMTree = xml.dom.minidom.parse(label_file)
    root = DOMTree.documentElement
    objects = root.getElementsByTagName('object')
    l = len(objects)
    for i in range(0,l):
        if str(objects[i].getElementsByTagName('difficult')[0].firstChild.data) == '1':
            continue
        fromid = category_info[objects[i].getElementsByTagName('name')[0].firstChild.data.lower()]
        label_vector[int(fromid)]+=1.0
        for j in range(i+1, l):
            if str(objects[j].getElementsByTagName('difficult')[0].firstChild.data)=='1':
                continue
            toid = category_info[objects[j].getElementsByTagName('name')[0].firstChild.data.lower()]
            if not (fromid == toid):
            	train_graph[fromid][toid] += 1
           	train_graph[toid][fromid] += 1   
    labels.append(label_vector)
labels = np.array(labels).astype(np.float32)
print(labels.shape)
exists = np.sum(labels, axis=0)
print(exists)
prob_train = (train_graph.T / exists).T
np.save('prob_trainval.npy',prob_train) 

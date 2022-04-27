##Simple script for generating train.txt, test.txt and valid.txt

import os
import random

data_T1 = []
data_T2 = []
data_T3 = []

T1_path = './T1'
T2_path = './T2'
T3_path = './T3'

def write_txt_file(img_list,filename):
  textfile = open(filename, "w")
  for element in img_list:
      textfile.write(str(element[0]) + "," +str(element[1]) + "\n")
  textfile.close()

def generatefile_list(data, path):
  for root, dir, filenames in os.walk(path):
    for filename in filenames:
      if filename.split(".")[1] == 'png' and filename.split('_')[0] == 'Depth':
        prefix, postfix = filename.split('_')
        xFilename = "FrameBuffer_" + postfix
        x_y = (os.path.join(root, xFilename), os.path.join(root, filename))
        data.append(x_y)

generatefile_list(data_T1, T1_path)
generatefile_list(data_T2, T1_path)
generatefile_list(data_T3, T1_path)

total_len = len(data_T1) + len(data_T3) + len(data_T2)

random.shuffle(data_T1)
random.shuffle(data_T2)
random.shuffle(data_T3)

unit_len = len(data_T1)
train_split = 0.8
valid_split = 0.1
train_index = int(unit_len * train_split) 
valid_index = int((train_split + valid_split) * unit_len)

train_list = data_T1[:train_index] + data_T2[:train_index] + data_T3[:train_index]
test_list = data_T1[valid_index:] + data_T2[valid_index:] + data_T3[valid_index:]
valid_list = data_T1[train_index:valid_index] + data_T2[train_index:valid_index] + data_T3[train_index:valid_index]


assert total_len == len(train_list) + len(test_list) + len(valid_list)

write_txt_file(train_list, 'train.txt')
write_txt_file(test_list, 'test.txt')
write_txt_file(valid_list, 'valid.txt')


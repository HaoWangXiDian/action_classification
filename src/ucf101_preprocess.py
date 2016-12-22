'''
Created on Dec 22, 2016

@author: Behrooz
'''
# this lines add the sequence classification submodule to the python path
import sys
sys.path.append('../sequence_analysis/src')

#############################

#############################
import os
import tensorflow as tf
import numpy as np

train_files = ['../ucf_train_test_files/trainlist01.txt', '../ucf_train_test_files/trainlist02.txt', '../ucf_train_test_files/trainlist03.txt']
test_files = ['../ucf_train_test_files/testlist01.txt', '../ucf_train_test_files/testlist02.txt', '../ucf_train_test_files/testlist03.txt']
parent_folder = '../UCF-101-Features/'
sequence_size = 32
def modify_test_files():
    class_map = {}
    class_ids = open('../ucf_train_test_files/classInd.txt')
    for line in class_ids:
        s = line.split(' ')
        class_map[s[1].strip()] = int(s[0])
    for file_name in test_files:
        file_content = []
        file = open(file_name, 'r')
        for line in file:
            label = class_map[line.split('/')[0]]
            file_content.append(line.strip() + ' ' + str(label) + '\n')
        file = open(file_name + 'm', 'w')
        file.writelines(file_content)
    
def generate_train():
    for file_name in train_files:
        file = open(file_name, 'r')
        for line in file:
            s = line.split(' ')
            label = int(s[1].strip())
            feature_file = parent_folder + s[0].split('/')[1][:-4] + '.npy'
            features = np.load(feature_file)
            step = features.shape[0] / sequence_size
            subsampled_features = features[0:features.shape[0]:step]
            print(label, subsampled_features)
            if (subsampled_features.shape[0] < sequence_size):
                print 'Error!!!!!!!'

def generate_test():
    for file_name in train_files:
        file = open(file_name, 'r')
        for line in file:
            s = line.split(' ')
            label = int(s[1].strip())
            feature_file = parent_folder + s[0].split('/')[1][:-4] + '.npy'
            features = np.load(feature_file)
            print(label, features.shape)
def main():
#     modify_test_files()
    generate_train()
if __name__ == "__main__":
#   flags.FLAGS(sys.argv)
    main()

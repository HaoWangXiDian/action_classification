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
sequence_size = 32.0
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
    fIndex = 1;
    for file_name in train_files:
        file = open(file_name, 'r')
        for line in file:
            s = line.split(' ')
            label = int(s[1].strip()) - 1
            feature_file = parent_folder + s[0].split('/')[1][:-4] + '.npy'
            features = np.load(feature_file)
            step = features.shape[0] / sequence_size
            subsampled_features = []
            j = 0    
            while j * step < features.shape[0]:
                subsampled_features.append(np.append(features[int(round(j * step))], [label]))
                j += 1		
            if len(subsampled_features) < sequence_size:
                subsampled_features.append(np.append(features[-1], [label]))
            subsampled_features = np.asanyarray(subsampled_features)
            if (subsampled_features.shape[0] < sequence_size):
                print 'Error!!!!!!!'
                print(line, subsampled_features.shape)
            subsampled_features.tofile(parent_folder + 'ucf_train' + str(fIndex) + '.bin')
            fIndex += 1
def generate_test():
    fIndex = 1
    for file_name in test_files:
        file = open(file_name, 'r')
        for line in file:
            s = line.split(' ')
            label = int(s[1].strip()) - 1
            feature_file = parent_folder + s[0].split('/')[1][:-4] + '.npy'
            features = np.load(feature_file)
            step = features.shape[0] / sequence_size
            subsampled_features = []
            j = 0    
            while j * step < features.shape[0]:
                subsampled_features.append(np.append(features[int(round(j * step))], [label]))
                j += 1        
            if len(subsampled_features) < sequence_size:
                subsampled_features.append(np.append(features[-1], [label]))
            subsampled_features = np.asanyarray(subsampled_features)
            if (subsampled_features.shape[0] < sequence_size):
                print 'Error!!!!!!!'
                print(line, subsampled_features.shape)
            subsampled_features.tofile(parent_folder + 'ucf_test' + str(fIndex) + '.bin')
            fIndex += 1
def main():
#     modify_test_files()
    generate_train()
    generate_test()
if __name__ == "__main__":
#   flags.FLAGS(sys.argv)
    main()

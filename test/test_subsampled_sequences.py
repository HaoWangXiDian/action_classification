'''
Created on Dec 22, 2016

@author: behrooz
'''
import sys
sys.path.append('../sequence_analysis/src')

import os
import tensorflow as tf
from sequence_reader import Sequence_Reader
import time
import re

def test_reader(FLAGS):
    
    sequence_reader = Sequence_Reader(FLAGS)
    
    reader_op = sequence_reader.get_next_training_batch(FLAGS)
    
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()    
    
    
    # Build an initialization operation to run below.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    index = 0
    while True:
        print('==========================')
        print ('index:' + str(index))
        x, y = sess.run(reader_op)
        print y
        index += 1
	break
#         print (x, y)


flags = tf.app.flags

# model parameters
flags.DEFINE_string("input_format", tf.float64, "input format tf.float32 or tf.float64")
flags.DEFINE_string("data_dir", ".", "The data Directory")
flags.DEFINE_string("train_file_range", [1, 2], "number of training files")
flags.DEFINE_string("train_file_name_prefix", 'ucf_train_', "number of training files")
flags.DEFINE_string("train_file_name_postfix", '.bin', "number of training files")
flags.DEFINE_string("test_file_range", [1, 2], "number of training files")
flags.DEFINE_string("test_file_name_prefix", 'ucf_train_', "number of training files")
flags.DEFINE_string("test_file_name_postfix", '.bin', "number of training files")
flags.DEFINE_integer("input_size", 2048, "Input dimensionality [4096]")
flags.DEFINE_integer("sequence_size", 32, "Sequence_size")

flags.DEFINE_integer("number_layers", 2, "Number of LSTM units")
flags.DEFINE_integer("number_hidden", 128, "Number of hidden units")

flags.DEFINE_integer("output_size", 101, "Output dimensionality [2]")



# hyper-parameters
flags.DEFINE_integer("number_epochs", 20, "Number of training epochs [20]")
flags.DEFINE_integer("batch_size", 1 , "Batch_size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for Adam [0.001]")
flags.DEFINE_float("weight_decay", 0.0005, "Weight decay [0.0005]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam optimizer [0.5]")
flags.DEFINE_float("MOVING_AVERAGE_DECAY", 0.9999, "MOVING_AVERAGE_DECAY")
flags.DEFINE_float("num_gpus", 1 , "MOVING_AVERAGE_DECAY")

flags.DEFINE_float("dropout", True, "add drop out layer? [default:True]")
flags.DEFINE_float("keep_prob", 0.5, "drop-out keep prob  [default:0.5]")
flags.DEFINE_float("stddev", 0.04, "stddev for variable initialization")

#
flags.DEFINE_boolean("train_mode", True, "Should set to False in case of evaluation")
flags.DEFINE_string("NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN", 70, "number of training files")
flags.DEFINE_string("log_device_placement", False, "log_device_placement")
flags.DEFINE_string("shuffle", False, "number of training files")
flags.DEFINE_string("max_steps", 100000, "number of maximum iterations")

flags.DEFINE_string("model_path", "./models/", "Model path [./models/]")
flags.DEFINE_string("model_prefix", "deep3PEAT", "Model prefix [ucf101_]")
flags.DEFINE_string("recurrent_unit", "LSTM", "Recurrent cell type [LSTM]")


# env CUDA_VISIBLE_DEVICES=1 python main.py --train_mode=False --test_samples=./train_files.txt
# env CUDA_VISIBLE_DEVICES=1 python main.py --train_mode=True --test_samples=./train_files.txt
def main():
    FLAGS = flags.FLAGS
    # initializing dataset
    test_reader(FLAGS)
if __name__ == "__main__":
#   flags.FLAGS(sys.argv)
    main()

'''
Created on Dec 22, 2016

@author: Behrooz
'''
# this lines add the sequence classification submodule to the python path
import sys
sys.path.append('../sequence_analysis/src')

#############################
import os
import tensorflow as tf
from sequence_reader import Sequence_Reader
import sequence_eval

############################

flags = tf.app.flags

# input format
flags.DEFINE_string("input_format", tf.float64, "input format tf.float32 or tf.float64")
flags.DEFINE_string("data_dir", "../UCF-101-Features/", "The data Directory")
flags.DEFINE_string("train_file_range", [1, 28747], "range of training files")
flags.DEFINE_string("train_file_name_prefix", "ucf_train", "Training Input files prefix")
flags.DEFINE_string("train_file_name_postfix", ".bin", "Training  Input files postfix")
flags.DEFINE_string("test_file_range", [1, 11213], "range of test files")
flags.DEFINE_string("test_file_name_prefix", "ucf_test", "Test Input files prefix")
flags.DEFINE_string("test_file_name_postfix", ".bin", "Test Input files postfix")

flags.DEFINE_integer("test_iterations_in_one_train_iteration", 100, "test_iterations_in_one_train_iteration")

# model parameters
flags.DEFINE_integer("input_size", 2048, "Input dimensionality [212]")
flags.DEFINE_integer("sequence_size", 32, "Sequence_size")

flags.DEFINE_integer("number_layers", 2, "Number of LSTM units")
flags.DEFINE_integer("number_hidden", 512, "Number of hidden units")

flags.DEFINE_integer("output_size", 101, "Output dimensionality [2]")
flags.DEFINE_integer("sequence_classifer_type", 1, "Sequence Classifier Type 1: original 2: Applies CNN")
flags.DEFINE_integer("feature_map_size", 32, "Size of the Feature Map if 1-D Convolution is used")





# hyper-parameters
flags.DEFINE_integer("number_epochs", 20, "Number of training epochs [20]")
flags.DEFINE_integer("batch_size", 5 , "Batch_size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for Adam [0.001]")
flags.DEFINE_float("weight_decay", 0.0005, "Weight decay [0.0005]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam optimizer [0.5]")
flags.DEFINE_float("MOVING_AVERAGE_DECAY", 0.9999, "MOVING_AVERAGE_DECAY")
flags.DEFINE_integer("num_gpus", 4 , "MOVING_AVERAGE_DECAY")
flags.DEFINE_float("per_process_gpu_memory_fraction", 0.4 , "per_process_gpu_memory_fraction")

flags.DEFINE_boolean("dropout", False, "add drop out layer? [default:True]")
flags.DEFINE_float("keep_prob", 0.5, "drop-out keep prob  [default:0.5]")
flags.DEFINE_float("stddev", 0.04, "stddev for variable initialization")

#
flags.DEFINE_string("train_dir", "./log/train/", "training directory")

flags.DEFINE_boolean("train_mode", True, "Should set to False in case of evaluation")
flags.DEFINE_integer("NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN", 100, "number of training files")
flags.DEFINE_boolean("log_device_placement", False, "log_device_placement")
flags.DEFINE_boolean("shuffle", False, "number of training files")
flags.DEFINE_integer("max_steps", 250000, "number of maximum iterations")

flags.DEFINE_string("model_path", "./models/", "Model path [./models/]")
flags.DEFINE_string("model_prefix", "UCF101", "Model prefix [ucf101_]")
flags.DEFINE_string("recurrent_unit", "LSTM", "Recurrent cell type [LSTM]")


#Test time parameters
flags.DEFINE_integer("number_of_test_examples", 1000, "number_of_test_examples")

# env CUDA_VISIBLE_DEVICES=1 python main.py --train_mode=False --test_samples=./train_files.txt
# env CUDA_VISIBLE_DEVICES=1 python main.py --train_mode=True --test_samples=./train_files.txt
def main():
    FLAGS = flags.FLAGS
    sequence_eval.eval(FLAGS)
if __name__ == "__main__":
#   flags.FLAGS(sys.argv)
    main()


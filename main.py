"""
This module is for creating a web application using Flask to transcribe music scores to ABC notation 
to facilitate the process of learning piano
"""
from flask import Flask, send_from_directory, render_template, request
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import tensorflow as tf

# Create app instance
app = Flask(__name__)

# Define paths
vocab_file = "vocabulary_semantic.txt"
model = "semantic/semantic_model.meta"




def setup():
    """
    Function to setup the trained OMR model

    Returns
    -------
    None.

    """
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()
    
    # Read in dictionary of vocabulary text
    vocab_dict = open(vocab_file, 'r')
    vocab_list = vocab_dict.read().splitlines()
    index_word = dict()
    # Map index to word 
    for idx, word in enumerate(vocab_list):
        index_word[idx] = word
    #print(f'index word: {index_word}')
    vocab_dict.close()
    
    # Restore weights
    saver = tf.compat.v1.train.import_meta_graph(model)
    saver.restore(sess, model[:-5])
    
    # Create graph
    graph = tf.compat.v1.get_default_graph()
    
    # Retrieve tensors frm model
    input_tensor = graph.get_tensor_by_name("model_input:0")
    #print(f"input tensor:{input_tensor}")
    len_seq = graph.get_tensor_by_name("seq_lengths:0")
    #print(f"len_seq: {len_seq}")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    #print(rnn_keep_prob)
    height_tensor = graph.get_tensor_by_name("input_height:0")
    #print(height_tensor)
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    #print(width_reduction_tensor)
    logits = tf.compat.v1.get_collection("logits")[0]
    #print(logits)
    
    # Retrieve constants from the model
    width_reduction, height = sess.run([width_reduction_tensor, height_tensor])
    
    decoded, _ = tf.nn.ctc_greedy_decoder(logits, len_seq)
    
    

# Run web application
if __name__ == "__main__":
    setup()
    
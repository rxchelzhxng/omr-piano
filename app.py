"""
This module is for creating a web application using Flask to transcribe music scores to ABC notation 
to facilitate the process of learning piano
"""
from flask import Flask, send_from_directory, render_template, request
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np
import tensorflow as tf

# Create app instance
app = Flask(__name__, static_url_path='')

# Render HTML template files
@app.route('/')
def index():
    return render_template('index.html', final_result=False)

# Make predictions on sent image
@app.route('/make_predictions', methods = ['GET', 'POST'])
def make_predictions():
    """
    Function to make predictions on the user input image with the trained model tensors. 
    Annotates music scores onto the input image.
    """
    if request.method == "POST":
        # Load in image as array
        file = request.files['file']
        image = Image.open(file).convert('L')
        imarr = np.array(image)
        
        # Resize image according to model height
        imarr = resize_image(imarr, height_const)
        # Normalize array
        imarr = normalize_image(imarr)
        imarr = np.asarray(imarr).reshape(1, imarr.shape[0], imarr.shape[1], 1)
        
        len_seqs = [imarr.shape[2] / width_const]
        # Make the prediction
        pred_results = sess.run(decoded, feed_dict={input_tensor: imarr,
                                                len_seq: len_seqs,
                                                rnn_keep_prob: 1.0})
        
        # Extract notes from vocab file with index list
        str_predictions = sparse_tensor_to_string(pred_results)
        notes_arr = []
        for idx in str_predictions[0]:
            notes_arr.append(index_word[idx])
        # Extract monophonic notes
        notes = []
        for note in notes_arr:
            # Check if score is a note
            if note[0:5] == "note-":
                # Append only the ABC notation note
                if not note[6].isdigit():
                    notes.append(note[5:7])
                else:
                    notes.append(note[5])
        
        # Extract image width and height
        image_size = (image.size[0], int(image.size[1]*1.5))
        # Create new image
        new_image = Image.new("RGB", image_size, (255, 255, 255))
        new_image.paste(image, box=None)
        new_imarr = np.array(new_image)
        height = int(new_imarr.shape[0])
        width = int(new_imarr.shape[1])
        
        # Annotate scores onto image
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype("ABeeZee-Regular.otf", 22)
        w = width / 9
        for n in notes:
            draw.text((w, height-40), n, fill=(0, 0, 0), font=font)
            w += (width / (len(notes) + 2.2))
        new_image.save("static/annotated.jpg")
     
        return render_template('index.html', final_result=True)
    


def sparse_tensor_to_string(sparse_tensor):
    """
    Function that takes in the sparse tensor from prediction and extracts the values
    Parameters
    ----------
    sparse_tensor : SparseTensorValue
    Sparse Tensor value from prediction

    Returns
    -------
    strs: list of lists containing values from prediction

    """
    indices= sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [ [] for i in range(dense_shape[0]) ]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])

        ptr = ptr + 1
    
    strs[b] = string
    
    return strs
    

def resize_image(image, height):
    """
    Function to resize image given the image array and height 

    Parameters
    ----------
    image : array
    Numpy array of PIL image
    height : integer
    Image height

    Returns
    -------
    resize_arr: Resized image array

    """
    # Calculate width
    width = int(float((height * image.shape[1]) / image.shape[0]))
    resize_arr = cv2.resize(image, (width, height))
    return resize_arr


def normalize_image(image):
    """
    Function to normalize image array to range from 0 to 1

    Parameters
    ----------
    image : array
    Numpy array of PIL image

    Returns
    -------
    normalize_arr: Normalized image array

    """
    normalize_arr = (255 - image) / 255 
    return normalize_arr

# Define model and vocab paths
vocab_file = "vocabulary_semantic.txt"
model = "semantic/semantic_model.meta"
  
# Setup the trained OMR model
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
    
# Read in dictionary of vocabulary text
vocab_dict = open(vocab_file, 'r')
vocab_list = vocab_dict.read().splitlines()
index_word = dict()
# Map index to word 
for idx, word in enumerate(vocab_list):
    index_word[idx] = word
   
vocab_dict.close()

# Restore weights
saver = tf.compat.v1.train.import_meta_graph(model)
saver.restore(sess, model[:-5])

# Create graph
graph = tf.compat.v1.get_default_graph()

# Retrieve tensors from model
input_tensor = graph.get_tensor_by_name("model_input:0")
len_seq = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.compat.v1.get_collection("logits")[0]

# Retrieve constants from the model
width_const, height_const = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, len_seq)


# Run web application
if __name__ == "__main__":
     app.run()
    
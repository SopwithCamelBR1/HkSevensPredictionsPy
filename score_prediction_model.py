import array as arr
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from random import randint
import re
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras as ks

debug = 1

#data_filename = "Data/world_rugby_sevens_series_data_manual.csv"
data_filename = "Data/world_rugby_sevens_score_data_manual.csv"

''' Load data
* ARGS:
'''
def load_data(filename):

  #data lists
  teamA_list = []
  teamB_list = []
  teamA_score_list = []
  teamB_score_list = []
  match_results = []

  with open(filename, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      teamA_list.append(row[0])
      teamB_list.append(row[1])
      stra = row[2]
      stra2 = re.sub('[^0-9]','', stra)
      teamA_score_list.append(stra2)
      strb = row[3]
      strb2 = re.sub('[^0-9]','', strb)
      teamB_score_list.append(strb2)
      
  teamA_list.pop(0)
  teamB_list.pop(0)
  teamA_score_list.pop(0)
  teamB_score_list.pop(0)
  
  for i in range(len(teamA_score_list)):
    if int(teamA_score_list[i]) > int(teamB_score_list[i]):
      match_results.append(1)
    else:
        match_results.append(0)
 
  #print lists:
  if debug >= 2:
    print(teamA_list)
    print(teamB_list)
    print(teamA_score_list)
    print(teamB_score_list)
    print(match_results)
    
  return teamA_list, teamB_list, teamA_score_list, teamB_score_list, match_results

''' Onehot encoding  
* Encode an array using onehot encoding 
* ARGS:
''' 
def onehot_array(array):
  enc = OneHotEncoder(handle_unknown='ignore')
  enc.fit(array)
  onehot_array = enc.transform(array).toarray()
  
  if debug >= 2:
    print("Onehot array features:")
    print("Features encoded: ", enc.get_feature_names())
    print("One hot encoded array: ", onehot_array)
    print("type: ", type(onehot_array))
    print("shape: ", onehot_array.shape)
    print("Indivial onehot features:")
    print("Indivual onehot example: ", str(onehot_array[0]))
    print("type item: " + str(type(onehot_array[0])))
    print("shape: ",onehot_array[0].shape)
    print("")
    
  return onehot_array, enc

#move item from one array to another
def split_array(array_orig, array_new, index):
  
  item_to_add = np.array([array_orig[index]])

  if debug >= 2:
    print("Index: ", index)
    print("Original Array pre-split: ", array_orig)
    print("New Array pre-split: ",array_new)
    print("Item to add pre-split: ",item_to_add)  
    print("Original Array shape pre-split: ",array_orig.shape)
    print("New Array shape pre-split: ",array_new.shape)
    print("Item to add shape pre-split: ",item_to_add.shape)
    
  array_new = np.append(array_new, item_to_add, axis=0)
  array_orig = np.delete(array_orig, index, 0)
  
  #print lists:
  if debug >= 2:
    print("Index: ", index)
    print("Original Array pre-split: ", array_orig)
    print("New Array post-split: ",array_new) 
    print("Original Array shape post-split: ",array_orig.shape)
    print("New Array shape post-split: ",array_new.shape)
  
  return array_orig, array_new
  
''' Dataset Creation
* Splits data into training and validate
* ARGS: 
'''   
def create_dataset(array1, array2, array3, array4, array5, percent_validate):
  if percent_validate >= 1.0 or percent_validate < 0.0:
    print("perecnt must be between 0 and 1")
  else:
    #orig_number = len(array1)
    orig_number1, dimension1 = array1.shape
    orig_number2, dimension2 = array2.shape
    orig_number3 = array3.shape
    orig_number4 = array4.shape
    orig_number5 = array5.shape
    val_number = int(orig_number1*percent_validate)
    train_number = orig_number1-val_number
    
    #debug
    if debug >= 2:
      print("orig_number: " + str(orig_number1))
      #print("dimension: " + str(dimension))      
      print("val_number: " + str(val_number))
      print("train_number: " + str(train_number)) 
    
    train1_array = array1
    train2_array = array2
    train3_array = array3
    train4_array = array4
    train5_array = array5  
    validate1_array = np.empty(shape=[0, dimension1])
    validate2_array = np.empty(shape=[0, dimension2])
    validate3_array = np.empty(shape=[0])
    validate4_array = np.empty(shape=[0])
    validate5_array = np.empty(shape=[0])
    
    num = orig_number1
    for i in range(val_number):
      index_to_swap=randint(0, num-1)
      train1_array, validate1_array = split_array(train1_array, validate1_array, index_to_swap)
      train2_array, validate2_array = split_array(train2_array, validate2_array, index_to_swap)
      train3_array, validate3_array = split_array(train3_array, validate3_array, index_to_swap)
      train4_array, validate4_array = split_array(train4_array, validate4_array, index_to_swap)
      train5_array, validate5_array = split_array(train5_array, validate5_array, index_to_swap)      
      num -= 1
      
  #debug
  if debug >= 2:
    print("train1_array: ", str(train1_array))
    print("train1_array chape: ", train1_array.shape)
    print("train2_array: ", str(train2_array))
    print("train2_array chape: ", train2_array.shape)
    print("train3_array: ", str(train3_array))
    print("train3_array chape: ", train3_array.shape)
    print("train4_array: ", str(train4_array))
    print("train4_array chape: ", train4_array.shape)
    print("train5_array: ", str(train5_array))
    print("train5_array chape: ", train5_array.shape)
    
    print("validate1_array: ", str(validate1_array))
    print("validate1_array shape: ", validate1_array.shape)
    print("validate2_array: ", str(validate2_array))
    print("validate2_array shape: ", validate2_array.shape)
    print("validate3_array: ", str(validate3_array))
    print("validate3_array shape: ", validate3_array.shape)
    print("validate4_array: ", str(validate4_array))
    print("validate4_array shape: ", validate4_array.shape)
    print("validate5_array: ", str(validate5_array))
    print("validate5_array shape: ", validate5_array.shape)
      
  return train1_array,  train2_array, train3_array, train4_array, train5_array, validate1_array, validate2_array, validate3_array, validate4_array, validate5_array

#activation function
def activation(input):
  output = tf.sigmoid(input)
  #debug  
  if debug >= 2:
    print("Activation input: ", input)
    print("Activation output: ", output)
  
  return output
      
''' Score Model Creation
* ARGS: 
'''  
def create_score_model(A_depth, B_depth, hidden_nodes, out_nodes):
  #arguments debug
  if debug >= 2:
    print("Model Dimensions")
    print("A_depth: ", A_depth)
    print("A_depth type: ", type(A_depth))
    print("B_depth: ", B_depth)
    print("B_depth type: ", type(B_depth))
    print("hidden_nodes: ", hidden_nodes)
    print("hidden_nodes type: ", type(hidden_nodes))
    print("out_nodes: ", B_depth)
    print("out_nodes type: ", type(out_nodes))
    print("")
    
  #Inputs
  tf_teamA_input = tf.placeholder(tf.float32, shape=[1, A_depth])
  tf_teamB_input = tf.placeholder(tf.float32, shape=[1, B_depth])
  #debug
  if debug >= 2:
    print("Node Inputs:")
    print("Team A tf input: ", tf_teamA_input)
    print("Team A tf input shape: ", tf.shape(tf_teamA_input))
    print("Team A tf input type: ", type(tf_teamA_input))
    print("Team B tf input: ", tf_teamB_input)
    print("Team B tf input shape: ", tf.shape(tf_teamB_input))
    print("Team B tf input type: ", type(tf_teamB_input))
    print("")    
  
  #Hidden Weights & Bias
  teamA_hidden_weights = tf.Variable(tf.random.truncated_normal([A_depth, hidden_nodes], stddev=1.0))
  teamB_hidden_weights = tf.Variable(tf.random.truncated_normal([B_depth, hidden_nodes], stddev=1.0))
  #hidden_bias = tf.Variable(tf.zeros([hidden_nodes]))
  #debug
  if debug >= 2:
    print("Hidden Weights:")
    print("Team A weights: ", teamA_hidden_weights)
    print("Team A weights shape: ", tf.shape(teamA_hidden_weights))
    print("Team A weights type: ", type(teamA_hidden_weights))
    print("Team B weights: ", teamB_hidden_weights)
    print("Team B weights shape: ", tf.shape(teamB_hidden_weights))
    print("Team B weights type: ", type(teamB_hidden_weights))
    print("")

  #Input X Hidden Weights
  A_inXw = tf.matmul(tf_teamA_input, teamA_hidden_weights) 
  B_inXw = tf.matmul(tf_teamB_input, teamB_hidden_weights)
  #debug
  if debug >= 2:
    print("Input x Hidden Weights:")
    print("A input X hidden weights: ", A_inXw)
    print("A input X hidden weights shape: ", tf.shape(A_inXw))
    print("A input X hidden weights type: ", type(A_inXw))
    print("B input X hidden weights: ", B_inXw)
    print("B input X hidden weights shape: ", tf.shape(B_inXw))
    print("B input X hidden weights type: ", type(B_inXw))
    print("")
  
  #Sumt Total Input
  total_input = A_inXw + B_inXw# + hidden_bias
  #debug
  if debug >= 2:
    print("Total Input:")
    print("Total Input: ", total_input)
    print("Total Input shape: ", tf.shape(total_input))
    print("Total Input type: ", type(total_input))
    print("")
  
  #Hidden layer Activation
  hidden_out = activation(total_input)
  #debug
  if debug >= 2:
    print("Hidden Output:")
    print("Hidden Output: ", hidden_out)
    print("Hidden Output type: ", type(hidden_out))
    print("")
    
  #Linear Wieghts & Bias
  linear_weights = tf.Variable(tf.random.truncated_normal([hidden_nodes, out_nodes], stddev=1.0))
  #linear_bias = tf.Variable(tf.zeros([hidden_nodes]))
  #debug
  if debug >= 2:
    print("Linear Weights:")
    print("Linear weights: ", linear_weights)
    print("Linear weights type: ", type(linear_weights))

  #Linear Output
  linear_output = tf.matmul(hidden_out, linear_weights)# + linear_bias 
  #debug
  if debug >= 2:
    print("Linear output:")
    print("A input X hidden weights: ", linear_output)
    print("A input X hidden weights type: ", type(linear_output))
    print("")
  
  #Labels
  train_labels = tf.placeholder(tf.int32, shape=[1,out_nodes])
  if debug >= 2:
    print("Labels/Target:")
    print("A input X hidden weights: ", train_labels)
    print("A input X hidden weights type: ", type(train_labels))
    print("")
  
  return linear_output, hidden_out, tf_teamA_input, tf_teamB_input, teamA_hidden_weights, teamB_hidden_weights, linear_weights, train_labels#, hidden_bias

'''Match Result Model Creation
*
'''
def create_match_result_model(A_depth, B_depth, hidden_nodes, out_nodes):
  #arguments debug
  if debug >= 2:
    print("Model Dimensions")
    print("A_depth: ", A_depth)
    print("A_depth type: ", type(A_depth))
    print("B_depth: ", B_depth)
    print("B_depth type: ", type(B_depth))
    print("hidden_nodes: ", hidden_nodes)
    print("hidden_nodes type: ", type(hidden_nodes))
    print("out_nodes: ", B_depth)
    print("out_nodes type: ", type(out_nodes))
    print("")
    
  #Inputs
  tf_teamA_input = tf.placeholder(tf.float32, shape=[1, A_depth])
  tf_teamB_input = tf.placeholder(tf.float32, shape=[1, B_depth])
  #debug
  if debug >= 2:
    print("Node Inputs:")
    print("Team A tf input: ", tf_teamA_input)
    print("Team A tf input shape: ", tf.shape(tf_teamA_input))
    print("Team A tf input type: ", type(tf_teamA_input))
    print("Team B tf input: ", tf_teamB_input)
    print("Team B tf input shape: ", tf.shape(tf_teamB_input))
    print("Team B tf input type: ", type(tf_teamB_input))
    print("")    
  
  #Hidden Weights & Bias
  teamA_hidden_weights = tf.Variable(tf.random.truncated_normal([A_depth, hidden_nodes], stddev=1.0))
  teamB_hidden_weights = tf.Variable(tf.random.truncated_normal([B_depth, hidden_nodes], stddev=1.0))
  #hidden_bias = tf.Variable(tf.zeros([hidden_nodes]))
  #debug
  if debug >= 2:
    print("Hidden Weights:")
    print("Team A weights: ", teamA_hidden_weights)
    print("Team A weights shape: ", tf.shape(teamA_hidden_weights))
    print("Team A weights type: ", type(teamA_hidden_weights))
    print("Team B weights: ", teamB_hidden_weights)
    print("Team B weights shape: ", tf.shape(teamB_hidden_weights))
    print("Team B weights type: ", type(teamB_hidden_weights))
    print("")

  #Input X Hidden Weights
  A_inXw = tf.matmul(tf_teamA_input, teamA_hidden_weights) 
  B_inXw = tf.matmul(tf_teamB_input, teamB_hidden_weights)
  #debug
  if debug >= 2:
    print("Input x Hidden Weights:")
    print("A input X hidden weights: ", A_inXw)
    print("A input X hidden weights shape: ", tf.shape(A_inXw))
    print("A input X hidden weights type: ", type(A_inXw))
    print("B input X hidden weights: ", B_inXw)
    print("B input X hidden weights shape: ", tf.shape(B_inXw))
    print("B input X hidden weights type: ", type(B_inXw))
    print("")
  
  #Sum Total Hidden Input
  hidden_input = A_inXw + B_inXw# + hidden_bias
  #debug
  if debug >= 2:
    print("Total Input:")
    print("Total Input: ", hidden_input)
    print("Total Input shape: ", tf.shape(hidden_input))
    print("Total Input type: ", type(hidden_input))
    print("")
  
  #Hidden layer Activation
  hidden_out = activation(hidden_input)
  #debug
  if debug >= 2:
    print("Hidden Output:")
    print("Hidden Output: ", hidden_out)
    print("Hidden Output type: ", type(hidden_out))
    print("")
    
  #Out Wieghts & Bias
  linear_weights = tf.Variable(tf.random.truncated_normal([hidden_nodes, out_nodes], stddev=1.0))
  #linear_bias = tf.Variable(tf.zeros([hidden_nodes]))
  #debug
  if debug >= 2:
    print("Linear Weights:")
    print("Linear weights: ", linear_weights)
    print("Linear weights type: ", type(linear_weights))

  #Final Input
  final_input = tf.matmul(hidden_out, linear_weights)# + linear_bias 
  #debug
  if debug >= 2:
    print("Linear output:")
    print("A input X hidden weights: ", final_input)
    print("A input X hidden weights type: ", type(final_input))
    print("")
    
  #Final layer Activation
  final_output = activation(final_input)
  #debug
  if debug >= 2:
    print("Hidden Output:")
    print("Hidden Output: ", final_output)
    print("Hidden Output type: ", type(final_output))
    print("")  
  
  #Labels
  train_labels = tf.placeholder(tf.int32, shape=[1, out_nodes])
  if debug >= 2:
    print("Labels/Target:")
    print("A input X hidden weights: ", train_labels)
    print("A input X hidden weights type: ", type(train_labels))
    print("")
  
  return final_output, hidden_out, tf_teamA_input, tf_teamB_input, teamA_hidden_weights, teamB_hidden_weights, linear_weights, train_labels#, hidden_bias

''' Predict Match Result
* ARGS: 
''' 
def predict_match_result(teamA, teamB, model, encA, encB):
  
  if debug >= 1:
    print("predict_match_results arguments:")
    print("teamA: ", teamA)
    print("teamA type: ", type(teamA))
    print("teamB: ", teamB)
    print("teamB type: ", type(teamB))
    print("encA: ", encA)
    print("encA features: ", encA.get_feature_names())
    print("encA params: ", encA.get_params())
    print("encB: ", encB)
    print("encB features: ", encB.get_feature_names())
    print("encB params: ", encB.get_params())
    print("")
  
  exampleA = [teamA]
  print("exampleA: ", exampleA)
  np_exampleA = np.array(exampleA).reshape(-1, 1)
  print("np_exampleA: ", np_exampleA, "\n")
  onehot_exampleA = encA.transform(np.array(exampleA).reshape(-1, 1)).toarray()
  exampleB = [teamB]
  onehot_exampleB = encB.transform(np.array(exampleB).reshape(-1, 1)).toarray()
  
  if debug >= 1:
    print("Examples:")
    print("One hot exampleA: ", onehot_exampleA)
    print("exampleA shape: ", onehot_exampleA.shape)
    print("One hot exampleB: ", onehot_exampleA)
    print("exampleB shape: ", onehot_exampleB.shape)
    print("")
    
  pred = session.run([model], feed_dict={r_tf_teamA_input:onehot_exampleA, r_tf_teamB_input:onehot_exampleB})
  pred_value = pred[0][0]
  
  if debug >= 2:
    print("Result Prediction:")
    print("Pred: ", pred)
    print("Pred type: ", type(pred))
    #print("Pred shape: ", pred.shape)
    print("Pred value: ", pred_value)
    print("")
  
  if pred_value < 0.5:
    winner = teamB
  else:
    winner = teamA
  
  if debug >= 2:
    print("Winner: ", winner)
    print("")
  
  return winner, pred 

''' Predict Score
* ARGS: 
''' 
def predict_score(teamA, teamB, model, encA, encB):
  
  if debug >= 1:
    print("print_score arguments:")
    print("teamA: ", teamA)
    print("teamA type: ", type(teamA))
    print("teamB: ", teamB)
    print("teamB type: ", type(teamB))
    print("encA: ", encA)
    print("encA features: ", encA.get_feature_names())
    print("encA params: ", encA.get_params())
    print("encB: ", encB)
    print("encB features: ", encB.get_feature_names())
    print("encB params: ", encB.get_params())
    print("")
  
  exampleA = [teamA]
  onehot_exampleA = encA.transform(np.array(exampleA).reshape(-1, 1)).toarray()
  exampleB = [teamB]
  onehot_exampleB = encB.transform(np.array(exampleB).reshape(-1, 1)).toarray()
  
  if debug >= 2:
    print("Examples:")
    print("One hot exampleA: ", onehot_exampleA)
    print("exampleA shape: ", onehot_exampleA.shape)
    print("One hot exampleB: ", onehot_exampleA)
    print("exampleB shape: ", onehot_exampleB.shape)
    print("")
  
  score = session.run([model], feed_dict={s_tf_teamA_input:onehot_exampleA, s_tf_teamB_input:onehot_exampleB})
  scoreA = score[0][0][0]
  scoreB = score[0][0][1]
  #debug
  if debug >= 2:
    print("Score Prediction:")
    print("Pred: ", score)
    print("Pred type: ", type(score))
    print("ScoreA: ", scoreA)
    print("ScoreA type: ", type(scoreA))
    print("ScoreB: ", scoreB)
    print("ScoreB type: ", type(scoreB))
    print("")
  
  return score, scoreA, scoreB
 
''' Print Match Prediction
* ARGS: 
''' 
def print_prediction(teamA, teamB, result_model, score_model, encA, encB):

  if debug >= 1:
    print("print_prediction arguments:")
    print("teamA: ", teamA)
    print("teamA type: ", type(teamA))
    print("teamB: ", teamB)
    print("teamB type: ", type(teamB))
    print("encA: ", encA)
    print("encA features: ", encA.get_feature_names())
    print("encA params: ", encA.get_params())
    print("encB: ", encB)
    print("encB features: ", encB.get_feature_names())
    print("encB params: ", encB.get_params())
    print("")

  winner, pred = predict_match_result(teamA, teamB, result_model, encA, encB)
  score, scoreA, scoreB = predict_score(teamA, teamB, score_model, encA, encB)
  print(teamA, " ", scoreA, " - ", scoreB, " ", teamB, " : WINNER - ", winner)
  #print(teamA, " ", scoreA, " - ", scoreB, " ", teamB, " : WINNER - ", winner)
  print("")

''' Running the Functions '''  
#getdata
teamA_list, teamB_list, teamA_score_list, teamB_score_list, match_result_list= load_data(data_filename)
#turn into (one hot) arrays
onehot_teamA_array, onehot_teamA_encoder = onehot_array(np.array(teamA_list).reshape(-1, 1))
onehot_teamB_array, onehot_teamB_encoder = onehot_array(np.array(teamB_list).reshape(-1, 1))
teamA_score_array = np.array(teamA_score_list)
teamB_score_array = np.array(teamB_score_list)
match_result_array = np.array(match_result_list)

''' Setting Variables '''
VALIDATE_PERCENT = 0
LEARNING_RATE = 0.1
EPOCHS = 20

''' Running the TF code '''
with tf.Session() as session:
  
  #create train and validation datasets
  teamA_train, teamB_train, teamA_score_train, teamB_score_train, match_result_train, teamA_validate, teamB_validate, teamA_score_validate, teamB_score_validate, match_result_validate = create_dataset(onehot_teamA_array, onehot_teamB_array, teamA_score_array, teamB_score_array, match_result_array, VALIDATE_PERCENT)

  A_x, A_y = teamA_train.shape
  B_x, B_y = teamB_train.shape

  #set up result model
  result_model_output, r_hidden_out, r_tf_teamA_input, r_tf_teamB_input, r_teamA_hidden_weights, r_teamB_hidden_weights, r_linear_weights, r_target = create_match_result_model(A_y, B_y, 5, 1)
  r_loss = tf.losses.mean_squared_error(r_target, result_model_output)
  r_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(r_loss)
  
  #set up score model
  score_model_output, s_hidden_out, s_tf_teamA_input, s_tf_teamB_input, s_teamA_hidden_weights, s_teamB_hidden_weights, s_linear_weights, s_target = create_score_model(A_y, B_y, 5, 2)
  s_loss = tf.losses.mean_squared_error(s_target, score_model_output)
  s_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(s_loss)

  #intialise tf variables
  session.run(tf.global_variables_initializer())
  
  #running/training the model
  for epoch in range(EPOCHS):
    for index in range(A_x):
      #run result model
      r_training, r_error, r_out = session.run([r_train_op, r_loss, result_model_output], feed_dict={r_target:[[match_result_train[index]]], r_tf_teamA_input:[teamA_train[index]], r_tf_teamB_input:[teamB_train[index]]})
      #debug
      if debug >= 1:
        print("Result Epoch: ", epoch)
        print("Result Index: ", index)
        print("Result Output ",  r_out)
        print("Result Loss ",  r_error)
        print("Result Training Operation ",  r_training)
        print("")

      #run score model
      s_training, s_error, s_out = session.run([s_train_op, s_loss, score_model_output], feed_dict={s_target:[[teamA_score_train[index], teamB_score_train[index]]], s_tf_teamA_input:[teamA_train[index]], s_tf_teamB_input:[teamB_train[index]]})
      #debug
      if debug >= 1:
        print("Score Epoch: ", epoch)
        print("Score Index: ", index)
        print("Score Output ",  s_out)
        print("Score Loss ",  s_error)
        print("Score Training Operation ",  s_training)
        print("")
        
  #Using the Model
  print("PREDICTIONS:\n")
  #Day 1
  print("Day 1:\n")
  print_prediction('Samoa 7s', ' Scotland 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('South Africa 7s', ' Japan 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('England 7s', ' Wales 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('USA 7s', ' Spain 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('New Zealand 7s', ' Australia 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Fiji 7s', ' Kenya 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Argentina 7s', ' Canada 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('France 7s', ' Portugal 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder) 
  #Day 2
  print("Day 2:\n")
  print_prediction('Samoa 7s', ' Japan 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('South Africa 7s', ' Scotland 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('England 7s', ' Spain 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('USA 7s', ' Wales 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('New Zealand 7s', ' Kenya 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Fiji 7s', ' Australia 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Argentina 7s', ' Portugal 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('France 7s', ' Canada 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Scotland 7s', ' Japan 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('South Africa 7s', ' Samoa 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Wales 7s', ' Spain 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('USA 7s', ' England 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Australia 7s', ' Kenya 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Fiji 7s', ' New Zealand 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('Canada 7s', ' Portugal 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)
  print_prediction('France 7s', ' Argentina 7s', result_model_output, score_model_output, onehot_teamA_encoder, onehot_teamB_encoder)


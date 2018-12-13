# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:13:45 2018

@author: Krishna
"""

import sys
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.datasets import load_svmlight_file
from gensim.models.word2vec import Word2Vec
import time
import multiprocessing
from gensim.models import KeyedVectors
from sklearn.preprocessing import MultiLabelBinarizer
import operator
from sklearn.metrics import f1_score
#from sklearn import preprocessing

def load_word2vec_model(filename):
    #model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    model = KeyedVectors.load_word2vec_format(filename, binary = True)
    
    print ("\nWord2vec model loaded.")
    return model


def extract_features(feature_list_file):
    features_list = []

    with open(feature_list_file) as f:
        lines = f.read().splitlines()
    
    for i in range(len(lines)):
        if 'ngram' in lines[i]:
        #if 'name' in lines[i]:
            features_list.append(lines[i].split('ngram=')[1].split(', field=')[0])
            #features_list.append(lines[i].split("name='")[1].split("', settings=")[0])
        else:
            features_list.append('special feature')
    
    features_list = np.array(features_list)
    
    return features_list


def load_sparse_dataset(filename, n_features):
    data = load_svmlight_file(filename, n_features=n_features, multilabel = True)
    X, y = np.array(data[0].todense()), data[1]
    
    '''
    for i in range(X.shape[0]):
        den = np.sqrt(np.sum(np.square(X[i])))
        if den > 0:
            X[i] = X[i]/den
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_normalized = min_max_scaler.fit_transform(X)
    '''    
            
    return X, y


def filter_unknown_words(model, word_list, X_train, X_test):
    #word_list = tf_transformer.get_feature_names()
    
    # Find list of unknown words
    word_ids = []
    for i in range(len(word_list)):
        if word_list[i] == 'special feature':
            word_ids.append(i)
        else:
            words_split = word_list[i].split()
            for j in range(len(words_split)):
                if words_split[j] not in model.wv.vocab:
                    word_ids.append(i)
                    continue
            
    # Delete list of unknown words from feature matrix and word_list
    X_train = np.delete(X_train, word_ids, axis = 1)
    X_test = np.delete(X_test, word_ids, axis = 1)
    
    word_list = np.delete(word_list, word_ids, axis = 0)
   
    # Create a dictionary of known words with their ids
    vocab_dict = dict()
    for i in range(len(word_list)):
        vocab_dict[word_list[i]] = i
    
    print ("New features:", len(word_list))
    return word_list, vocab_dict, X_train, X_test


def generate_word_vectors(model, word_list):
    word_vectors = []
    for word in word_list:
        word_split = word.split()
        wordvec_temp = np.zeros(300)
        for i in range(len(word_split)):
            wordvec_temp += model.wv.get_vector(word_split[i])
        word_vectors.append(wordvec_temp/len(word_split))
    return np.array(word_vectors)


def get_train_batch(X_train, y_train, batchsize):
    
    num = np.random.choice(X_train.shape[0], size = batchsize, replace = False)
    batch_X = (X_train)[num]
    batch_y = (y_train)[num]
        
    return (batch_X, batch_y)

# Train a logistic regression model
def train_sklearn_logistic_regression(X_train, X_test, y_train, y_test, logs):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    print("Train accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))
    
    logs.writelines("Train accuracy: " + str(accuracy_score(y_train, model.predict(X_train))) + '\n')
    logs.writelines("Test accuracy: " +  str(accuracy_score(y_test, model.predict(X_test))) + '\n')

# Train the word-vector model
def train_word_vector_model(X_train, X_test, y_train, y_test, n_classes, word_vectors, learning_rate, training_epochs, batch_size, display_step, logs, inter_op_parallelism_threads, intra_op_parallelism_threads, classification_type, with_regularization, reg_parameter, threshold):
    #one_hot = MultiLabelBinarizer()
    #y_test = one_hot.fit_transform(y_test)
    #y_train = one_hot.transform(y_train)
    
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, [None, n_classes])
    
    # Set model weights
    W = tf.Variable(tf.zeros([300, n_classes]))
    theta = tf.matmul(word_vectors.astype(np.float32), W)
    b = tf.Variable(tf.zeros([n_classes]))
    
    # Construct model
    model_output = tf.matmul(x, theta) + b
    
    if classification_type == 'multi-class':
        pred_prob = tf.nn.softmax(model_output)
        normal_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = model_output, labels = y)
        #normal_loss = -tf.reduce_sum(y*tf.log(pred_prob), reduction_indices=[1])
        #normal_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y)
        correct_prediction = tf.equal(tf.argmax(pred_prob, 1), tf.argmax(y, 1))
        
    elif classification_type == 'multi-label':
        pred_prob = tf.nn.sigmoid(model_output)
        normal_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y)
        correct_prediction = tf.reduce_all(tf.equal(tf.cast(tf.greater_equal(pred_prob, threshold), tf.float32), y), reduction_indices = 1)
        
    if with_regularization == 'True':
        reg_l2 = tf.nn.l2_loss(theta)
        beta = reg_parameter
        cost = tf.reduce_sum(normal_loss) + (beta * reg_l2)
        #cost = tf.reduce_mean(normal_loss + beta * reg_l2)
    else:
        cost = tf.reduce_sum(normal_loss)
        #cost = tf.reduce_mean(normal_loss)
    
    # Gradient Descent
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9, 1e-6).minimize(cost)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    config = tf.ConfigProto(device_count={"CPU": multiprocessing.cpu_count()},
                            inter_op_parallelism_threads=inter_op_parallelism_threads,
                            intra_op_parallelism_threads=intra_op_parallelism_threads)
    
    with tf.Session(config=config) as sess:
    
        # Run the initializer
        sess.run(init)
    
        # Training cycle
        for epoch in range(training_epochs):
            #print("Now training epoch:", epoch)
            avg_cost = 0.
            total_batch = int(X_train.shape[0]/batch_size)
            
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = get_train_batch(X_train, y_train, batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch) % display_step == 0:
                acc_train = accuracy.eval({x: X_train, y: y_train})
                acc_test = accuracy.eval({x: X_test, y: y_test})
                
                # Calc instance_f1 - train and test
                pred_prob_train = pred_prob.eval({x: X_train})
                y_pred_train = np.copy(np.greater_equal(pred_prob_train, threshold).astype(int))
                instance_f1_train = calc_instance_f1(y_train, y_pred_train)
                
                pred_prob_test = pred_prob.eval({x: X_test})
                y_pred_test = np.copy(np.greater_equal(pred_prob_test, threshold).astype(int))
                instance_f1_test = calc_instance_f1(y_test, y_pred_test)
                
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                print("Set Accuracy train:", acc_train)
                print("Set Accuracy test:", acc_test)
                print("Instance-F1 train:", instance_f1_train)
                print("Instance-F1 test:", instance_f1_test)
                
                logs.writelines("Epoch: " + str(epoch) + " cost= " + str(avg_cost) + '\n')
                logs.writelines("Set Accuracy train: " + str(acc_train) + '\n')
                logs.writelines("Set Accuracy test: " + str(acc_test) + '\n')
                logs.writelines("Instance-F1 train: " + str(instance_f1_train) + '\n')
                logs.writelines("Instance-F1 test: " + str(instance_f1_test) + '\n')
        
        print("Optimization Finished!")
        
        # Save results
        Theta = theta.eval()
        Weights = W.eval()
        
        pred_prob_train = pred_prob.eval({x: X_train})
        y_pred_train = np.copy(np.greater_equal(pred_prob_train, threshold).astype(int))
        instance_f1_train = calc_instance_f1(y_train, y_pred_train)
        
        pred_prob_test = pred_prob.eval({x: X_test})
        y_pred_test = np.copy(np.greater_equal(pred_prob_test, threshold).astype(int))
        instance_f1_test = calc_instance_f1(y_test, y_pred_test)
        
        acc_train = accuracy.eval({x: X_train, y: y_train})
        acc_test = accuracy.eval({x: X_test, y: y_test})
        
        print("Set Accuracy train:", acc_train)
        print("Set Accuracy test:", acc_test)
        print("Instance-F1 train:", instance_f1_train)
        print("Instance-F1 test:", instance_f1_test)
        
        logs.writelines("Optimization Finished!\n")
        logs.writelines("Set Accuracy train: " + str(acc_train) + '\n')
        logs.writelines("Set Accuracy test: " + str(acc_test) + '\n')
        logs.writelines("Instance-F1 train: " + str(instance_f1_train) + '\n')
        logs.writelines("Instance-F1 test: " + str(instance_f1_test) + '\n')
        
    return Theta, Weights, pred_prob_test


def train_logistic_regression_model(X_train, X_test, y_train, y_test, n_classes, learning_rate, training_epochs, batch_size, display_step, logs, inter_op_parallelism_threads, intra_op_parallelism_threads, classification_type, with_regularization, reg_parameter, threshold):
    #one_hot = MultiLabelBinarizer()
    #y_test = one_hot.fit_transform(y_test)
    #y_train = one_hot.transform(y_train)
    
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, n_classes]) # 0-9 digits recognition => 10 classes
    
    # Set model weights
    W = tf.Variable(tf.zeros([X_train.shape[1], n_classes]))
    #W = tf.Variable(np.array(np.random.randn(X_train.shape[1], n_classes)*np.sqrt(2/(X_train.shape[1] + n_classes)), dtype = np.float32))
    b = tf.Variable(tf.zeros([n_classes]))
    
    # Construct model
    model_output = tf.matmul(x, W) + b
    
    if classification_type == 'multi-class':
        pred_prob = tf.nn.softmax(model_output)
        normal_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = model_output, labels = y)
        #normal_loss = -tf.reduce_sum(y*tf.log(pred_prob), reduction_indices=[1])
        #normal_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y)
        correct_prediction = tf.equal(tf.argmax(pred_prob, 1), tf.argmax(y, 1))
        
    elif classification_type == 'multi-label':
        pred_prob = tf.nn.sigmoid(model_output)
        normal_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y)
        correct_prediction = tf.reduce_all(tf.equal(tf.cast(tf.greater_equal(pred_prob, threshold), tf.float32), y), reduction_indices = 1)
        
    if with_regularization == 'True':
        reg_l2 = tf.nn.l2_loss(W)
        beta = reg_parameter
        cost = tf.reduce_sum(normal_loss) + (beta * reg_l2)
        #cost = tf.reduce_mean(normal_loss + beta * reg_l2)
    else:
        cost = tf.reduce_sum(normal_loss)
        #cost = tf.reduce_mean(normal_loss)
    
    # Gradient Descent
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9, 1e-6).minimize(cost)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    config = tf.ConfigProto(device_count={"CPU": multiprocessing.cpu_count()},
                            inter_op_parallelism_threads=inter_op_parallelism_threads,
                            intra_op_parallelism_threads=intra_op_parallelism_threads)
    
    with tf.Session(config=config) as sess:
    
        # Run the initializer
        sess.run(init)
    
        # Training cycle
        for epoch in range(training_epochs):
            #print("Now training epoch:", epoch)
            avg_cost = 0.
            total_batch = int(X_train.shape[0]/batch_size)
            
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = get_train_batch(X_train, y_train, batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch) % display_step == 0:
                acc_train = accuracy.eval({x: X_train, y: y_train})
                acc_test = accuracy.eval({x: X_test, y: y_test})
                
                # Calc instance_f1 - train and test
                pred_prob_train = pred_prob.eval({x: X_train})
                y_pred_train = np.copy(np.greater_equal(pred_prob_train, threshold).astype(int))
                instance_f1_train = calc_instance_f1(y_train, y_pred_train)
                
                pred_prob_test = pred_prob.eval({x: X_test})
                y_pred_test = np.copy(np.greater_equal(pred_prob_test, threshold).astype(int))
                instance_f1_test = calc_instance_f1(y_test, y_pred_test)
                
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                print("Set Accuracy train:", acc_train)
                print("Set Accuracy test:", acc_test)
                print("Instance-F1 train:", instance_f1_train)
                print("Instance-F1 test:", instance_f1_test)
                
                logs.writelines("Epoch: " + str(epoch) + " cost= " + str(avg_cost) + '\n')
                logs.writelines("Set Accuracy train: " + str(acc_train) + '\n')
                logs.writelines("Set Accuracy test: " + str(acc_test) + '\n')
                logs.writelines("Instance-F1 train: " + str(instance_f1_train) + '\n')
                logs.writelines("Instance-F1 test: " + str(instance_f1_test) + '\n')
        
        print("Optimization Finished!")
        
        # Save results
        pred_prob_train = pred_prob.eval({x: X_train})
        y_pred_train = np.copy(np.greater_equal(pred_prob_train, threshold).astype(int))
        instance_f1_train = calc_instance_f1(y_train, y_pred_train)
        
        pred_prob_test = pred_prob.eval({x: X_test})
        y_pred_test = np.copy(np.greater_equal(pred_prob_test, threshold).astype(int))
        instance_f1_test = calc_instance_f1(y_test, y_pred_test)
        
        acc_train = accuracy.eval({x: X_train, y: y_train})
        acc_test = accuracy.eval({x: X_test, y: y_test})
        
        print("Set Accuracy train:", acc_train)
        print("Set Accuracy test:", acc_test)
        print("Instance-F1 train:", instance_f1_train)
        print("Instance-F1 test:", instance_f1_test)
        
        logs.writelines("Optimization Finished!\n")
        logs.writelines("Set Accuracy train: " + str(acc_train) + '\n')
        logs.writelines("Set Accuracy test: " + str(acc_test) + '\n')
        logs.writelines("Instance-F1 train: " + str(instance_f1_train) + '\n')
        logs.writelines("Instance-F1 test: " + str(instance_f1_test) + '\n')
        
    return pred_prob_test


def train_all_models(X_train, X_test, y_train, y_test, n_classes, word_vectors, learning_rate, n_epochs, display_step, logs, inter_op_parallelism_threads, intra_op_parallelism_threads, classification_type, with_regularization, reg_parameter, threshold):
    # Train logistic regression on the dataset
    #print ("\nTraining sklearn logistic regression..")
    #logs.writelines('\nTraining sklearn logistic regression..\n')
    #train_sklearn_logistic_regression(X_train, X_test, y_train, y_test, logs)
    
    print ("\nTraining vanilla logistic regression model..")
    logs.writelines('\nTraining vanilla logistic regression model..\n')
    pred_lr = train_logistic_regression_model(X_train, X_test, y_train, y_test,
                                              n_classes = n_classes,
                                              learning_rate = learning_rate,
                                              training_epochs = n_epochs,
                                              batch_size = int(X_train.shape[0]*.2),
                                              display_step = display_step,
                                              logs = logs,
                                              inter_op_parallelism_threads = inter_op_parallelism_threads, 
                                              intra_op_parallelism_threads = intra_op_parallelism_threads,
                                              classification_type = classification_type, 
                                              with_regularization = with_regularization,
                                              reg_parameter = reg_parameter,
                                              threshold = threshold)
    
    # Train word-vector model on the dataset
    print ("\nTraining word-vector model..")
    logs.writelines('\nTraining word-vector model..\n')
    theta, weights, pred_wv = train_word_vector_model(X_train, X_test, y_train, y_test,
                                                      n_classes = n_classes,
                                                      word_vectors = word_vectors,
                                                      learning_rate = learning_rate,
                                                      training_epochs = n_epochs,
                                                      batch_size = int(X_train.shape[0]*.2),
                                                      display_step = display_step,
                                                      logs = logs,
                                                      inter_op_parallelism_threads = inter_op_parallelism_threads, 
                                                      intra_op_parallelism_threads = intra_op_parallelism_threads,
                                                      classification_type = classification_type, 
                                                      with_regularization = with_regularization,
                                                      reg_parameter = reg_parameter,
                                                      threshold = threshold)
    
    logs.writelines('\n')
    
    #return pred_lr, pred_wv
    return pred_lr, pred_wv, theta, weights


def write_final_predictions(labels, pred_prob, y_test, threshold, filename):
    predictions = open(filename, 'w')
    
    for i in range(len(pred_prob)):
        ind_top5 = pred_prob[i].argsort()[-5:][::-1]
        ind_thr = np.where(pred_prob[i] > threshold)[0]
        ind = np.array(list(set(ind_thr).union(set(ind_top5))))
        top_pred_list = []
        for k in range(len(ind)):
            top_pred_list.append([labels[ind[k]], pred_prob[i][ind[k]]])
        top_pred_list.sort(key = operator.itemgetter(1), reverse = True)
        doc = ''
        for j in range(len(top_pred_list)):
            if y_test[i][top_pred_list[j][0]] == 1:
                doc+='=='+str(top_pred_list[j][0])+':'
                doc+=str("{:.5f}".format(top_pred_list[j][1]))+'==,'
            else:
                doc+=str(top_pred_list[j][0])+':'
                doc+=str("{:.5f}".format(top_pred_list[j][1]))+','
        
        true_ind = list(np.where(y_test[i]==1)[0])
        missed_labels = list(set(true_ind).difference(set(list(ind))))
        missed_labels.sort()
        if len(missed_labels)>0:
            doc+='\t'
            for l in range(len(missed_labels)):
                doc+='=='+str(missed_labels[l])+'==,'
        predictions.writelines(doc[:-1] + '\n')
    
    predictions.close()
    
    
def write_features(features_list, filename):
    features_file = open(filename, 'w')
    
    for i in range(len(features_list)):
        features_file.writelines(features_list[i] + '\n')
        
    features_file.close()
    

def calc_instance_f1(y_true, y_pred):
    f1 = []
    for i in range(len(y_true)):
        if (y_pred[i] == 0).all():
            f1.append(0.0)
        else:
            f1.append(f1_score(y_true[i], y_pred[i]))
    
    instance_f1 = np.mean(np.array(f1))
    
    return instance_f1

if __name__ == '__main__':
    
    inputdata = sys.argv
    
    train_file = inputdata[1]
    test_file = inputdata[2]
    feature_list_file = inputdata[3]
    pretrained_word2vec_folderpath = inputdata[4]
    label_translator_file = inputdata[5]
    learning_rate = float(inputdata[6])
    n_epochs = int(inputdata[7])
    display_step = int(inputdata[8])
    es_index_name = inputdata[9]
    output_folderpath = inputdata[10]
    retrain_wordvectors = inputdata[11]
    pretrained_word2vec_file = inputdata[12]
    inter_op_parallelism_threads = int(inputdata[13])
    intra_op_parallelism_threads = int(inputdata[14])
    class_type = inputdata[15]
    reg = inputdata[16]
    thr = float(inputdata[17])
    train_wordvector_model = inputdata[18]
    reg_parameter = float(inputdata[19])
    
    if class_type == 'multi-class':
        if reg == 'True':
            output_foldername = output_folderpath + 'run_' + str(int(time.time())) + '.' + es_index_name + '.lr=' + str(learning_rate) + '.epochs=' + str(n_epochs) + '.regularization=' + str(reg) + '.reg.parameter=' + str(reg_parameter) +  '.' + class_type + os.sep
        else:
            output_foldername = output_folderpath + 'run_' + str(int(time.time())) + '.' + es_index_name + '.lr=' + str(learning_rate) + '.epochs=' + str(n_epochs) + '.regularization=' + str(reg) +  '.' + class_type + os.sep
    elif class_type == 'multi-label':
        if reg == 'True':
            output_foldername = output_folderpath + 'run_' + str(int(time.time())) + '.' + es_index_name + '.lr=' + str(learning_rate) + '.epochs=' + str(n_epochs) + '.regularization=' + str(reg) + '.reg.parameter=' + str(reg_parameter) + '.' + class_type + '.pred_threshold=' + str(thr) + os.sep
        else:
            output_foldername = output_folderpath + 'run_' + str(int(time.time())) + '.' + es_index_name + '.lr=' + str(learning_rate) + '.epochs=' + str(n_epochs) + '.regularization=' + str(reg) +  '.' + class_type + '.pred_threshold=' + str(thr) + os.sep
    
    features_list = extract_features(feature_list_file)
    X_train, y_train = load_sparse_dataset(train_file, len(features_list))
    X_test, y_test = load_sparse_dataset(test_file, len(features_list))
    #y_test = [x[:-1] for x in y_test]
    
    y_test = [item[0] for item in y_test]
    it = iter(y_test)
    y_test = list(zip(it))
    
    if retrain_wordvectors == 'True':
        model = Word2Vec.load(pretrained_word2vec_folderpath + "retrained-word2vec.model")
    else:
        model = KeyedVectors.load_word2vec_format(pretrained_word2vec_file, binary = True)
    #model = load_word2vec_model(pretrained_word2vec_folderpath)
    
    if train_wordvector_model == 'True':
        features_list, vocab_dict, X_train, X_test = filter_unknown_words(model, features_list, X_train, X_test)
        
        
        import random
        random.seed(13)
        for i in range(X_train.shape[1]):
            if random.random() <= 0.95:
                X_train[:,i] = 0 * X_train[:,i]
        
            
        word_vectors = generate_word_vectors(model, features_list)
    
    
    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)
    
    logs = open(output_foldername + 'logs.txt', 'w')
    
    #############
    with open(label_translator_file) as f:
        lines = f.read().splitlines()
    
    #labels = []
    #for i in range(len(lines)):
    #    labels.append(int(lines[i].split('=')[0]))
        
    #labels = np.array(labels)
    
    
    labels = []
    for i in range(len(lines)):
        labels.append(tuple([int(lines[i].split('=')[0])]))
        
    n_classes = len(labels)
    
    one_hot = MultiLabelBinarizer()
    labels_onehot = one_hot.fit_transform(labels)
    y_train = one_hot.transform(y_train)
    y_test = one_hot.transform(y_test)
    
    labels = np.array(labels).flatten()
    ##############
    
    if train_wordvector_model == 'False':
        print("\nTotal Features:", len(features_list))
        logs.writelines("\nTotal Features: " +  str(len(features_list)))
        
        print("\nTraining with the original dataset..")
        logs.writelines("\nTraining with the original dataset..")
        
        print ("\nTraining vanilla logistic regression model..")
        logs.writelines('\nTraining vanilla logistic regression model..\n')
        
        start = time.time()
        
        pred_lr = train_logistic_regression_model(X_train, X_test, y_train, y_test,
                                                  n_classes = n_classes,
                                                  learning_rate = learning_rate,
                                                  training_epochs = n_epochs,
                                                  batch_size = int(X_train.shape[0]*.2),
                                                  display_step = display_step,
                                                  logs = logs,
                                                  inter_op_parallelism_threads = inter_op_parallelism_threads, 
                                                  intra_op_parallelism_threads = intra_op_parallelism_threads,
                                                  classification_type = class_type, 
                                                  with_regularization = reg,
                                                  reg_parameter = reg_parameter,
                                                  threshold = thr)
        
        end = time.time()
    else:
        start = time.time()
        
        pred_lr, pred_wv, theta, weights = train_all_models(X_train, X_test, y_train, y_test,
                                                            n_classes = n_classes,
                                                            word_vectors = word_vectors,
                                                            learning_rate = learning_rate,
                                                            n_epochs = n_epochs,
                                                            display_step = display_step,
                                                            logs = logs,
                                                            inter_op_parallelism_threads = inter_op_parallelism_threads,
                                                            intra_op_parallelism_threads = intra_op_parallelism_threads,
                                                            classification_type = class_type,
                                                            with_regularization = reg,
                                                            reg_parameter = reg_parameter,
                                                            threshold = thr)
        
        end = time.time()
        
    print('Total time: ', end-start)
    
    logs.close()
    
    # Saving final results as pickled objects
    print ("\nSaving data..")
    
    #np.save(output_foldername + 'theta_orig', theta_orig)
    #np.save(output_foldername + 'weights_orig', weights_orig)
    
    np.save(output_foldername + 'features_list.npy', features_list)
    #np.savetxt(fname = output_foldername + 'predictions.txt', 
    #           X = pred_prob, fmt = '%1.10f', delimiter = ' ')
    #np.save(output_foldername + 'predictions', pred_prob)
    
    np.save(output_foldername + 'pred_lr.npy', pred_lr)
    
    
    write_final_predictions(labels, pred_lr, y_test, thr, output_foldername + 'predictions_LR.txt')
    
    if train_wordvector_model == 'True':
        write_final_predictions(labels, pred_wv, y_test, thr, output_foldername + 'predictions_WV.txt')
        np.save(output_foldername + 'vocab_dict', vocab_dict)
        np.save(output_foldername + 'pred_wv.npy', pred_wv)
        np.save(output_foldername + 'theta.npy', theta)
        np.save(output_foldername + 'weights.npy', weights)
    write_features(features_list, output_foldername + 'features_list.txt')
    
    print ("\nPre-processed data present at", output_foldername)
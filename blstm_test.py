# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 08:52:47 2017

@author: ZMJ
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#Import MNIST data
from tf.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#Parameters
learning_rate=0.001
trianing_iters=100000
batch_size=128
display_step=10

#Network Parameters
n_input=28
n_steps=28
n_hidden=128
n_classes=10

#tf Graph input
x=tf.placeholder("float",[None,n_steps,n_input])
y=tf.placeholder("float",[None,n_classes])

#Define weights
weights={
	"out":tf.Variable(tf.truncated_normal([2*n_hidden,n_classes]))
	}
biases={
	"out":tf.Variable(tf.zeros([n_classes]))
	}

#Define BiRNN
def BiRNN(2,weights,biases):
	#Current data shape:(batch_szie,n_steps,n_input)
	#Reqiured data shape:'n_steps' tensors list of shape(batch_size,n_input)
	#Unstack to get a list of 'n_steps' tensors of shape(batch_size,n_input)
	
	x=tf.unstack(x,n_steps,1)
	
	#Define bw_lstm
	lstm_bw_cell=rnn.BasicalLSTMCell(n_hidden,forget_biases=0.)
	#Define_fw_lstm
	lstm_bw_cell=rnn.BasicalLSTMCell(n_hidden,forget_biases=0.)
	
	#Get lstm cell output
	try:
		output,_,_=rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,\
																						dtype=tf.float32)
	except Exception:#Old tensorflow version only returns outputs not states
		output=rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,\
																						dtype=tf.float32)
		return tf.matmul(output,weights["out"])+biases["out"]

pred=BiRNN(x,weights,biases)

#Define loss and optimizer
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#Evaluate model
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initializing the variables
init=tf.initialize_all_variables()

#Launch the graph
with tf.Session as sess:
	sess.run(init)
	step=1
	while step*batch_size<training_iters:
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		#Reshape data to get 28 seq of 28 elements
		batch_x=batch_x.reshape([batch_size,n_steps,n_input])
		#Run optimization op(backProp)
		sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
		if step%display_step==0:
			acc=sess.run(accuracy)
			l=sess.run(loss)
			print("Iter %s:loss %s,accuracy %s"%(step,l,acc))
		step+=1
	print("Optimization finished!")
			

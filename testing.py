import numpy as np
import tensorflow as tf

from keras import backend as K

x = np.array([0, 0.1, 0.2, 0.3])
xt = tf.convert_to_tensor(x, dtype=tf.float32)
xm = K.mean(xt)

y = np.array([0.5, 0.6, 0.7, -0.2])
yt = tf.convert_to_tensor(y, dtype=tf.float32)
ym = K.mean(yt)

pm = xm * ym

rmse = K.sqrt(K.mean(K.square(xt - yt), axis=-1))

dt = xt - yt

#initialize the variable
init_op = tf.global_variables_initializer()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(xt))
    print (sess.run(yt))
    print (sess.run(dt))
    print (sess.run(xm))
    print (sess.run(ym))
    print (sess.run(rmse))


############################################################

x = np.array([43, 21, 25, 42, 57, 59])
xt = tf.convert_to_tensor(x, dtype=tf.float32)


y = np.array([99, 65, 79, 75, 87, 81])
yt = tf.convert_to_tensor(y, dtype=tf.float32)


mx = K.mean(xt)
my = K.mean(yt)
xm, ym = xt-mx, yt-my
r_num = K.mean(xm * ym)
#r_den = K.sum(K.sum(K.square(xm)) * K.sum(K.square(ym)))
r_den = K.sqrt(K.mean(K.square(xm))) * K.sqrt(K.mean(K.square(ym)))
r = r_num / r_den

with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    print (sess.run(r))
############################################################
def rescale2neutral(listN, currentmin, currentmax):
    # takes a tensor with values between 1 and M
    # translates it to [-M/2, M/2]
    neutral = (currentmax + currentmin)/2
    v =  listN.get_shape()
    dim = v[0].value
   
    ind = [i for i in range(dim)]

    listN = tf.Variable(listN)
    listN = tf.scatter_update(listN, ind, listN-neutral)

    return listN
############################################################
def SAGR(y_true,y_pred):
    '''Computes Sign Agreement Metric (SAGR)'''
    y_true_neut = rescale2neutral(y_true, 1, 7)
    y_pred_neut = rescale2neutral(y_pred, 1, 7)

#    if((K.less(y_pred,0.5) and K.less(y_true,0.5)) or (K.greater(y_pred,0.5) and K.greater(y_true,0.5)) or
#       (K.equal(y_pred,0.5) and K.equal(y_true,0.5))):
#        sumF+=1
    #result = tf.cond(K.less(y_pred_neut,0), true_fn=(lambda: tf.add(sumF, 1)), false_fn =(lambda: None))

    temp = y_true.get_shape()
    dim = temp.num_elements()
    sumF = 0
    for i in range(dim):
        s_true = tf.sign(tf.gather(y_true_neut, i)) 
        s_pred = tf.sign(tf.gather(y_pred_neut, i))
        sumF = tf.cond(tf.equal(s_true,s_pred) , true_fn=(lambda: tf.add(sumF, 1)), false_fn =(lambda: tf.add(sumF, 0)))
    
    return(sumF)
############################################################
x = np.array([4, -2, 5, 4])
xt = tf.convert_to_tensor(x, dtype=tf.float32)

y = np.array([4, 2, 6, 7])
yt = tf.convert_to_tensor(y, dtype=tf.float32)

xtm = rescale2neutral(xt, 1, 7)
ytm = rescale2neutral(yt, 1, 7)


sagrxy = SAGR(xt, yt)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init) 
    print (sess.run(xt))
    print (sess.run(yt))
    print (sess.run(xtm))
    print (sess.run(ytm))
    print (sess.run(sagrxy))
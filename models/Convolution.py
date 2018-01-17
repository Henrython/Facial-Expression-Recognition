import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

class ConvNet128(object):
    def __init__(self,Height,Width,Channel,lam,drop_out,learning_rate):
        self.cache={"loss":[],"accuracy":[]}
        self.H=Height
        self.Width=Width
        self.lam=lam
        self.drop=drop_out
        self.learning_rate=learning_rate
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,Height,Width,Channel],name="input_x")
        self.Y=tf.placeholder(dtype=tf.int32,shape=[None],name="input_y")
        self.is_training=tf.placeholder(dtype=tf.bool,name="is_training")
        self._BuildGraph()
    
    def Conv(self,input,filters,kernel_size,strides,padding="SAME",activation=tf.nn.relu):
        return tf.layers.conv2d(input,filters=filters,kernel_size=kernel_size,strides=strides,padding="SAME",activation=activation,\
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),\
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lam))
    
    def Pooling(self,input,window_shape,method,padding,strides):
        return tf.nn.pool(input,window_shape,method,padding,strides=strides)
    
    def batch_norm(self,input,is_training):
        return tf.layers.batch_normalization(inputs=input,axis=-1,momentum=0.95,epsilon=0.0001,center=True,scale=True,training=is_training,\
                                               trainable=True)
    def _BuildGraph(self):
        X=self.X
        lam=self.lam
        drop=self.drop
        is_training=self.is_training
        
        with tf.name_scope("conv1"):
            layer=self.Conv(X,128,[3,3],[1,1])
 
        #print(layer.shape)
        with tf.name_scope("batch_norm1"):
            layer=self.batch_norm(layer,is_training)
        
        #print(layer.shape)
        with tf.name_scope("drop_out1"):
            layer=tf.layers.dropout(layer,drop,training=is_training)
        
        with tf.name_scope("conv2"):
            layer=self.Conv(layer,64,[3,3],[2,2])
         
        #print(layer.shape)
        with tf.name_scope("max_pool"):
            layer=self.Pooling(layer,[2,2],"MAX","SAME",[2,2])
        
        #print(layer.shape)        
        with tf.name_scope("conv3"): 
            layer=self.Conv(layer,64,[3,3],[2,2])
   
        #print(layer.shape)
        with tf.name_scope("avg_pool"):
            layer=self.Pooling(layer,[3,3],"AVG","VALID",[1,1])
        
        #print(layer.shape)
        with tf.name_scope("flat"):
            layer=tf.reshape(layer,[-1,4*4*64])
     
        #print(layer.shape)
        with tf.name_scope("drop_out2"):
            layer=tf.layers.dropout(layer,rate=drop,training=is_training)
                
        #print(layer.shape)
        with tf.name_scope("dense"):
            layer=tf.layers.dense(layer,7,kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))
    
        #print(layer.shape)
        self.predict_op=layer
        
        with tf.name_scope("loss"):
            self.loss_func=tf.losses.softmax_cross_entropy
            self.loss_val=self.compute_cost(layer,self.loss_func)
      
        self.global_step=tf.Variable(0,trainable=False,name='global_step')
        
        with tf.name_scope("learning_rate_decay"):
            self.decay_learning=tf.train.exponential_decay(self.learning_rate,self.global_step,2000,0.97,staircase=False)       
        
        with tf.name_scope("optimizer"):
            self.optimizer=tf.train.AdamOptimizer(self.decay_learning)
        
        with tf.name_scope("training_op"):
            self.train_op=self.objective(self.loss_val,self.optimizer,self.global_step)

    
    def compute_cost(self,y_hat,loss_function):
        output_size=y_hat.shape[-1]
        with tf.name_scope("training_loss"):
            loss=loss_function(tf.one_hot(tf.cast(self.Y,tf.int32),depth=output_size,axis=-1),logits=y_hat)
            mean_loss=tf.reduce_mean(loss)
        return mean_loss
    
    def objective(self,loss,optimizer,global_step):
        '''compute the training objective you want e.g min{loss}'''
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss,global_step)
        return train_op
        
    def train(self,sess,Xd,Yd,y_hat,loss_val,train_object=None,epoch=1,batch_size=64,print_every=50,plot_losses=True):
        train_indicies=np.arange(Xd.shape[0])
        pred=tf.argmax(y_hat,axis=1,output_type=tf.int32)
        true=self.Y
        correct_prediction = tf.equal(pred, true)
        variables=[loss_val,correct_prediction,None]
        training=train_object is not None
        if training:
            variables[-1]=train_object
        iter_ct=0
        for e in range(epoch):
            correct=0
            losses=[]
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
                actual_batch_size = len(idx)
                feed_dict={self.X:Xd[idx,:,:,:],self.Y:Yd[idx],self.is_training:training}
                loss,corr,_=sess.run(variables,feed_dict=feed_dict)
                losses.append(loss*actual_batch_size)
                correct+=np.sum(corr)
                if iter_ct%print_every==0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_ct,loss,np.sum(corr)/actual_batch_size))
                iter_ct+=1
            total_loss=np.sum(losses)/Xd.shape[0]
            total_accuracy=correct/Xd.shape[0]
            self.cache["loss"].append(total_loss)
            self.cache["accuracy"].append(total_accuracy)
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                .format(total_loss,total_accuracy,e+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
    
    
    def Fit(self,sess,Xd,Yd,epoch=1):
        self.train(sess,Xd,Yd,self.predict_op,self.loss_val,self.train_op,epoch)
    
    def Predict(self,sess,Xd):
        N=Xd.shape[0]
        batch_size=64
        y_hat=self.predict_op
        train_indicies=np.arange(N)
        prediction=np.empty(shape=[N,y_hat.shape[-1]],dtype=np.float32)
        for i in range(int(math.ceil(N/batch_size))):
            start_idx = (i*batch_size)%Xd.shape[0]
            idx=train_indicies[start_idx:start_idx+batch_size]
            actual_batch_size=len(idx)
            feed_dict={self.X:Xd[idx,:],self.is_training:False}            
            prediction[idx,:]=sess.run(y_hat,feed_dict=feed_dict)
        return prediction
    
        
        
class ConvNet32(ConvNet128):
    def _BuildGraph(self):
        X=self.X
        lam=self.lam
        drop=self.drop
        is_training=self.is_training
        
        with tf.name_scope("conv1"):
            conv1=self.Conv(X,32,[5,5],[1,1])
 
        #print(conv1.shape)
        with tf.name_scope("batch_norm1"):
            conv1_bn=self.batch_norm(conv1,is_training)
                                               
        with tf.name_scope("conv2"):
            conv2=self.Conv(conv1_bn,32,[5,5],[1,1])
        
        #print(conv2.shape)
        with tf.name_scope("drop_out1"):
            conv2_drop=tf.layers.dropout(conv2,drop,training=is_training)
        
        with tf.name_scope("conv3"):
            conv3=self.Conv(conv2_drop,64,[3,3],[2,2])
         
        #print(conv3.shape)
        with tf.name_scope("max_pool1"):
            conv3_max=self.Pooling(conv3,[2,2],"MAX","SAME",[2,2])
        
        #print(conv3.shape)        
        with tf.name_scope("conv4"): 
            conv4=self.Conv(conv3_max,64,[3,3],[2,2])
 
        #print(conv4.shape)
        with tf.name_scope("drop_out2"):
            conv4_drop=tf.layers.dropout(conv4,rate=drop,training=is_training)
        
        with tf.name_scope("conv5"):
            conv5=self.Conv(conv4_drop,128,[2,2],[1,1])
            
        #print(conv5.shape)
        with tf.name_scope("conv6"):
            conv6=self.Conv(conv5,128,[2,2],[1,1])
        
        
        #print(conv6.shape)
        with tf.name_scope("avg_pool"):
            conv6_avg=self.Pooling(conv6,[3,3],"AVG","VALID",[1,1])
           
        #print(conv6_avg.shape) 
        with tf.name_scope("flat"):
            conv6_flat=tf.reshape(conv6_avg,[-1,4*4*128])
           
       # print(conv6_flat.shape)
        with tf.name_scope("dense1"):
            dense1=tf.layers.dense(conv6_flat,576,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))
     
        #print(dense1.shape)
        with tf.name_scope("drop_out3"):
            dense1_drop=tf.layers.dropout(dense1,rate=drop,training=is_training)
      
        #print(dense1_drop.shape)
        with tf.name_scope("dense2"):
            dense2=tf.layers.dense(dense1_drop,7,kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))
    
        #print(dense2.shape)
        self.predict_op=dense2
        
        with tf.name_scope("loss"):
            self.loss_func=tf.losses.softmax_cross_entropy
            self.loss_val=self.compute_cost(dense2,self.loss_func)
      
        self.global_step=tf.Variable(0,trainable=False,name='global_step')
        
        with tf.name_scope("learning_rate_decay"):
            self.decay_learning=tf.train.exponential_decay(self.learning_rate,self.global_step,2000,0.97,staircase=False)       
        
        with tf.name_scope("optimizer"):
            self.optimizer=tf.train.AdamOptimizer(self.decay_learning)
        
        with tf.name_scope("training_op"):
            self.train_op=self.objective(self.loss_val,self.optimizer,self.global_step)
            
        

class ResNet(ConvNet128):

    def _ResUnit(self,input_layer,kernel_numbers,kernel_size,is_training,name):
        with tf.name_scope(name):
            with tf.name_scope("batch_norm_1"):
                unit1=self.batch_norm(input_layer,is_training)
            with tf.name_scope("activation"):
                unit2=tf.nn.relu(unit1)
            with tf.name_scope("conv"):
                unit3=self.Conv(unit2,kernel_numbers,kernel_size,[1,1])
            with tf.name_scope("batch_norm_2"):
                unit4=self.batch_norm(unit2,is_training)
            with tf.name_scope("conv"):
                unit5=self.Conv(unit4,kernel_numbers,kernel_size,[1,1],activation=None)
            with tf.name_scope("add"):
                unit6=unit5+input_layer
        return unit6
    
    def _BuildGraph(self):
        X=self.X
        lam=self.lam
        drop=self.drop
        is_training=self.is_training
        
        with tf.name_scope("7x7conv/2"):
            layer=self.Conv(X,64,[7,7],[1,1])
        for i in range(1,3):
            for j in range(3):
                layer=self._ResUnit(layer,64,[3,3],is_training,"ResBlock")
            with tf.name_scope("conv_stride"+str(i+1)):
                layer=self.Conv(layer,64,[3,3],[2,2])
        
        with tf.name_scope("drop_out0"):
            layer=tf.layers.dropout(layer,rate=drop,training=is_training)
        
        with tf.name_scope("3x3conv"):
            layer=self.Conv(layer,128,[3,3],[1,1])
        
        for i in range(3,5):
            for j in range(2):
                layer=self._ResUnit(layer,128,[3,3],is_training,"ResBlock")
            with tf.name_scope("conv_stride"+str(i+1)):                
                layer=self.Conv(layer,128,[3,3],[2,2])
            with tf.name_scope("drop_out"+str(i)):
                layer=tf.layers.dropout(layer,rate=drop,training=is_training)
                #print(layer.shape)
        
        with tf.name_scope("avg_pool"):
            layer=self.Pooling(layer,[3,3],"AVG","VALID",[1,1])
            #print(layer.shape)
            
        with tf.name_scope("dense"):
            layer=tf.reshape(layer,[-1,128])
            layer=tf.layers.dense(layer,7,kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))
            #print(layer.shape)
        self.predict_op=layer
        
        with tf.name_scope("loss"):
            self.loss_func=tf.losses.softmax_cross_entropy
            self.loss_val=self.compute_cost(layer,self.loss_func)
      
        self.global_step=tf.Variable(0,trainable=False,name='global_step')
        
        with tf.name_scope("learning_rate_decay"):
            self.decay_learning=tf.train.exponential_decay(self.learning_rate,self.global_step,2000,0.97,staircase=False)       
        
        with tf.name_scope("optimizer"):
            self.optimizer=tf.train.AdamOptimizer(self.decay_learning)
        
        with tf.name_scope("training_op"):
            self.train_op=self.objective(self.loss_val,self.optimizer,self.global_step)
        
        
                        
            
        
        
        
import tensorflow as tf
import pandas as pd
import numpy as np
import math

class RnnCell(object):
    def __init__(self,feature_size,cell_size,x_weight_initializer,hidden_weight_initializer,\
	bias_initializer,layer="Layer1"):
        self.cell_size=cell_size
        self.feature_size=feature_size
        self.x_weight_initializer=x_weight_initializer
        self.hidden_weight_initializer=hidden_weight_initializer
        self.bias_initializer=bias_initializer        
        with tf.variable_scope("Rnn_input_weight"+layer,initializer=x_weight_initializer):
            self.Wx=tf.get_variable('Wx',dtype=tf.float64,shape=[cell_size,feature_size])
        with tf.variable_scope("Rnn_hidden_weight"+layer,initializer=hidden_weight_initializer):
            self.Wh=tf.get_variable('Wh',dtype=tf.float64,shape=[cell_size,cell_size])
        with tf.variable_scope("Rnn_bias_weight"+layer,initializer=bias_initializer):
            self.Wb=tf.get_variable('Wb',dtype=tf.float64,shape=[cell_size,1])
    def __call__(self,input,hidden):
        """cell_size must equal to hidden size"""
        assert hidden.shape[0]==self.cell_size
        out=next_hidden=tf.tanh(tf.matmul(self.Wx,input)+tf.matmul(self.Wh,hidden)+self.Wb)
        return out,next_hidden

class LstmCell(object):
    def __init__(self,feature_size,cell_size,x_weight_initializer,hidden_weight_initializer,\
	bias_initializer,forget_bias=1,peep=False,layer="layer1"):
        self.cell_size=cell_size
        self.feature_size=feature_size
        self.x_weight_initializer=x_weight_initializer
        self.hidden_weight_initializer=hidden_weight_initializer
        self.bias_initializer=bias_initializer
        self.peep=peep
        self.layer=layer
        with tf.variable_scope("Lstm_x_weight"+layer,initializer=self.x_weight_initializer):
            # weight matrix
            self.Ix=tf.get_variable("Ix",dtype=tf.float64,shape=[cell_size,feature_size])
            self.Fx=tf.get_variable('Fx',dtype=tf.float64,shape=[cell_size,feature_size])
            self.Ox=tf.get_variable('Ox',dtype=tf.float64,shape=[cell_size,feature_size])
            self.Gx=tf.get_variable('Gx',dtype=tf.float64,shape=[cell_size,feature_size])
        with tf.variable_scope("Lstm_hidden_weight"+layer,initializer=self.hidden_weight_initializer):
            self.Ih=tf.get_variable('Ih',dtype=tf.float64,shape=[cell_size,cell_size])
            self.Fh=tf.get_variable('Fh',dtype=tf.float64,shape=[cell_size,cell_size])
            self.Oh=tf.get_variable('Oh',dtype=tf.float64,shape=[cell_size,cell_size])
            self.Gh=tf.get_variable('Gh',dtype=tf.float64,shape=[cell_size,cell_size])
        with tf.variable_scope("Lstm_peep_weight"+layer,initializer=tf.random_normal_initializer()):
            self.Ip=tf.get_variable("Ip",dtype=tf.float64,shape=[cell_size,1])
            self.Fp=tf.get_variable("Fp",dtype=tf.float64,shape=[cell_size,1])
            self.Op=tf.get_variable("Op",dtype=tf.float64,shape=[cell_size,1])
        with tf.variable_scope("Lstm_bias_weight"+layer,initializer=self.bias_initializer):
            self.Fb=tf.get_variable("Fb",dtype=tf.float64,shape=[cell_size,1],initializer=tf.constant_initializer(forget_bias))
            self.Ib=tf.get_variable("Ib",dtype=tf.float64,shape=[cell_size,1])
            self.Ob=tf.get_variable("Ob",dtype=tf.float64,shape=[cell_size,1])
            self.Gb=tf.get_variable("Gb",dtype=tf.float64,shape=[cell_size,1])
    
    def __call__(self,input,prev_states):
        '''prev_states should be tuple of previous hidden and cell'''
        assert input.shape[0]==self.feature_size and prev_states[0].shape[0]==self.cell_size\
        and prev_states[1].shape[0]==self.cell_size
        H1,C1=prev_states
        with tf.name_scope("input_gate"):
            I1=tf.sigmoid(tf.matmul(self.Ix,input)+tf.matmul(self.Ih,H1)+self.peep*self.Ip*C1+self.Ib)
        with tf.name_scope("forget_gate"):
            F1=tf.sigmoid(tf.matmul(self.Fx,input)+tf.matmul(self.Fh,H1)+self.peep*self.Fp*C1+self.Fb)
        with tf.name_scope("gate"):
            G1=tf.tanh(tf.matmul(self.Gx,input)+tf.matmul(self.Gh,H1)+self.Gb)
        with tf.name_scope("cell_state"):
            next_C=I1*G1+F1*C1
        with tf.name_scope("output_gate"):
            O1=tf.sigmoid(tf.matmul(self.Ox,input)+tf.matmul(self.Oh,H1)+self.peep*self.Op*C1+self.Ob)
        with tf.name_scope("hidden_state"):
            out=next_hidden=O1*tf.tanh(next_C)    
        return (out,(next_hidden,next_C))
        
class GRU(object):
    def __init__(self,feature_size,cell_size,x_weight_initializer,hidden_weight_initializer,\
	bias_initializer,layer="layer1"):
        self.cell_size=cell_size
        self.feature_size=feature_size
        self.x_weight_initializer=x_weight_initializer
        self.hidden_weight_initializer=hidden_weight_initializer
        self.bias_initializer=bias_initializer        
        with tf.variable_scope("GRU_x_weigth"+layer,initializer=self.x_weight_initializer):
            #weight matrix
            self.Zx=tf.get_variable("Zx",dtype=tf.float64,shape=[cell_size,feature_size])
            self.Rx=tf.get_variable("Rx",dtype=tf.float64,shape=[cell_size,feature_size])
            self.Hx=tf.get_variable("Hx",dtype=tf.float64,shape=[cell_size,feature_size])
        with tf.variable_scope("GRU_hidden_weight"+layer,initializer=self.hidden_weight_initializer):
            self.Zh=tf.get_variable("Zh",dtype=tf.float64,shape=[cell_size,cell_size])
            self.Rh=tf.get_variable("Rh",dtype=tf.float64,shape=[cell_size,cell_size])
            self.Hh=tf.get_variable("Hh",dtype=tf.float64,shape=[cell_size,cell_size])
        with tf.variable_scope("GRU_bias_weight"+layer,initializer=self.bias_initializer):
            self.Zb=tf.get_variable("Zb",dtype=tf.float64,shape=[cell_size,1])
            self.Rb=tf.get_variable("Rb",dtype=tf.float64,shape=[cell_size,1])
            self.Hb=tf.get_variable("Hb",dtype=tf.float64,shape=[cell_size,1])
    def __call__(self,input,prev_hidden):
        assert prev_hidden.shape[0]==self.cell_size
        H=prev_hidden
        with tf.name_scope("update_gate"):
            Z=tf.sigmoid(tf.matmul(self.Zx,input)+tf.matmul(self.Zh,H)+self.Zb)
        with tf.name_scope("reset_gate"):
            R=tf.sigmoid(tf.matmul(self.Rx,input)+tf.matmul(self.Rh,H)+self.Rb)
        with tf.name_scope("candidate_activation"):
            H_hat=tf.tanh(tf.matmul(self.Hx,input)+tf.matmul(self.Hh,R*H)+self.Hb)
        with tf.name_scope("next_hidden_layer"):
            out=next_hidden=(1-Z)*H+Z*H_hat  
        return out,next_hidden
        
class LstmpCell(LstmCell):
    def __init__(self,feature_size,cell_size,projection_size,weight_initializer,\
	bias_initializer,forget_bias=1,peep=False,layer="layer1"):
        self.cell_size=cell_size
        self.feature_size=feature_size
        self.projection_size=projection_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.peep=peep
        self.layer=layer
        with tf.variable_scope("Lstmp_weight"+layer,initializer=self.weight_initializer):
            self.Ix=tf.get_variable("Ix",dtype=tf.float64,shape=[cell_size,feature_size])
            self.Fx=tf.get_variable('Fx',dtype=tf.float64,shape=[cell_size,feature_size])
            self.Ox=tf.get_variable('Ox',dtype=tf.float64,shape=[cell_size,feature_size])
            self.Gx=tf.get_variable('Gx',dtype=tf.float64,shape=[cell_size,feature_size])
            self.Ih=tf.get_variable('Ih',dtype=tf.float64,shape=[cell_size,projection_size])
            self.Fh=tf.get_variable('Fh',dtype=tf.float64,shape=[cell_size,projection_size])
            self.Oh=tf.get_variable('Oh',dtype=tf.float64,shape=[cell_size,projection_size])
            self.Gh=tf.get_variable('Gh',dtype=tf.float64,shape=[cell_size,projection_size])
        with tf.variable_scope("Lstmp_peep_weight"+layer,initializer=tf.random_normal_initializer()):
            self.Ip=tf.get_variable("Ip",dtype=tf.float64,shape=[cell_size,1])
            self.Fp=tf.get_variable("Fp",dtype=tf.float64,shape=[cell_size,1])
            self.Op=tf.get_variable("Op",dtype=tf.float64,shape=[cell_size,1])
        with tf.variable_scope("Lstmp_bias_weight"+layer,initializer=self.bias_initializer):
            self.Fb=tf.get_variable("Fb",dtype=tf.float64,shape=[cell_size,1],initializer=tf.constant_initializer(forget_bias))
            self.Ib=tf.get_variable("Ib",dtype=tf.float64,shape=[cell_size,1])
            self.Ob=tf.get_variable("Ob",dtype=tf.float64,shape=[cell_size,1])
            self.Gb=tf.get_variable("Gb",dtype=tf.float64,shape=[cell_size,1])
        with tf.variable_scope("Lstmp_projection_weight"+layer,initializer=self.weight_initializer):
            self.Pro=tf.get_variable("Pro",dtype=tf.float64,shape=[projection_size,cell_size])
    
    def __call__(self,input,prev_states):
        P,C=prev_states
        assert input.shape[0]==self.feature_size
        assert P.shape[0]==self.projection_size and C.shape[0]==self.cell_size
        with tf.name_scope("input_gate"):
            I=tf.sigmoid(tf.matmul(self.Ix,input)+tf.matmul(self.Ih,P)+self.peep*self.Ip*C+self.Ib)
        with tf.name_scope("forget_gate"):
            F=tf.sigmoid(tf.matmul(self.Fx,input)+tf.matmul(self.Fh,P)+self.peep*self.Fp*C+self.Fb)
        with tf.name_scope("gate"):
            G=tf.tanh(tf.matmul(self.Gx,input)+tf.matmul(self.Gh,P)+self.Gb)
        with tf.name_scope("next_cell_state"):
            C=I*G+F*C
        with tf.name_scope("output_gate"):
            O=tf.sigmoid(tf.matmul(self.Ox,input)+tf.matmul(self.Oh,P)+self.peep*self.Op*C+self.Ob)
        with tf.name_scope("next_hidden_state"):
            H=O*tf.tanh(C)
        with tf.name_scope("projected_hidden_state"):
            out=P=tf.matmul(self.Pro,H)
        return (out,(P,C))

        
class DLstmModel(object):
    def __init__(self,depth,sequence_len,feature_size,cell_size,output_size,x_weight_initializer,hidden_weight_initializer,\
    bias_initializer,forget_bias=1,peep=False):
        assert depth>0 and isinstance(depth,int)
        assert sequence_len>0 and isinstance(sequence_len,int)
        assert feature_size>0 and isinstance(feature_size,int)
        assert isinstance(cell_size,(tuple,list)),"cell_size should specify each layers' cell size"
        assert len(cell_size)==depth,"cell_size and depth should match"
        assert output_size>0 and isinstance(output_size,int)
        for c in cell_size:
            assert c>0
        self.feature_size=feature_size
        self.cell_size=cell_size
        self.depth=depth
        self.sequence_len=sequence_len
        self.output_size=output_size
        self.X=tf.placeholder(dtype=tf.float64,shape=[sequence_len,feature_size,None],name="input_x")
        self.Y=tf.placeholder(dtype=tf.float64,shape=[None],name="input_y")
        self.cells={}
        self.states={}
        for i in range(1,depth+1):
            self.states["layer"+str(i)]=(tf.placeholder(dtype=tf.float64,shape=[cell_size[i-1],None],\
            name="hidden_state"+str(i)),tf.placeholder(dtype=tf.float64,shape=[cell_size[i-1],None],name="cell_state"+str(i)))
            if i==1:
                self.cells["layer"+str(i)]=LstmCell(feature_size,cell_size[i-1],\
                x_weight_initializer,hidden_weight_initializer,bias_initializer,forget_bias,peep,"layer"+str(i))
            else:
                self.cells["layer"+str(i)]=LstmCell(cell_size[i-2],cell_size[i-1],\
                x_weight_initializer,hidden_weight_initializer,bias_initializer,forget_bias,peep,"layer"+str(i))
        self.name="dlstm"
    
    def BuildGraph(self,look_back,pooling="average"):
        assert look_back<=self.sequence_len and look_back>=1
        assert pooling=="average" or "weighted"
        L=self.X.shape[0]
        out_collection=[]
        state_collection=dict(self.states)
        for t in range(L):
            with tf.name_scope("time_step"+str(t)):
                link=[]#for passing states between each layer
                for l in range(1,self.depth+1):
                    with tf.name_scope("layer_"+str(l)):
                        if l==1:
                            state=state_collection["layer"+str(l)]
                            out,state=self.cells["layer"+str(l)](self.X[t,:,:],state)
                            if self.depth==1:
                                out_collection.append(out)
                            else:
                                link.append(out)
                            state_collection["layer"+str(l)]=state
                        else:
                            state=state_collection["layer"+str(l)]
                            out,state=self.cells["layer"+str(l)](link[-1],state)
                            if l == self.depth:
                                out_collection.append(out)
                            else:
                                link.append(out)
                            state_collection["layer"+str(l)]=state
        
        if pooling == "average":
            with tf.name_scope("average_pooling_layer"):
                y_hat=sum(out_collection[-1:(-look_back)-1:-1])/look_back
                weight=tf.get_variable(dtype=tf.float64,shape=[self.output_size,self.cell_size[-1]],\
                name="look_back_weight")
                bias=tf.get_variable(dtype=tf.float64,shape=[self.output_size,1],initializer=tf.zeros_initializer(),\
                name="look_back_bias")
                y_hat=tf.matmul(weight,y_hat)+bias
        
        elif pooling == "weighted":
            with tf.name_scope("weighte_pooling_layer"):
                y_hat=tf.zeros([self.output_size,self.cell_size[-1]])
                for i in range(1,look_back+1):
                    weight=tf.get_variable(dtype=tf.float64,shape=[self.output_size,self.cell_size[-1]],\
                    name="look_back_weight"+str(i))
                    y_hat+=tf.matmul(weight,out_collection[-i])
                bias=tf.get_variable(dtype=tf.float64,shape=[self.output_size],initializer=tf.zeros_initializer(),\
                name="look_back_bias")
                y_hat+=bias
            
        else:
            raise ValueError
        
        return y_hat

    def compute_cost(self,y,loss_function,task):
        assert y.shape[0]==self.output_size
        if task=="classification":
            loss=loss_function(tf.one_hot(tf.cast(self.Y,tf.int64),depth=self.output_size,axis=0),logits=y)
            mean_loss=tf.reduce_mean(loss)
        elif task=="regression":
            mean_loss=loss_function(self.Y,tf.squeeze(y))
        else:
            raise ValueError
        return mean_loss
        
    def objective(self,loss,optimizer,global_step=None):
        '''compute the training objective you want e.g min{loss}'''
        if global_step:                  
            obj=optimizer.minimize(loss,global_step)
        else:
            obj=optimizer.minimize(loss)
        return obj
        
    def Train(self,sess,task,Xd,Yd,y_hat,loss_val,train_object,epoch=1,batch_size=64,print_every=50):
        '''sess should be the tensor session you running, Xd is training set, Yd is true data, loss_val is the loss you defined, y_hat is the p
        y, train_object is the objective (minimize the loss), epoch is how many times you want to run on full training set, batch_size is how m
        in each mini batch, print every is frequency of output result.'''
        train_indicies=np.arange(Xd.shape[2])
        
        if task=="classification":
            correct_prediction = tf.equal(tf.argmax(y_hat,0), tf.cast(self.Y,tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            variables=[loss_val,correct_prediction,train_object]
        elif task=="regression":
            variables=[loss_val,train_object]
        else:
            raise ValueError
            
        iter_ct=0
        for e in range(epoch):
            correct=0
            losses=[]
            corr=[]
            for i in range(int(math.ceil(Xd.shape[2]/batch_size))):
                #generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[2]
                idx = train_indicies[start_idx:start_idx+batch_size]
                actual_batch_size = len(idx)
            
                feed_dict={}
                for l in range(self.depth):
                    feed_dict[self.states["layer"+str(l+1)]]=(np.zeros((self.cell_size[l],actual_batch_size)),\
                    np.zeros((self.cell_size[l],actual_batch_size)))
                feed_dict[self.X]=Xd[:,:,idx]
                feed_dict[self.Y]=Yd[idx]
                
                if task=="classification":
                    loss,corr,_=sess.run(variables,feed_dict=feed_dict)
                    correct+=np.sum(corr)                     
                else:
                    loss,_=sess.run(variables,feed_dict=feed_dict)
                    
                losses.append(loss*actual_batch_size)
                if  iter_ct % print_every == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g}".format(iter_ct,loss))
                iter_ct+=1
            total_loss = np.sum(losses)/Xd.shape[2]
            if task=="classification":
                total_correction=correct/Xd.shape[2]
                print("Epoch {1}, Overall loss = {0:.3g} and accuracy = {2:.3g}".format(total_loss,e+1,total_correction))
            else:
                print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss,e+1))
             
    def Predict(self,sess,Xd,y_hat):
        N=Xd.shape[-1]
        batch_size=64
        train_indicies=np.arange(N)
        prediction=np.empty(shape=[self.output_size,N],dtype=np.float64)
        for i in range(int(math.ceil(N/batch_size))):
            start_idx = (i*batch_size)%Xd.shape[2]
            idx=train_indicies[start_idx:start_idx+batch_size]
            actual_batch_size=len(idx)
            feed_dict={}
            for l in range(self.depth):
                feed_dict[self.states["layer"+str(l+1)]]=(np.zeros((self.cell_size[l],actual_batch_size)),\
                np.zeros((self.cell_size[l],actual_batch_size)))
            feed_dict[self.X]=Xd[:,:,idx]            
            prediction[:,idx]=sess.run(y_hat,feed_dict=feed_dict)
        return prediction

class GruModel(DLstmModel):
    def __init__(self,depth,sequence_len,feature_size,cell_size,output_size,x_weight_initializer,hidden_weight_initializer,\
    bias_initializer):
        assert depth>0 and isinstance(depth,int)
        assert sequence_len>0 and isinstance(sequence_len,int)
        assert feature_size>0 and isinstance(feature_size,int)
        assert isinstance(cell_size,(tuple,list)),"cell_size should specify each layers' cell size"
        assert len(cell_size)==depth,"cell_size and depth should match"
        assert output_size>0 and isinstance(output_size,int)
        for c in cell_size:
            assert c>0   
        self.feature_size=feature_size
        self.depth=depth
        self.sequence_len=sequence_len
        self.cell_size=cell_size
        self.output_size=output_size
        self.X=tf.placeholder(dtype=tf.float64,shape=[sequence_len,feature_size,None],name="input_x")
        self.Y=tf.placeholder(dtype=tf.float64,shape=[None],name="input_y")
        self.cells={}
        self.states={}
        for i in range(1,depth+1):
            self.states["layer"+str(i)]=tf.placeholder(dtype=tf.float64,shape=[cell_size[i-1],None],\
            name="hidden_state"+str(i))
            if i==1:
                self.cells["layer"+str(i)]=GRU(feature_size,cell_size[i-1],\
                x_weight_initializer,hidden_weight_initializer,bias_initializer,"layer"+str(i))
            else:
                self.cells["layer"+str(i)]=GRU(cell_size[i-2],cell_size[i-1],\
                x_weight_initializer,hidden_weight_initializer,bias_initializer,"layer"+str(i))
        self.name="gru"
    
    def Train(self,sess,task,Xd,Yd,y_hat,loss_val,train_object,epoch=1,batch_size=64,print_every=50):
        '''sess should be the tensor session you running, Xd is training set, Yd is true data, loss_val is the loss you defined, y_hat is the p
        y, train_object is the objective (minimize the loss), epoch is how many times you want to run on full training set, batch_size is how m
        in each mini batch, print every is frequency of output result.'''
        train_indicies=np.arange(Xd.shape[2])
        
        if task=="classification":
            correct_prediction = tf.equal(tf.argmax(y_hat,0), tf.cast(self.Y,tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            variables=[loss_val,correct_prediction,train_object]
        elif task=="regression":
            variables=[loss_val,train_object]
        else:
            raise ValueError
            
        iter_ct=0
        for e in range(epoch):
            correct=0
            losses=[]
            corr=[]
            for i in range(int(math.ceil(Xd.shape[2]/batch_size))):
                #generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[2]
                idx = train_indicies[start_idx:start_idx+batch_size]
                actual_batch_size = len(idx)
            
                feed_dict={}
                for l in range(self.depth):
                    feed_dict[self.states["layer"+str(l+1)]]=np.zeros((self.cell_size[l],actual_batch_size))
                feed_dict[self.X]=Xd[:,:,idx]
                feed_dict[self.Y]=Yd[idx]
                
                if task=="classification":
                    loss,corr,_=sess.run(variables,feed_dict=feed_dict)
                    correct+=np.sum(corr)                     
                else:
                    loss,_=sess.run(variables,feed_dict=feed_dict)
                    
                losses.append(loss*actual_batch_size)
                if  iter_ct % print_every == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g}".format(iter_ct,loss))
                iter_ct+=1
            total_loss = np.sum(losses)/Xd.shape[2]
            if task=="classification":
                total_correction=correct/Xd.shape[2]
                print("Epoch {1}, Overall loss = {0:.3g} and accuracy = {2:.3g}".format(total_loss,e+1,total_correction))
            else:
                print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss,e+1))   
    
    def Predict(self,sess,Xd,y_hat):
        N=Xd.shape[-1]
        batch_size=64
        train_indicies=np.arange(N)
        prediction=np.empty(shape=[self.output_size,N],dtype=np.float64)
        for i in range(int(math.ceil(N/batch_size))):
            start_idx = (i*batch_size)%Xd.shape[2]
            idx=train_indicies[start_idx:start_idx+batch_size]
            actual_batch_size=len(idx)
            feed_dict={}
            for l in range(self.depth):
                feed_dict[self.states["layer"+str(l+1)]]=np.zeros((self.cell_size[l],actual_batch_size))
            feed_dict[self.X]=Xd[:,:,idx]            
            prediction[:,idx]=sess.run(y_hat,feed_dict=feed_dict)
        return prediction        
 
    
    
    
'''
@note: Official implementation of ELTRA

@author: masoud
'''
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as tkb
from argparse import ArgumentParser
from AP_scores import AP_Scores_
from Weighted_DCG_ListMLE_topK import Weighted_DCG_ListMLE_topK
import math


class CustomCallback_verbose_check(tf.keras.callbacks.Callback):
    '''
        1- Representing custom verbose informtation
        2- checking loss
    '''
    def on_epoch_end(self, epoch, logs=None):
        ## checking loss
        print("Epoch: {} .... lr: {}; loss: {}".format(epoch+1,round(float(tkb.get_value(self.model.optimizer.learning_rate)),5),round(logs['loss'],3)))    
    def on_train_end(self, logs=None):
        print("Stop training; .... Final loss: {}".format(round(logs['loss'],3)))
    
def get_model(dim,out_len, learning_rate, reg_rate):
    model = tf.keras.Sequential()
    model.add(layers.Dense(dim, activation='linear', input_shape=(out_len,), name='layer_0'))
    model.add(layers.Dense(out_len,activation='relu', kernel_regularizer=regularizers.L2(reg_rate), bias_regularizer=regularizers.L2(reg_rate), name='layer_1'))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate), loss = Weighted_DCG_ListMLE_topK(), run_eagerly = True)
    return model

def ELTRA(args):
    print()
    if not os.path.exists(args.graph):
        print('ERROR: graph is invalid ...!')
        return
    if args.dataset_name=='':
        print('ERROR: dataset name is invalid ...!')
        return
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## disabling GPU

    print('=================================================================== Similarity Computation =================================================================')
    if args.graph_type == 'directed': 
        directed_gr = True
        damping_fact = 0.6
    else:
        directed_gr = False 
        damping_fact = 0.4       
    top_indices,dcg_vals = AP_Scores_(graph=args.graph, iterations=args.itr, damping_factor=damping_fact, impr_fact=args.impr, topK=args.topk, directed_graph=directed_gr,loss='weighted_listMLE_topK')
    
    print('=========================================================================== ARGUMENTS ==========================================================================')
    if args.bch==-1: # calculating batch size for the input graph
        args.bch = pow(2, round(math.log2(len(top_indices)*0.05)))  
        if args.bch<1:
            print('*** Computed bch values ({}) is invalid; probably the graph is very small! set the bch value manually ***'.format(args.bch))
            print('')
            return
        print('The batch size is computed and set as {}'.format(args.bch),'\n')
    print(args,'\n')
    info = args.result_dir+args.dataset_name+'_ELTRA_Imf_'+str(args.impr*10).split('.')[0]+'_IT'+str(args.itr)+'_Reg'+str(args.reg).split('.')[1]+'_dim'+str(args.dim)+'_bch'+str(args.bch)+'_Top'+str(args.topk)

    print('======================================================================== Model Training ========================================================================')                    
    tf_input = tf.eye(len(top_indices), dtype='int32') # one-hot vecors as input
    model = get_model(int(args.dim), len(top_indices), args.lr, args.reg)
    if args.early_stop: ## apply early stopping
        callback_EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.wait_thr, mode='min', restore_best_weights=True) ## defines a callback for early stop
        model.fit(x=tf_input,y=tf.concat([top_indices,dcg_vals],axis=1), epochs=args.epc, batch_size=args.bch,callbacks=[CustomCallback_verbose_check(),callback_EarlyStopping],verbose=0)
    else:
        model.fit(x=tf_input,y=tf.concat([top_indices,dcg_vals],axis=1), epochs=args.epc, batch_size=args.bch,callbacks=[CustomCallback_verbose_check()],verbose=0)
        
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)        
        
    emb_0 = model.get_layer('layer_0').weights[0][:].numpy()
    emb_1 = model.get_layer('layer_1').weights[0][:].numpy().T
    emb_file_SR = open(info+'_source.emb','w')
    emb_file_TR = open(info+'_target.emb','w')
    emb_file_SR.write(str(len(emb_0))+'\t'+str(args.dim)+'\n')
    emb_file_TR.write(str(len(emb_0))+'\t'+str(args.dim)+'\n')    
    for row in range(0,len(emb_0)):
        emb_file_SR.write(str(row))
        emb_file_TR.write(str(row))
        emb_val_SR = ''
        emb_val_TR = ''
        for col in range(0, int(args.dim)):
            emb_val_SR = emb_val_SR + '\t' + str(emb_0[row][col])
        for col in range(0, int(args.dim)):
            emb_val_TR = emb_val_TR + '\t' + str(emb_1[row][col])            
        emb_file_SR.write(emb_val_SR+'\n')  
        emb_file_TR.write(emb_val_TR+'\n')  
    emb_file_SR.close()
    emb_file_TR.close()    
    print('The embedding results are written in source and target files ....\n\n')
    

def parse_args(graph='',dataset_name='',result_dir='result_test/', dimension=128, topK=-1, iterations=3, epochs=150, batch_size=-1, learning_rate=0.0025, 
               reg_rate=0.001, early_stop=True, wait_thr=20, gpu_on=True, graph_type='directed', impr_fact = 0.6):
    parser = ArgumentParser(description="Run ELTRA, a double-vector similarity-based embedding method.")
    parser.add_argument('--graph', nargs='?', default=graph, help='Input graph')       
    parser.add_argument('--dataset_name', nargs='?', default=dataset_name, help='Dataset name')   
    parser.add_argument('--result_dir', nargs='?', default=result_dir, help='Destination to save the source and target embedding')   
    parser.add_argument('--dim', type=int, default=dimension, help='Dimensionality of embedding')  
    parser.add_argument('--topk', type=int, default=topK, help='Number of nodes in topK; default is -1 (i.e., topK is automatically computed as explained in the paper), otherwise input the desired value')    
    parser.add_argument('--itr', type=int, default=iterations, help='Number of Iterations to compute AP-Scores, default is 3')
    parser.add_argument('--epc', type=int, default=epochs, help='Number of epochs for training, default is 150')
    parser.add_argument('--bch', type=int, default=batch_size, help='Number of instances in a batch, default is -1 (i.e., it is automatically computed as explained in the paper), otherwise input the desired value')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate, default is 0.0025')
    parser.add_argument('--reg', type=float, default=reg_rate, help='Regularization parameter which is highly suggested to be 0.001 and 0.0001 with directed and undirected graphs, respectively; default is 0.001')
    parser.add_argument('--early_stop', type=bool, default=early_stop, help='The flag indicating to stop the training process if the loss stops improving')
    parser.add_argument('--wait_thr', type=int, default=wait_thr, help='Number of epochs with no loss improvement after which training will be stopped, default is 20')
    parser.add_argument('--gpu', type=bool, default=gpu_on, help='The flag to run computation on GPU')
    parser.add_argument('--graph_type', default=graph_type, help='Indicates the graph type, default is directed')
    parser.add_argument('--impr', default=impr_fact, help='Importance factor of out-links over in-links to compute AP-Scores, default is 0.6')

    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    ELTRA(args)

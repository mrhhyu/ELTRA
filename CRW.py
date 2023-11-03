'''
Created on March 21, 2023
Implements Matrix form of CRW 
@author: masoud
'''
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import fetch_topK_DENSE as prData
from scipy.sparse import identity 
    
def CRW (graph='', iterations=0, damping_factor=0.8):
    '''
        Computes CRW scores           
    '''
    print("Starting CRW computation with '{}' on '{}' itrations, and C '{}'...".format(graph,iterations,damping_factor)+'\n')    

    #============================================================================================
        # reading graph with networkX
    #============================================================================================    
    G = nx.read_edgelist(graph, create_using=nx.DiGraph(), nodetype = int)
    nodes = sorted(G.nodes())
    adj = nx.adjacency_matrix(G,nodelist=nodes, weight=None)
    adj_normalized_inlink = normalize(adj, norm='l1', axis=0).T # column normalized adj    
    adj_normalized_outlink = normalize(adj, norm='l1', axis=1) # row normalized adj  
    result_matrix = identity(len(nodes),dtype=float)
    result_matrix = result_matrix.todense()
    for itr in range (1, iterations+1):           
        print("Iteration "+str(itr)+' ....')
        result_matrix =  damping_factor/2.0 * ( (adj_normalized_inlink * result_matrix) + ( adj_normalized_outlink * result_matrix) ) 
        np.fill_diagonal(result_matrix,1)
        print (result_matrix)     ## you can write down the result_matrix in a file or process it here            



#'''
CRW(graph="/home/masoud/backup_1/workspace/iterative-adamic/src/graphs/AdaSimStar.txt", 
           iterations=3, 
           damping_factor =0.6,
           ) 


#'''           


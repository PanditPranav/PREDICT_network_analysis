"""Running basic code:
Importing packages, setting working directory, 
printing out date"""

import os as os
os.chdir('C:/Users/Falco/Desktop/directory/Missing_links_in_viral_host_communities/')
import datetime as dt
str(dt.datetime.now())
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from pandas_ml import ConfusionMatrix
data_path = 'C:/Users/Falco/Desktop/directory/Missing_links_in_viral_host_communities/data'
output_path = 'C:/Users/Falco/Desktop/directory/Missing_links_in_viral_host_communities/outputs'
from HPnex import functions as f
from HPnex import classification as classify
from HPnex import fitting_functions as fitt

import numpy as np
import networkx as nx
#np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
#from pandas_ml import ConfusionMatrix
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import model_selection
import math
height = 6
font = 12

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.cross_validation import
from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import learning_curve
#from pandas_ml import ConfusionMatrix
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#### Standardize continuous variables
from sklearn.preprocessing import StandardScaler
from sklearn  import preprocessing
#from pandas_ml import ConfusionMatrix
from HPnex import functions as f
### Running cross validation scores and predictions
from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.style as style
style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_context("notebook", font_scale=1.30, rc={"lines.linewidth": 0.8})
import itertools as itertools
import pandas as pd
import joblib

###############################################################################################################################
###############################################################################################################################


def generete_temp_network(virus, hosts, ViralFamily, PubMed, BPnx, Gc_complete,IUCN, virus_df):
    #print('this generete_temp_network function is in predict_multi_file')
    temp_BPnx = BPnx.copy() ## making a copy of original Bipartite network
    #print (temp_BPnx.number_of_nodes()) ## checking number of nodes
    #virus_nodes = [x for x,y in temp_BPnx.nodes(data=True) if y['type']=='virus'] #creating list of virus nodes from bipartite network
    BPNX_dd = pd.DataFrame.from_dict(dict(temp_BPnx.nodes(data=True)), orient='index')
    virus_nodes = list(BPNX_dd[BPNX_dd['type'] == 'virus'].index)
    df = pd.DataFrame({'Virus2':virus_nodes}) # converting them to a dataframe
    df['Virus1'] = virus # dataframe with all possible combinations of new virus and viruses from BPnx
    temp_BPnx.add_node(virus, virusname=virus, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    #print (temp_BPnx.number_of_nodes()) ## rechecking number of nodes
    for h in hosts:
        if len(hosts)>1:
            print (h)
        temp_BPnx.add_edge(virus, h) ## adding new edge to the Bpnxtemp
    def get_n_shared_hosts(c): ## calculating number of neighbours for our new virus
            return len(list(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2'])))
    df['n_shared_hosts'] = df.apply(get_n_shared_hosts, axis=1)
    #print('New edges in unipartite network '+ str(df['n_shared_hosts'].sum()))
    n_e = df['n_shared_hosts'].sum()
    def addsharedhosts (c): ## identifiying number of neighbours for our new virus
            return  sorted(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2']))
    df["shared_hosts"] = df.apply(addsharedhosts, axis=1)
    
    def add_hosts_orders (c):
        order_list = IUCN[IUCN.ScientificName.isin(c['shared_hosts'])]['Order'].unique().tolist()
        return order_list
    df["shared_orders"] = df.apply(add_hosts_orders, axis=1)
    
    new_edges = df[df['n_shared_hosts']>0] ### list of new edges for new viruses
    #print(new_edges.shape)
    Gc_complete_temp  = Gc_complete.copy() ## creating a temporary copy of GC complete
    Gc_complete_temp.add_node(virus, ViralFamily=ViralFamily, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    
    for index, row in new_edges.iterrows():
            if row['n_shared_hosts'] > 0:
                Gc_complete_temp.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']), orders = ','.join(row['shared_orders']))
                
    edges_to_predict = df[df['n_shared_hosts']==0]
    edges_to_predict = edges_to_predict[edges_to_predict.Virus2 != 'nan']
    virus_df_temp = virus_df.copy()
    virus_df_temp.loc[len(virus_df_temp)]=[virus, ViralFamily, math.log(PubMed),1, 1] 
    return Gc_complete_temp, edges_to_predict, virus_df_temp, df, n_e


###############################################################################################################################
###############################################################################################################################


def generete_temp_network_known(virus, hosts, ViralFamily, PubMed, BPnx, Gc_complete,IUCN, virus_df):
    #print('this generete_temp_network function is in predict_multi_file')
    temp_BPnx = BPnx.copy() ## making a copy of original Bipartite network
    temp_BPnx.remove_node(virus)
    #print (temp_BPnx.number_of_nodes()) ## checking number of nodes
    #virus_nodes = [x for x,y in temp_BPnx.nodes(data=True) if y['type']=='virus'] #creating list of virus nodes from bipartite network
    BPNX_dd = pd.DataFrame.from_dict(dict(temp_BPnx.nodes(data=True)), orient='index')
    virus_nodes = list(BPNX_dd[BPNX_dd['type'] == 'virus'].index)
    df = pd.DataFrame({'Virus2':virus_nodes}) # converting them to a dataframe
    df['Virus1'] = virus # dataframe with all possible combinations of new virus and viruses from BPnx
    temp_BPnx.add_node(virus, virusname=virus, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    #print (temp_BPnx.number_of_nodes()) ## rechecking number of nodes
    for h in hosts:
        if len(hosts)>1:
            print (h)
        temp_BPnx.add_edge(virus, h) ## adding new edge to the Bpnxtemp
    def get_n_shared_hosts(c): ## calculating number of neighbours for our new virus
            return len(list(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2'])))
    df['n_shared_hosts'] = df.apply(get_n_shared_hosts, axis=1)
    #print('New edges in unipartite network '+ str(df['n_shared_hosts'].sum()))
    n_e = df['n_shared_hosts'].sum()
    def addsharedhosts (c): ## identifiying number of neighbours for our new virus
            return  sorted(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2']))
    df["shared_hosts"] = df.apply(addsharedhosts, axis=1)
    
    def add_hosts_orders (c):
        order_list = IUCN[IUCN.ScientificName.isin(c['shared_hosts'])]['Order'].unique().tolist()
        return order_list
    df["shared_orders"] = df.apply(add_hosts_orders, axis=1)
    
    new_edges = df[df['n_shared_hosts']>0] ### list of new edges for new viruses
    #print(new_edges.shape)
    Gc_complete_temp  = Gc_complete.copy() ## creating a temporary copy of GC complete
    Gc_complete_temp.add_node(virus, ViralFamily=ViralFamily, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    
    for index, row in new_edges.iterrows():
            if row['n_shared_hosts'] > 0:
                Gc_complete_temp.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']), orders = ','.join(row['shared_orders']))
                
    edges_to_predict = df[df['n_shared_hosts']==0]
    edges_to_predict = edges_to_predict[edges_to_predict.Virus2 != 'nan']
    virus_df_temp = virus_df.copy()
    virus_df_temp.loc[len(virus_df_temp)]=[virus, ViralFamily, math.log(PubMed),1, 1] 
    return Gc_complete_temp, edges_to_predict, virus_df_temp, df, n_e




###############################################################################################################################
###############################################################################################################################

def preprocessing_x(data_frame, network, virus_df_temp, virus_df):

    #predictors = [
    #   'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
    #    'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2', 'neighbors_n', 
    #    'adamic_adar', 'resource', 'preferential_attach'
    #]
    #print('this function is in predict_multi file')
    predictors = [
       'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2',
        'VirusCluster1', 'VirusCluster2', 'resource', 'preferential_attach'
    ]

    predictors = [
       'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2',
       
    ]

    ################################################################################################################################
    ################################################################################################################################
    def jaccard (c):
        return sorted(nx.jaccard_coefficient(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["jaccard"] = data_frame.apply(jaccard, axis=1)
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating difference in betweenness centrality')
    btw = nx.betweenness_centrality(network, 25)
    def betweenDiff(c):
        return abs(btw[c['Virus1']] - btw[c['Virus2']])
    data_frame["betweeness_diff"] = data_frame.apply(betweenDiff, axis=1) 
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating node clusters')
    #import community
    #partition = community.best_partition(network)
    #import community
    from community import community_louvain
    partition = community_louvain.best_partition(network)
    ################################################################################################################################
    ################################################################################################################################
    
    def virus1_cluster(c):
        return partition[c['Virus1']]
                         
    data_frame['VirusCluster1'] = data_frame.apply(virus1_cluster, axis=1) 
                         
    def virus2_cluster(c):
        return partition[c['Virus2']]
                         
    data_frame['VirusCluster2'] = data_frame.apply(virus2_cluster, axis=1) 
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating is nodes are in a same cluster')
    
    def in_same_cluster(c):
        if(partition[c['Virus1']] == partition[c['Virus2']]):
            return True
        else:
            return False
    data_frame["in_same_cluster"] = data_frame.apply(in_same_cluster, axis=1)
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating difference in degree')
    degree = nx.degree(network)
    def degreeDiff(c):
        return abs(degree[c['Virus1']] - degree[c['Virus2']])
    data_frame["degree_diff"] = data_frame.apply(degreeDiff, axis=1) 
    
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating adamic/adar index')
    #def adar (c):
    #    return sorted(nx.adamic_adar_index(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    #data_frame["adamic_adar"] = data_frame.apply(adar, axis=1)
        
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating Resource coefficients')
    def resource (c):
        return sorted(nx.resource_allocation_index(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["resource"] = data_frame.apply(resource, axis=1)
        
    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating preferential attachment coefficients')
    def preferential  (c):
        return sorted(nx.preferential_attachment(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["preferential_attach"] = data_frame.apply(preferential, axis=1)
    
    ################################################################################################################################
    ################################################################################################################################
    def hasShortestPath (c):
            return nx.has_path(network, c['Virus1'], c['Virus2'])

    data_frame["hasPath"] = data_frame.apply(hasShortestPath, axis=1)
    data_frame['hasPath'] = np.where(data_frame['hasPath']== True, 1, 0)
    
    ################################################################################################################################
    ################################################################################################################################
    #print ('listing neighbors')
    def neighbors  (c):
        l = sorted(nx.common_neighbors(network, c['Virus1'],c['Virus2']))
        return str(l)[1:-1]
    data_frame["neighbors"] = data_frame.apply(neighbors, axis=1)

    ################################################################################################################################
    ################################################################################################################################
    #print ('calculating number of neighbors')
    def neighbors_n  (c):
        return len(sorted(nx.common_neighbors(network, c['Virus1'],c['Virus2'])))
    data_frame["neighbors_n"] = data_frame.apply(neighbors_n, axis=1)
    
    ################################################################################################################################
    ################################################################################################################################
    

    virus_df_temp = virus_df_temp.sort_values(['virus_name', 'PubMed_Search_ln'])
    virus_df_temp = virus_df_temp.drop_duplicates(subset = 'virus_name', keep='last')
    data_frame = pd.merge(data_frame,virus_df_temp[['virus_name','viral_family','PubMed_Search_ln']], left_on='Virus1', right_on='virus_name', how='left')
    data_frame = pd.merge(data_frame,virus_df_temp[['virus_name','viral_family', 'PubMed_Search_ln']], left_on='Virus2', right_on='virus_name', how='left')
    data_frame['ViralFamily1'] = data_frame['viral_family_x']
    data_frame['ViralFamily2'] = data_frame['viral_family_y']

    data_frame['PubMed_Search_ln1'] = data_frame['PubMed_Search_ln_x']
    data_frame['PubMed_Search_ln2'] = data_frame['PubMed_Search_ln_y']

    del data_frame['viral_family_y']
    del data_frame['viral_family_x']
    del data_frame['PubMed_Search_ln_x']
    del data_frame['PubMed_Search_ln_y']
    del data_frame['virus_name_x']
    del data_frame['virus_name_y']
    def MatchFamily(c):
        if c.ViralFamily1 == c.ViralFamily2:
            return 'True'
        else:
            return 'False'
    data_frame['FamilyMatch'] = data_frame.apply(MatchFamily, axis=1)

    #print ('difference in PubMed hits')
    def PubMed_hits(c):
        return abs(c.PubMed_Search_ln1 - c.PubMed_Search_ln2)
    data_frame['PubMed_diff'] = data_frame.apply(PubMed_hits, axis=1)
    
    data_frame['in_same_cluster'] =np.where(data_frame['in_same_cluster']== True, 1, 0)
    data_frame['FamilyMatch'] =np.where(data_frame['FamilyMatch']== 'True', 1, 0)

    #print ('difference in PubMed hits')
    X = data_frame[list(predictors)].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    data_processed = pd.DataFrame(X_std, columns=predictors)

    ### Encoding categorical variables
    le = preprocessing.LabelEncoder()
    le.fit(virus_df.viral_family.unique())
    data_frame['F1'] = le.transform(data_frame.ViralFamily1.fillna('Not_Assinged'))
    data_frame['F2'] = le.transform(data_frame.ViralFamily2.fillna('Not_Assinged'))
    data_processed['F1'] = data_frame.F1
    data_processed['F2'] = data_frame.F2
    data_processed.fillna(0, inplace=True)
    #print (data_processed.shape)

    return data_processed

###############################################################################################################################
###############################################################################################################################

def prediction(temp_x, clf_multi, inv_dictionary, plot = False):
    #inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.iteritems())
    inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.items())
    prediction = pd.DataFrame(clf_multi.predict(temp_x)).replace(inv_dictionary)
    temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
    #temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
    probs = clf_multi.predict_proba(temp_x)
    prob_max =[] 
    for i in range (len(probs)):
        prob_max.append(np.amax(probs[i], axis = 1))
    max_prob = pd.DataFrame(prob_max).T
    prediction = prediction.join(max_prob, lsuffix='_pr', rsuffix='_prob')
    if plot:
        fig, (ax1) = plt.subplots(figsize = [6,6])
        a = prediction.copy()
        sns.boxenplot(x = '0_prob',y ='0_pr', data=a, ax= ax1)
        sns.stripplot(x="0_prob", y="0_pr", data=a, jitter= True, color='#252525', ax= ax1)
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Predicted Order')
        ax1.set_title('The type of host sharing between new and known viruses')
        fig.tight_layout()
        plt.show()
        summary = a.groupby('0_pr').agg({'0_prob':['mean', 'std', 'count']})
        summary.columns = ['mean probability', 'std', 'number of links']
        summary.index.rename('Link Type', inplace= True)
        print(summary)
    return prediction

###############################################################################################################################
###############################################################################################################################

def run_predictions(virus, hosts, PubMed, ViralFamily, BPnx, Gc_complete, virus_df,
                    clf_multi, inv_dictionary, IUCN, plot = False):
    
    ## creates a new network by adding the new virus and generating edges for the newly found virus based on the host. Used BPnx for it
    ## to predict is is the dataframe for for observed edges, will be used for comparison /validation during corss validation in cases for known viruses
    
    temp_Gc, to_predict, virus_df_temp, new_netowrk_df, new_edges_temp = generete_temp_network(virus = virus, 
                                                               hosts =hosts,
                                                               PubMed = PubMed,
                                                               ViralFamily = ViralFamily, 
                                                                               IUCN = IUCN,
                                                               BPnx = BPnx, Gc_complete = Gc_complete, virus_df = virus_df)
    
    ##preprocesses the new virus data to create a 
    temp_x = preprocessing_x(data_frame = to_predict,
                             network = temp_Gc,
                             virus_df_temp = virus_df_temp, 
                             virus_df = virus_df
                             )

    result = prediction(temp_x= temp_x,
                        clf_multi= clf_multi,
                        inv_dictionary= inv_dictionary, plot= plot)
    
    result = result.join(to_predict)
    return result, new_edges_temp #, to_predict



###############################################################################################################################
###############################################################################################################################

def run_predictions_known(virus, hosts, PubMed, ViralFamily, BPnx, Gc_complete, virus_df,
                    clf_multi, inv_dictionary, IUCN, plot = False):
    
    ## creates a new network by adding the new virus and generating edges for the newly found virus based on the host. Used BPnx for it
    ## to predict is is the dataframe for for observed edges, will be used for comparison /validation during corss validation in cases for known viruses
    
    temp_Gc, to_predict, virus_df_temp, new_netowrk_df, new_edges_temp = generete_temp_network_known(virus = virus, 
                                                               hosts =hosts,
                                                               PubMed = PubMed,
                                                               ViralFamily = ViralFamily, 
                                                                               IUCN = IUCN,
                                                               BPnx = BPnx, Gc_complete = Gc_complete, virus_df = virus_df)
    
    ##preprocesses the new virus data to create a 
    temp_x = preprocessing_x(data_frame = to_predict,
                             network = temp_Gc,
                             virus_df_temp = virus_df_temp, 
                             virus_df = virus_df
                             )

    result = prediction(temp_x= temp_x,
                        clf_multi= clf_multi,
                        inv_dictionary= inv_dictionary, plot= plot)
    
    result = result.join(to_predict)
    return result, new_edges_temp #, to_predict




###############################################################################################################################
###############################################################################################################################

def prediction_validation_multiclass(temp_x, clf_multi, inv_dictionary, plot = False):
    #inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.iteritems())
    inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.items())
    Order_prediction = pd.DataFrame(clf_multi.predict(temp_x)).replace(inv_dictionary)
    #prediction = prediction.join(Order_prediction)
    temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
    #temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11','f12', 'f13']
    probs = clf_multi.predict_proba(temp_x)
    prob_max =[] 
    for i in range (len(probs)):
        prob_max.append(np.amax(probs[i], axis = 1))
    max_prob = pd.DataFrame(prob_max).T
    prediction = prediction.join(max_prob, lsuffix='_pr', rsuffix='_shakyata')
    return prediction

###############################################################################################################################
###############################################################################################################################
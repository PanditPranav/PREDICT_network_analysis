"""Running basic code:
Importing packages, setting working directory, 
printing out date"""

import os as os
os.chdir('C:/Users/falco/Desktop/directory/Missing_links_in_viral_host_communities/')
import datetime as dt
str(dt.datetime.now())
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from pandas_ml import ConfusionMatrix
data_path = 'C:/Users/falco/Desktop/directory/Missing_links_in_viral_host_communities/data'
output_path = 'C:/Users/falco/Desktop/directory/Missing_links_in_viral_host_communities/outputs'
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


def generete_temp_network(virus, hosts, ViralFamily, PubMed, BPnx_group, Gc,IUCN, virus_df):
    #print('this function is in multiclass validation file 1st function')
    import math
    temp_BPnx = BPnx_group.copy() 
    #print (temp_BPnx.number_of_nodes()) ## checking number of nodes
    virus_nodes = [x for x,y in temp_BPnx.nodes(data=True) if y['type']=='virus'] #creating list of virus nodes from bipartite network
    df = pd.DataFrame({'Virus2':virus_nodes}) # converting them to a dataframe
    df['Virus1'] = virus # dataframe with all possible combinations of new virus and viruses from BPnx
    temp_BPnx.add_node(virus, virusname=virus, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    #print (temp_BPnx.number_of_nodes()) ## rechecking number of nodes
    for h in hosts:
        temp_BPnx.add_edge(virus, h) ## adding new edge to the Bpnxtemp
    def get_n_shared_hosts(c): ## calculating number of neighbours for our new virus
            return len(list(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2'])))
    df['n_shared_hosts'] = df.apply(get_n_shared_hosts, axis=1)
    def addsharedhosts (c): ## identifiying number of neighbours for our new virus
            return  sorted(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2']))
    df["shared_hosts"] = df.apply(addsharedhosts, axis=1)
    
    def add_hosts_orders (c):
        order_list = IUCN[IUCN.ScientificName.isin(c['shared_hosts'])]['Order'].unique().tolist()
        return order_list
    df["shared_orders"] = df.apply(add_hosts_orders, axis=1)
    
    new_edges = df[df['n_shared_hosts']>0] ### list of new edges for new viruses
    #print(new_edges.shape)
    Gc_temp  = Gc.copy() ## creating a temporary copy of GC complete
    Gc_temp.add_node(virus, ViralFamily=ViralFamily, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    for index, row in new_edges.iterrows():
            if row['n_shared_hosts'] > 0:
                Gc_temp.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']), 
                                orders = ','.join(row['shared_orders']))
    #edges_to_predict = df[df['n_shared_hosts']==0]
    edges_to_predict = df
    edges_to_predict = edges_to_predict[edges_to_predict.Virus2 != 'nan']
    virus_df_temp = virus_df.copy()
    virus_df_temp.loc[len(virus_df_temp)]=[virus, ViralFamily, math.log(PubMed),1, 1]
    return Gc_temp, edges_to_predict, virus_df_temp

###############################################################################################################################
###############################################################################################################################

def prediction(temp_x, clf_multi, inv_dictionary):
    #print('this function is in multiclass validation file 1st function')
    inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.iteritems())
    Order_prediction = pd.DataFrame(clf_multi.predict(temp_x)).replace(inv_dictionary)
    temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
    #temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11','f12', 'f13']
    probs = clf_multi.predict_proba(temp_x)
    prob_max =[] 
    for i in range (len(probs)):
        prob_max.append(np.amax(probs[i], axis = 1))
    max_prob = pd.DataFrame(prob_max).T
    prediction = Order_prediction.join(max_prob, lsuffix='_pr', rsuffix='_shakyata')
    return prediction

###############################################################################################################################
###############################################################################################################################

def cross_validation_predict(virus, hosts, ViralFamily, PubMed, BPnx_group, Gc, virus_df, clf_multi, inv_dictionary):
    #print('this function is in multiclass validation file')
    from HPnex import predict_multi as pred_m
    Gc_temp_group, edges_to_predict, virus_df_temp = generete_temp_network(virus = virus, 
                                                    hosts = hosts,            
                                                    ViralFamily = ViralFamily,
                                                    PubMed = PubMed,
                                                    BPnx_group = BPnx_group,
                                                    Gc = Gc,
                                                    virus_df = virus_df)
    temp_x = pred_m.preprocessing_x(data_frame = edges_to_predict,
                         network = Gc_temp_group,
                         virus_df_temp = virus_df_temp, 
                         virus_df = virus_df)

    pred_group = prediction(temp_x =temp_x, clf_multi  =clf_multi, inv_dictionary = inv_dictionary)
    result_group = pred_group.join(edges_to_predict)
    return result_group, edges_to_predict

###############################################################################################################################
###############################################################################################################################

def generete_temp_network(virus, hosts, ViralFamily, PubMed, BPnx_group, Gc,IUCN, virus_df):
    #print('this function is in multiclass validation file')
    import math
    temp_BPnx = BPnx_group.copy() 
    #print (temp_BPnx.number_of_nodes()) ## checking number of nodes
    #virus_nodes = [x for x,y in temp_BPnx.nodes(data=True) if y['type']=='virus'] 
    q_df = pd.DataFrame.from_dict(dict(BPnx_group.nodes(data=True)), orient='index')
    q_df = q_df.loc[q_df.index.dropna()]
    virus_nodes = q_df[q_df['type'] == 'virus'].index.tolist()#creating list of virus nodes from bipartite network
    df = pd.DataFrame({'Virus2':virus_nodes}) # converting them to a dataframe
    df['Virus1'] = virus # dataframe with all possible combinations of new virus and viruses from BPnx
    temp_BPnx.add_node(virus, virusname=virus, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    #print (temp_BPnx.number_of_nodes()) ## rechecking number of nodes
    for h in hosts:
        temp_BPnx.add_edge(virus, h) ## adding new edge to the Bpnxtemp
    def get_n_shared_hosts(c): ## calculating number of neighbours for our new virus
            return len(list(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2'])))
    df['n_shared_hosts'] = df.apply(get_n_shared_hosts, axis=1)
    def addsharedhosts (c): ## identifiying number of neighbours for our new virus
            return  sorted(nx.common_neighbors(temp_BPnx, c['Virus1'],c['Virus2']))
    df["shared_hosts"] = df.apply(addsharedhosts, axis=1)
    
    def add_hosts_orders (c):
        order_list = IUCN[IUCN.ScientificName.isin(c['shared_hosts'])]['Order'].unique().tolist()
        return order_list
    df["shared_orders"] = df.apply(add_hosts_orders, axis=1)
    
    new_edges = df[df['n_shared_hosts']>0] ### list of new edges for new viruses
    #print(new_edges.shape)
    Gc_temp  = Gc.copy() ## creating a temporary copy of GC complete
    Gc_temp.add_node(virus, ViralFamily=ViralFamily, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    for index, row in new_edges.iterrows():
            if row['n_shared_hosts'] > 0:
                Gc_temp.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']), 
                                orders = ','.join(row['shared_orders']))
    #edges_to_predict = df[df['n_shared_hosts']==0]
    edges_to_predict = df
    edges_to_predict = edges_to_predict[edges_to_predict.Virus2 != 'nan']
    virus_df_temp = virus_df.copy()
    virus_df_temp.loc[len(virus_df_temp)]=[virus, ViralFamily, math.log(PubMed),1, 1]
    return Gc_temp, edges_to_predict, virus_df_temp

def prediction(temp_x, clf_multi, inv_dictionary):
    #print('prediction function is in multiclass validation file 2nd function')
    #print(temp_x.shape)
    inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.iteritems())
    temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
    #temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
    Order_prediction = pd.DataFrame(clf_multi.predict(temp_x)).replace(inv_dictionary)
    #print (Order_prediction.shape)
    probs = clf_multi.predict_proba(temp_x)
    prob_max =[] 
    for i in range (len(probs)):
        prob_max.append(np.amax(probs[i], axis = 1))
    max_prob = pd.DataFrame(prob_max).T
    prediction = Order_prediction.join(max_prob, lsuffix='_pr', rsuffix='_shakyata')
    return prediction

def cross_validation_predict(virus, hosts, ViralFamily, PubMed, BPnx_group, Gc, virus_df, clf_multi, inv_dictionary, IUCN):
    #print('cross_validation_predict function is in multiclass validation file')
    from HPnex import predict_multi as pred_m
    Gc_temp_group, edges_to_predict, virus_df_temp = generete_temp_network(virus = virus, 
                                                    hosts = hosts,            
                                                    ViralFamily = ViralFamily,
                                                    PubMed = PubMed,
                                                    BPnx_group = BPnx_group,
                                                    IUCN = IUCN,                       
                                                    Gc = Gc,
                                                    virus_df = virus_df)
    temp_x = pred_m.preprocessing_x(data_frame = edges_to_predict,
                         network = Gc_temp_group,
                         virus_df_temp = virus_df_temp, 
                         virus_df = virus_df)

    pred_group = prediction(temp_x =temp_x, clf_multi  =clf_multi, inv_dictionary = inv_dictionary)
    result_group = pred_group.join(edges_to_predict)
    return result_group, edges_to_predict


###############################################################################################################################
###############################################################################################################################

def run_cross_validation(i, df, XGB, data_path, virus_df, IUCN):
    
    print('run_cross_validation function is in multiclass validation file')
    from HPnex import functions as f
    from HPnex import classification as classify
    from HPnex import fitting_functions as fitt
    
    print('running model for group '+ str(i) )
    df_temp = df[df.group != i]
    import pickle
    dictionary = pickle.load(open("C:/Users/falco/Desktop/directory/Missing_links_in_viral_host_communities/outputs/dictionary_order_humans.pkl", "rb")) 
    inv_dictionary = {v: k for k, v in dictionary.iteritems()}
    print ("first construct bipartite network to reterive original data information about shared hosts")
    BPnx_group = f.construct_bipartite_taxa_virus_network(
        dataframe=df_temp,
        taxa_level = 'Order',
        network_name='Go',
        plot=False,
        filter_file=False,
        taxonomic_filter=None)

    print('generation of observed network after removing group '+ str(i))
    Gc_df, Gc = f.construct_unipartite_taxa_level_virus_virus_network(
        dataframe=df_temp,
        taxa_level = 'Order',
        network_name='Gc Order level',
        layout_func='fruchterman_reingold',
        plot=False,
        filter_file=False,
        taxonomic_filter=None,
        return_df=True)

    print ('getting network data for Observed network using Gc and BPnx_group')
    Multiclass_data = fitt.get_complete_network_data_for_fitting_multiclass(Gc = Gc, BPnx = BPnx_group, data_path= data_path, 
                                                    virus_df = virus_df, Species_file_name='\IUCN Mammals, Birds, Reptiles, and Amphibians.csv')

    print('preprocessing data for fitting model')
    from xgboost import XGBClassifier
    #### Standardize continuous variables
    from sklearn.preprocessing import StandardScaler
    from sklearn  import preprocessing
    from pandas_ml import ConfusionMatrix
    from HPnex import functions as f
    ### Running cross validation scores and predictions
    from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

    model_data = Multiclass_data 

    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    #def run_mulitlabel_model(model_data, cv, rf, virus_df, Gc_data):

    #predictors = [
    #   'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
    #    'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2', 'neighbors_n', 
    #    'adamic_adar', 'resource', 'preferential_attach'
    #]
    
    
    predictors = [
       'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2',
       
    ]

    import sklearn
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC, LinearSVC
    from sklearn import preprocessing
    from sklearn_pandas import DataFrameMapper
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


    model_data['shared_hosts_label'] = model_data['orders_label'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
    Y_ml_df = model_data['shared_hosts_label'].apply(str).str.strip("['']").str.replace("'", "").str.strip().str.split(', ', expand = True)
    Y_ml_df = Y_ml_df.replace(dictionary)
    X = model_data[list(predictors)].values


    #### Standardize continuous variables
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    data_processed = pd.DataFrame(X_std, columns=predictors)
    #data_processed.head()

    ### Encoding categorical variables
    le = preprocessing.LabelEncoder()
    le.fit(virus_df.viral_family.unique())
    model_data['F1'] = le.transform(model_data.ViralFamily1.fillna('Not_Assinged'))
    model_data['F2'] = le.transform(model_data.ViralFamily2.fillna('Not_Assinged'))
    data_processed['F1'] = model_data.F1
    data_processed['F2'] = model_data.F2
    data_processed.fillna(0, inplace=True)

    print('fitting the model for group '+ str(i))

    from HPnex import functions as f
    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.multioutput import MultiOutputClassifier


    XGB = XGB
    
    multi_target_classifier = MultiOutputClassifier(XGB, n_jobs=1)



    multi_target_classifier.fit(data_processed, Y_ml_df.fillna(19).values)
    print(multi_target_classifier)


    print ('predicting using fitted model for group '+ str(i))

    predict_df = df[df.group == i]
    predict_df = predict_df.groupby('Virus').agg({'Order':'unique',
                                     'viral_family':'unique',
                                     'PubMed_Search':'unique'}) #,ScientificName , 'PubMed_Search']
    predict_df['viral_family'] = predict_df['viral_family'].str.get(0)
    predict_df['PubMed_Search'] = predict_df['PubMed_Search'].str.get(0).astype(int)
    predict_df.reset_index(inplace = True)
    
    print ('running predictions')
    RESULT = []
    e_predict = []
    for index, row in predict_df.dropna().iterrows():
        result, edges_to_predict   = cross_validation_predict(virus =row['Virus'],
                                            hosts = row['Order'], 
                                            PubMed = row['PubMed_Search'], 
                                            ViralFamily = row['viral_family'], 
                                            BPnx_group = BPnx_group,
                                            Gc = Gc, 
                                            virus_df = virus_df, 
                                            clf_multi = multi_target_classifier,
                                            inv_dictionary = inv_dictionary, 
                                             IUCN = IUCN)
        RESULT.append(result)
        e_predict.append(edges_to_predict)

    result_group = pd.concat(RESULT, axis=0)
    edges_group = pd.concat(e_predict, axis=0)
    
    return result_group, edges_group

#######################################################################################################################################
#######################################################################################################################################


def run_cross_validation(i, df, XGB, data_path, virus_df, IUCN):
    print('run_cross_validation function is in multiclass validation file')
    from HPnex import functions as f
    from HPnex import classification as classify
    from HPnex import fitting_functions as fitt
    
    print('running model for group '+ str(i) )
    df_temp = df[df.group != i]
    import pickle
    dictionary = pickle.load(open("C:/Users/falco/Desktop/directory/Missing_links_in_viral_host_communities/outputs/dictionary_order_humans.pkl", "rb")) 
    inv_dictionary = {v: k for k, v in dictionary.iteritems()}
    print ("first construct bipartite network to reterive original data information about shared hosts")
    BPnx_group = f.construct_bipartite_host_virus_network(
        dataframe=df_temp,
        network_name='Go',
        plot=False,
        filter_file=False,
        taxonomic_filter=None)


    print('generation of observed network after removing group '+ str(i))
    Gc_df, Gc = f.construct_unipartite_virus_virus_network_order(
        dataframe=df_temp,
        network_name='all_network',
        IUCN = IUCN,
        layout_func='fruchterman_reingold',
        plot=False,
        filter_file=False,
        taxonomic_filter=None,
        return_df=True)

    print ('getting network data for Observed network using Gc and BPnx_group')
    Multiclass_data = fitt.get_complete_network_data_for_fitting_multiclass(Gc = Gc, BPnx = BPnx_group, data_path= data_path, 
                                                    virus_df = virus_df, Species_file_name='\IUCN Mammals, Birds, Reptiles, and Amphibians.csv')

    print('preprocessing data for fitting model')
    from xgboost import XGBClassifier
    #### Standardize continuous variables
    from sklearn.preprocessing import StandardScaler
    from sklearn  import preprocessing
    from pandas_ml import ConfusionMatrix
    from HPnex import functions as f
    ### Running cross validation scores and predictions
    from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

    model_data = Multiclass_data 

    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    #def run_mulitlabel_model(model_data, cv, rf, virus_df, Gc_data):

    #predictors = [
    #   'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
    #    'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2', 'neighbors_n', 
    #    'adamic_adar', 'resource', 'preferential_attach'
    #]
    
    predictors = [
       'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2'
    ]
    #predictors = [
    #   'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
    #    'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2',
    #    'VirusCluster1', 'VirusCluster2', 'resource', 'preferential_attach'
    #]


    import sklearn
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC, LinearSVC
    from sklearn import preprocessing
    from sklearn_pandas import DataFrameMapper
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


    model_data['shared_hosts_label'] = model_data['orders_label'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
    Y_ml_df = model_data['shared_hosts_label'].apply(str).str.strip("['']").str.replace("'", "").str.strip().str.split(', ', expand = True)
    Y_ml_df = Y_ml_df.replace(dictionary)
    X = model_data[list(predictors)].values


    #### Standardize continuous variables
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    data_processed = pd.DataFrame(X_std, columns=predictors)
    #data_processed.head()

    ### Encoding categorical variables
    le = preprocessing.LabelEncoder()
    le.fit(virus_df.viral_family.unique())
    model_data['F1'] = le.transform(model_data.ViralFamily1.fillna('Not_Assinged'))
    model_data['F2'] = le.transform(model_data.ViralFamily2.fillna('Not_Assinged'))
    data_processed['F1'] = model_data.F1
    data_processed['F2'] = model_data.F2
    data_processed.fillna(0, inplace=True)

    print('fitting the model for group '+ str(i))

    from HPnex import functions as f
    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.multioutput import MultiOutputClassifier


    XGB = XGB
    
    multi_target_classifier = MultiOutputClassifier(XGB, n_jobs=1)



    multi_target_classifier.fit(data_processed, Y_ml_df.fillna(19).values)
    print(multi_target_classifier)


    print ('predicting using fitted model for group '+ str(i))

    predict_df = df[df.group == i]
    predict_df = predict_df.groupby('Virus').agg({
                                     'ScientificName': 'unique',
                                     'Order':'unique',
                                     'viral_family':'unique',
                                     'PubMed_Search':'unique'}) #,ScientificName , 'PubMed_Search']
    print('scientific names')
    predict_df['viral_family'] = predict_df['viral_family'].str.get(0)
    predict_df['PubMed_Search'] = predict_df['PubMed_Search'].str.get(0).astype(int)
    predict_df.reset_index(inplace = True)
    
    print ('running predictions')
    RESULT = []
    e_predict = []
    for index, row in predict_df.dropna().iterrows():
        result, edges_to_predict   = cross_validation_predict(virus =row['Virus'],
                                            hosts = row['ScientificName'], 
                                            PubMed = row['PubMed_Search'], 
                                            ViralFamily = row['viral_family'], 
                                            BPnx_group = BPnx_group,
                                            Gc = Gc, 
                                            virus_df = virus_df, 
                                            clf_multi = multi_target_classifier,
                                            inv_dictionary = inv_dictionary, 
                                            IUCN = IUCN)
        RESULT.append(result)
        e_predict.append(edges_to_predict)

    result_group = pd.concat(RESULT, axis=0)
    edges_group = pd.concat(e_predict, axis=0)
    
    return result_group, edges_group


#######################################################################################################################################
#######################################################################################################################################

def generate_score(cv_preds, cv_epreds, virus_df, i, plot = False):
    print('generate_score function is in multiclass validation file')
    r_group = pd.concat([cv_preds, cv_epreds], axis=1)
    cols = r_group.filter(regex='_pr').columns.tolist()
    #r_group['combined_orders']=r_group[['0_pr', '10_pr',u'11_pr', '12_pr', '13_pr', '14_pr', '15_pr', '16_pr', '17_pr', '1_pr',
    #'2_pr', '3_pr', '4_pr', '5_pr', '6_pr', '7_pr', '8_pr', '9_pr']].values.tolist()
    r_group['combined_orders']=r_group[cols].values.tolist()
    r_group = r_group.loc[:,~r_group.columns.duplicated()]
    r_group['shared_hosts'] = r_group['shared_orders'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
    
    r_group['combined_orders'] = r_group['combined_orders'].apply(lambda x: set(x))
    r_group['shared_hosts'] = r_group['shared_hosts'].apply(lambda x: set(x))
    r_group['shared_hosts'] = r_group['shared_hosts'].apply(lambda x: map(str.title, x))
    
    print('accuracy matrix based on first prediction' )
    a =r_group[['0_pr', 'shared_hosts']]
    a = pd.concat([a, a.shared_hosts.apply(pd.Series)], axis=1)
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    cm = confusion_matrix(a['0_pr'], a[0])
    print ('Accuracy Score :',accuracy_score(a['0_pr'],  a[0]))
    print('Classification Report : ')
    print (classification_report(a['0_pr'],  a[0]))
    
    r_group['TP'] = [list(set(a).intersection(set(b))) for a, b in zip(r_group.shared_hosts, r_group.combined_orders)]
    r_group['FP'] = [list(set(b).difference(set(a))) for a, b in zip(r_group.shared_hosts, r_group.combined_orders)]
    r_group['FN'] = [list(set(a).difference(set(b))) for a, b in zip(r_group.shared_hosts, r_group.combined_orders)]
    #r_group = pd.merge(r_group, virus_df, left_on='Virus1', right_on='virus_name', how='left')
    r_group['group'] = i 
    #r_group.group.fillna(0, inplace= True)
    
    m = []
    for g in r_group.group.unique():
        temp_r = r_group[r_group.group == g]
        TP = temp_r['TP'].apply(pd.Series).stack().reset_index(drop=True).value_counts()
        FP = temp_r['FP'].apply(pd.Series).stack().reset_index(drop=True).value_counts()
        FN = temp_r['FN'].apply(pd.Series).stack().reset_index(drop=True).value_counts()
        matrix_group = pd.concat([TP, FP, FN], axis = 1)
        matrix_group.columns = ['TP', 'FP', 'FN']
        matrix_group['Group'] = g
        matrix_group['PPV'] = matrix_group.TP.fillna(0)/(matrix_group.TP.fillna(0)+ matrix_group.FP.fillna(0))
        matrix_group['Sensitivity'] = matrix_group.TP.fillna(0)/(matrix_group.TP.fillna(0)+ matrix_group.FN.fillna(0))
        m.append(matrix_group)
    matrix = pd.concat(m, axis=0).reset_index()
    matrix['support'] = matrix.TP+ matrix.FP +matrix.FN
    matrix['f1-score'] = 2*((matrix['PPV']*matrix['Sensitivity'])/(matrix['PPV']+matrix['Sensitivity']))
    matrix.columns = ['Order', 'TP', 'FP', 'FN', 'Group', 'PPV', 'Sensitivity', 'support', 'f1-score']
    if plot:
        import matplotlib.style as style
        style.use('fivethirtyeight')
        plt.rcParams['font.family'] = 'Times New Roman'


        sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 0.8})
        validation_matrix = matrix
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2,  figsize = [12,8], sharey= False)

        sns.boxplot(x="support", y="Order", data=validation_matrix.dropna(), ax = ax1)
        sns.stripplot(x="support", y="Order", data=validation_matrix.dropna(), jitter= True, color='#252525', ax= ax1)
        ax1.set_xlabel('support')
        ax1.set_title('Predicting Shared Host Order\n\n\n', horizontalalignment = 'center', loc = 'left', fontsize=16)
        text1 = 'Sample size for validation of XGBoost model performance in correctly predicting the type of links (host order) between two viruses\nthat did not share hosts in the observed network '+ r'$G_o$'+ ' and shared hosts in '+ r'$G_c$'+'.\n'
        ax1.text(-0.3, 0.99, text1, verticalalignment='bottom', 
                 horizontalalignment='left',
                 transform=ax1.transAxes,
                 color='gray', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_ylabel('')

        sns.boxplot(x="f1-score", y="Order", data=validation_matrix.dropna(), ax = ax2)
        sns.stripplot(x="f1-score", y="Order", data=validation_matrix.dropna(), jitter= True, color='#252525', ax= ax2)
        ax2.set_xlabel('f1-score')
        ax2.set_xlim(0,1.02)
        ax2.set_ylabel('')


        sns.boxplot(x="Sensitivity", y="Order", data=validation_matrix.dropna(), ax = ax3)
        sns.stripplot(x="Sensitivity", y="Order", data=validation_matrix.dropna(), jitter= True, color='#252525', ax= ax3)
        ax3.set_xlim(0,1.02)
        ax3.set_xlabel('Sensitivity')
        ax3.set_ylabel('')

        sns.boxplot(x="PPV", y="Order", data=validation_matrix.dropna(), ax = ax4)
        sns.stripplot(x="PPV", y="Order", data=validation_matrix.dropna(), jitter= True, color='#252525', ax= ax4)
        ax4.set_xlim(0,1.02)
        ax4.set_xlabel('Positive Predictive Value')
        ax4.set_ylabel('')
        plt.tight_layout()
        #plt.savefig('outputs/XGBoost_order_prediction_performance.png', dpi = 600)
        plt.show()
    
    return matrix
        



"""Running basic code:
Importing packages, setting working directory, 
printing out date"""

import os as os
os.chdir('C:/Users/Falco/Desktop/directory/Missing_links_in_viral_host_communities/')
import datetime as dt
str(dt.datetime.now())
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pandas_ml import ConfusionMatrix
data_path = 'C:\Users\Falco\Desktop\directory\Missing_links_in_viral_host_communities\data'
output_path = 'C:\Users\Falco\Desktop\directory\Missing_links_in_viral_host_communities\outputs'
from HPnex import functions as f
from HPnex import classification as classify
from HPnex import fitting_functions as fitt

import numpy as np
import networkx as nx
#np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix
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
from pandas_ml import ConfusionMatrix
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#### Standardize continuous variables
from sklearn.preprocessing import StandardScaler
from sklearn  import preprocessing
from pandas_ml import ConfusionMatrix
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
from sklearn.model_selection import GridSearchCV



###############################################################################################################################
###############################################################################################################################


def generete_temp_network(virus, hosts, ViralFamily, PubMed, BPnx_group, Gc, virus_df):
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
    new_edges = df[df['n_shared_hosts']>0] ### list of new edges for new viruses
    #print(new_edges.shape)
    Gc_temp  = Gc.copy() ## creating a temporary copy of GC complete
    Gc_temp.add_node(virus, ViralFamily=ViralFamily, type='virus', bipartite = 1) ## adding new node to the Bpnxtemp
    for index, row in new_edges.iterrows():
            if row['n_shared_hosts'] > 0:
                Gc_temp.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']))
    #edges_to_predict = df[df['n_shared_hosts']==0]
    edges_to_predict = df
    edges_to_predict = edges_to_predict[edges_to_predict.Virus2 != 'nan']
    virus_df_temp = virus_df.copy()
    virus_df_temp.loc[len(virus_df_temp)]=[virus, ViralFamily, math.log(PubMed),1, 1]
    return Gc_temp, edges_to_predict, virus_df_temp

###############################################################################################################################
###############################################################################################################################

def prediction(temp_x, clf_multi, inv_dictionary):
    inv_dictionary  = dict((k, v.title()) for k,v in inv_dictionary.iteritems())
    Order_prediction = pd.DataFrame(clf_multi.predict(temp_x)).replace(inv_dictionary)
    temp_x.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
    probs = clf_multi.predict_proba(temp_x)
    prob_max =[] 
    for i in range (len(probs)):
        prob_max.append(np.amax(probs[i], axis = 1))
    max_prob = pd.DataFrame(prob_max).T
    prediction = Order_prediction.join(max_prob, lsuffix='_pr', rsuffix='_prob')
    return prediction

###############################################################################################################################
###############################################################################################################################

def cross_validation_predict(virus, hosts, ViralFamily, PubMed, BPnx_group, Gc, virus_df, clf_multi, inv_dictionary):
    from HPnex import prediction as pred
    Gc_temp_group, edges_to_predict, virus_df_temp = generete_temp_network(virus = virus, 
                                                    hosts = hosts,            
                                                    ViralFamily = ViralFamily,
                                                    PubMed = PubMed,
                                                    BPnx_group = BPnx_group,
                                                    Gc = Gc,
                                                    virus_df = virus_df)
    temp_x = pred.preprocessing_x(data_frame = edges_to_predict,
                         network = Gc_temp_group,
                         virus_df_temp = virus_df_temp, 
                         virus_df = virus_df)

    pred_group = prediction(temp_x =temp_x, clf_multi  =clf_multi, inv_dictionary = inv_dictionary)
    result_group = pred_group.join(edges_to_predict)
    return result_group, edges_to_predict


###############################################################################################################################
###############################################################################################################################

def grid_cross_validation_data(i, df, data_path, virus_df):
    from HPnex import functions as f
    from HPnex import classification as classify
    from HPnex import fitting_functions as fitt
    
    print('running model for group '+ str(i) )
    df_temp = df[df.group != i]
    import pickle
    dictionary = pickle.load(open("C:\Users\Falco\Desktop\directory\Missing_links_in_viral_host_communities\outputs/dictionary_order_humans.pkl", "rb")) 
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

    predictors = [
        'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2'
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


    model_data['shared_hosts_label'] = model_data['shared_hosts_c'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
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
    
    return data_processed, Y_ml_df.fillna(19).values

    
    
from IPython.display import HTML
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import YouTubeVideo 
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.colors import ListedColormap
import networkx as nx
import urllib
import os as os
import pandas as pd
import numpy as np
import itertools

import networkx as nx

from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, ResetTool, PanTool, WheelZoomTool
import  bokeh.models.graphs as graphs
#from bokeh.model.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4




#######################################################################################################
#######################################################################################################

def run_classification_model(model_data, cv, rf, virus_df):
    from HPnex import functions as f
    #predictors = [
    #   'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
    #    'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2', 'neighbors_n', 
    #    'adamic_adar', 'resource', 'preferential_attach'
    #]
    
    predictors = [
       'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2'
    ]

    # datasets for sklearn
    Y = model_data["label"].values
    X = model_data[list(predictors)].values

    #### Standardize continuous variables
    from sklearn.preprocessing import StandardScaler
    from sklearn  import preprocessing
    from pandas_ml import ConfusionMatrix
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
    ### Running cross validation scores and predictions
    from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
    
    
    scores = cross_val_score(rf, data_processed, Y, cv=cv)
    print('\nAccuracy of model on cross validation dataset while training')
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(rf, data_processed, Y, cv=cv)

    print ('accuracy', accuracy_score(Y, y_pred))
    print(
        '\nprecision = positive predictive value\nrecall = sensitivity\nf-1 Score = harmonic average of precision and racall\nsupport  = n\n'
    )
    print (classification_report(Y, y_pred))
    
    cr = precision_recall_fscore_support(Y, y_pred)
    confusion_matrix = ConfusionMatrix(Y, y_pred)
    confusion_matrix.plot(
        backend='seaborn', normalized=False, cmap='Blues', annot=True, fmt='d')
    plt.show()
    data_processed['Virus1'] = model_data.Virus1
    data_processed['Virus2'] = model_data.Virus2
    data_processed['Prediction'] = y_pred
    return data_processed, scores, confusion_matrix, cr

#######################################################################################################
#######################################################################################################

def generate_data_for_multilabel_model(training_network, training_network_data, i,
                                       BPnx, data_path, virus_df, dictionary, Species_file_name, plot= False):
    from HPnex import functions as f
    IUCN = pd.read_csv(data_path+ Species_file_name)
    IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
    IUCN.loc[IUCN.ScientificName== 'Homo sapiens', 'Order'] = 'Humans'
    print('we have ' + str(len(training_network.edges)) + ' edges in training network (1 to 5 groups)')
    ## randomly assigning groups to all edges to remove
    ## Copying Gc to Go
    Go = training_network.copy()
    # remove group 1
    ebunch = ((u, v) for u, v, d in Go.edges(data=True)
              if d['remove_group'] == i)
    Go.remove_edges_from(ebunch)
    print('we have ' + str(len(Go.edges)) + ' edges in observed network ' + str(i))

    net_name = 'Observed network ' + str(i)
    print (net_name)
    if plot:
        f.plot_unipartite_network(
            title=net_name,
            network=Go,
            network_name=net_name,
            layout_func='fruchterman_reingold')
    
    """ Develop Dataset for the Go """
    print('\nDevelop Dataset for the Go\n')
    """STEP 2"""
    vlist = list(Go.nodes())
    """STEP 3"""
    d = pd.DataFrame(list(itertools.combinations(vlist, 2)))
    d.columns = ['Virus1', 'Virus2']
    """STEP 4"""

    def get_n_shared_hosts(c):
        return len(list(nx.common_neighbors(BPnx, c['Virus1'], c['Virus2'])))

    d['n_shared_hosts_c'] = d.apply(get_n_shared_hosts, axis=1)

    def addsharedhosts(c):
        return sorted(nx.common_neighbors(BPnx, c['Virus1'], c['Virus2']))

    d["shared_hosts_c"] = d.apply(addsharedhosts, axis=1)

    print ('getting Order and Family values for shared hosts')
    def getOrders (c):
        orderlist = []
        if len(c.shared_hosts_c) > 0:
            for h in (c.shared_hosts_c):
                try:
                    orderlist.append(IUCN.loc[IUCN['ScientificName'] == h, 'Order'].iloc[0])
                except:
                    orderlist.append('MatchNotFound')
        return orderlist
    d['orders_label'] = d.apply(getOrders, axis=1)


    """STEP 5"""
    ebunch = ((u, v) for u, v, d in training_network.edges(data=True)
              if d['remove_group'] == i)
    to_remove = pd.DataFrame(list(ebunch))
    to_remove.columns = ['Virus1', 'Virus2']
    to_remove['n_shared_hosts'] = 0
    to_remove['shared_hosts'] = [list() for x in range(len(to_remove.index))]
    """STEP 6"""
    m = pd.merge(d, to_remove, on=['Virus1', 'Virus2'], how='left')
    m.n_shared_hosts.fillna(m.n_shared_hosts_c, inplace=True)
    m.shared_hosts.fillna(m.shared_hosts_c, inplace=True)
    
    print("\nCalculating topographical features for 'Go'\n")

    Go_data = f.calculate_features_taxa_level(
        data_frame=m,
        network=Go,
        taxa_level = 'Genus',
        Species_file_name='\IUCN Mammals, Birds, Reptiles, and Amphibians.csv',
        data_path=data_path,
        virus_df=virus_df)

    print("\nGenerating model data lables for 'Go'\n")
    
    """STEP 8"""
    model_data = Go_data[Go_data.n_shared_hosts == 0]## Crucial step: Go_data has all pairs but those pairs from removed edges will have zero as their n_shared host along with known negatives
    model_data['label'] = np.where(model_data['n_shared_hosts_c'] > 0, 1, 0)

    model_data[
        'PubMedSeach_sum'] = model_data.PubMed_Search_ln1 + model_data.PubMed_Search_ln2


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

    # datasets for sklearn
    #ml = MultiLabelBinarizer().fit(Gc_data.shared_hosts.apply(lambda y: ['No_Sharing'] if len(y)==0 else y)) 
    #model_data['shared_hosts_label'] = model_data['shared_hosts_c'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
    
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
    
    return model_data, data_processed, Y_ml_df


#######################################################################################################
#######################################################################################################
def run_multilabel_model(model_data, cv, rf, Y_ml):
    from HPnex import functions as f
    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.multioutput import MultiOutputClassifier
    
    
    multi_target_classifier = MultiOutputClassifier(rf, n_jobs=1)
    
    print('\nCross validation score stared')
    cv_scores = cross_val_score(estimator=multi_target_classifier,
                              X=model_data,
                              y=Y_ml,
                              cv=cv)
    
    print('\nAccuracy of model on cross validation dataset while training')
    print("Accuracy: %0.6f (+/- %0.6f)" % (cv_scores.mean(), cv_scores.std() * 2))
    print('\nCross validation prediction started')
    preds_multilable = cross_val_predict(estimator=multi_target_classifier,
                                  X=model_data,
                                  y=Y_ml,
                                  cv=cv)
    
    #print('\nCross validation prediction of probability started')
    #preds_multilable_proba = cross_val_predict(estimator=multi_target_classifier,
    #                                           X=model_data,
    #                                           y=Y_ml,
    #                                           cv=cv, method = 'predic_proba')

    return cv_scores, preds_multilable, Y_ml
    
    
    
#######################################################################################################
#######################################################################################################
    
#######################################################################################################
#######################################################################################################































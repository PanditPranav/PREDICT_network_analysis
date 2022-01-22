def construct_combined_network(PREDICT_file, Viral_Spill_file, network_name, output_path, plot= False, return_df = False):
    print ('reading PREDICT data')
    
    P = pd.read_csv(PREDICT_file, encoding='ISO-8859-1', low_memory=False)
    P['ScientificName'] = P['ScientificName'].str.replace('[^\x00-\x7F]','')
    
    P['TaxaGroup'] = P.TaxaGroup.str.replace('[^\x00-\x7F]','')
    P['Order'] = P.Order.str.replace('[^\x00-\x7F]','')
    P['Class'] = P.Class.str.replace('[^\x00-\x7F]','')
    P['Family'] = P.Family.str.replace('[^\x00-\x7F]','')
    P['AnimalID'] = P.AnimalID.str.replace('[^\x00-\x7F]','')
    P['Virus'] = P.virus_name.str.replace('[^\x00-\x7F]','')
    P['ViralFamily'] = P.viral_family.str.replace('[^\x00-\x7F]','')

    """hosttaxa: creating dataframe of unique hosts and their characteristics to generate nodes"""
    hosttaxaP = P.groupby(['ScientificName']).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    vlistP = P.virus_name.dropna().unique().tolist()
    
    print ('reading ViralSpill data')
        
    #BP_VS = construct_bipartite_host_virus_network(data_filename = Viral_Spill_file, network_name= 'BP_VS', plot=False)
    if ".pickle" in Viral_Spill_file:
        V = pd.read_pickle(Viral_Spill_file)
    else:
        V = pd.read_csv(Viral_Spill_file, encoding='ISO-8859-1', low_memory=False)
        
    V['ScientificName'] = V['ScientificName'].str.replace('[^\x00-\x7F]','')
    V['viral_family'] = V.viral_family.str.replace('[^\x00-\x7F]','')

    """hosttaxa: creating dataframe of unique hosts and their characteristics to generate nodes"""
    hosttaxaV = V.groupby(['ScientificName']).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    vlistV = V.Virus.dropna().unique().tolist()
    
    vlist = list(set(vlistP).union(set(vlistV)))
    print 'Total viruses= '+ str(len(vlist))
    
    print 'Total hosts= '+str(len(list(set(hosttaxaP.ScientificName.unique().tolist()).union(set(hosttaxaV.ScientificName.unique().tolist())))))
    hosttaxa = pd.merge(hosttaxaP, hosttaxaV, on=['ScientificName'], how = 'outer')
    
    from networkx.algorithms import bipartite
    CN=nx.Graph()

    """Initiating host nodes"""

    for index, row in hosttaxa.iterrows():
        CN.add_node(row['ScientificName'], type="host", speciesname = row['ScientificName'], bipartite = 0 )

    """Initiating virus nodes"""

    for virus in vlist:
        CN.add_node(virus, type="virus", virusname = virus, bipartite = 1)

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    for index, row in P.iterrows():
        if row.ConfirmationResult == 'Positive':
            CN.add_edge(row['ScientificName'], row['Virus'], AnimalID = 'AnimalID', weight = 1)


    """Iterating through the raw data to add Edges if a virus is found in a host"""
    for index, row in V.iterrows():
        CN.add_edge(row['ScientificName'], row['Virus'], weight = 1)
        
    
    """Creating positions of the nodes"""
    #layout = nx.spring_layout(DG, k = 0.05, scale=2) #
    layout = nx.fruchterman_reingold_layout(CN, k = 0.05, iterations=50)
    """write graph """
    #nx.write_graphml(CN, network_name+".graphml")
    
    """Here we will copllapse the Bipartite network to monopartite
    Nodes will be viruses
    Edges will be hosts they share the virus with"""

    df = pd.DataFrame(list(itertools.combinations(vlist, 2)))
    df.columns = ['Virus1', 'Virus2']

    def get_n_shared_hosts(c):
        return len(sorted(nx.common_neighbors(CN, c['Virus1'],c['Virus2'])))
    df['n_shared_hosts'] = df.apply(get_n_shared_hosts, axis=1)
    
    print 'we have '+str(df.shape[0])+' virus pairs in our model'
    
    def addsharedhosts (c):
        l =  sorted(nx.common_neighbors(CN, c['Virus1'],c['Virus2']))
        #return str(l)[1:-1]
        return l
    df["shared_hosts"] = df.apply(addsharedhosts, axis=1)
    #df.head()
    
    CN_U = nx.Graph()

    """Initiating virus nodes"""

    for virus in pd.unique(df[['Virus1', 'Virus2']].values.ravel()).tolist():
        if virus not in vlistV:
            CN_U.add_node(virus, type="virus", virusname = virus, bipartite = 1, PREDICT = 'Yes')
        else:
            CN_U.add_node(virus, type="virus", virusname = virus, bipartite = 1, PREDICT = 'No')

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    for index, row in df.iterrows():
        if row['n_shared_hosts']> 0:
            CN_U.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'])

    """Creating positions of the nodes"""
    #layout = nx.spring_layout(DG, k = 0.05, scale=2) #
    layout = nx.fruchterman_reingold_layout(CN_U, k = 0.05, iterations=500, scale=2 )
    """write graph """
    nx.write_graphml(CN_U,output_path+'/'+network_name+".graphml")
    
    if plot:
        plot_unipartite_network(title = network_name,network = CN_U, network_name = network_name, layout_func = 'fruchterman_reingold')
    if return_df:
        return df, CN_U

#######################################################################################################
#######################################################################################################

def process_testing_data(testing_data, virus_df):
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
    from xgboost import XGBClassifier
    from sklearn.multiclass import OneVsRestClassifier
    predictors = [
        'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2'
    ]
    tdf = testing_data[predictors]
    #### Standardize continuous variables
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(tdf)
    data_processed = pd.DataFrame(X_std, columns=predictors)
    #data_processed.head()

    ### Encoding categorical variables
    le = preprocessing.LabelEncoder()
    le.fit(virus_df.viral_family.unique())
    testing_data['F1'] = le.transform(testing_data.ViralFamily1.fillna('Not_Assinged'))
    testing_data['F2'] = le.transform(testing_data.ViralFamily2.fillna('Not_Assinged'))
    data_processed['F1'] = testing_data.F1
    data_processed['F2'] = testing_data.F2
    data_processed.fillna(0, inplace=True)
    return data_processed

#############################################################################################
#############################################################################################


def generate_data_for_multilabel_model(Gc, Gc_data, i, BPnx, data_path, virus_df, Species_file_name, plot= False):
    from HPnex import functions as f
    IUCN = pd.read_csv(data_path+ Species_file_name)
    IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
    
    print('we have ' + str(len(Gc.edges)) + ' edges in complete network')
    ## randomly assigning groups to all edges to remove
    ## Copying Gc to Go
    Go = Gc.copy()
    # remove group 1
    ebunch = ((u, v) for u, v, d in Go.edges(data=True)
              if d['remove_group'] == i)
    Go.remove_edges_from(ebunch)
    print('we have ' + str(len(Go.edges)) + ' edges in observed network ' + str(i))

    net_name = 'Observed network ' + str(i)
    print net_name
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
    ebunch = ((u, v) for u, v, d in Gc.edges(data=True)
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
    model_data = Go_data[Go_data.n_shared_hosts == 0]
    model_data['label'] = np.where(model_data['n_shared_hosts_c'] > 0, 1, 0)

    model_data[
        'PubMedSeach_sum'] = model_data.PubMed_Search_ln1 + model_data.PubMed_Search_ln2


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

    # datasets for sklearn
    ml = MultiLabelBinarizer().fit(Gc_data.shared_hosts.apply(lambda y: ['No_Sharing'] if len(y)==0 else y)) 
    model_data['shared_hosts_label'] = model_data['shared_hosts_c'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
    Y_ml = ml.transform(model_data.shared_hosts_label)
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
    return data_processed, Y_ml, ml
#######################################################################################################
#######################################################################################################

def generate_data_for_multilabel_model(Gc, i, BPnx, data_path, virus_df, Species_file_name):
    from HPnex import functions as f
    
    IUCN = pd.read_csv(data_path+ Species_file_name)
    IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
    
    print('we have ' + str(len(Gc.edges)) + ' edges in complete network')
    ## randomly assigning groups to all edges to remove
    ## Copying Gc to Go
    Go = Gc.copy()
    # remove group 1
    ebunch = ((u, v) for u, v, d in Go.edges(data=True)
              if d['remove_group'] == i)
    Go.remove_edges_from(ebunch)
    print('we have ' + str(len(Go.edges)) + ' edges in observed network ' + str(i))

    net_name = 'Observed network ' + str(i)
    print net_name
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
    ebunch = ((u, v) for u, v, d in Gc.edges(data=True)
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

    Go_data = f.calculate_features(
        data_frame=m,
        network=Go,
        Species_file_name='\IUCN Mammals, Birds, Reptiles, and Amphibians.csv',
        data_path=data_path,
        virus_df=virus_df)

    print("\nGenerating model data lables for 'Go'\n")
    """STEP 8"""
    model_data = Go_data[Go_data.n_shared_hosts == 0]
    model_data['label'] = np.where(model_data['n_shared_hosts_c'] > 0, 1, 0)

    model_data[
        'PubMedSeach_sum'] = model_data.PubMed_Search_ln1 + model_data.PubMed_Search_ln2


    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    #def run_mulitlabel_model(model_data, cv, rf, virus_df, Gc_data):

    predictors = [
        'jaccard', 'betweeness_diff', 'in_same_cluster', 'degree_diff',
        'FamilyMatch', 'PubMed_diff', 'PubMed_Search_ln1', 'PubMed_Search_ln2'
    ]
    # datasets for sklearn
    ml = MultiLabelBinarizer().fit(Gc_data.orders.apply(lambda y: ['No_Sharing'] if len(y)==0 else y)) 
    model_data['orders_label'] = model_data['orders_label'].apply(lambda y: ['No_Sharing'] if len(y)==0 else y)
    Y_ml = ml.transform(model_data.orders_label)
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
    model_data['F1'] = le.transform(model_data.ViralFamily1)
    model_data['F2'] = le.transform(model_data.ViralFamily2)
    data_processed['F1'] = model_data.F1
    data_processed['F2'] = model_data.F2
    data_processed.fillna(0, inplace=True)
    return data_processed, Y_ml
#######################################################################################################
#######################################################################################################


def run_multilabel_model(model_data, cv, rf, virus_df, Y_ml, ml):
    from HPnex import functions as f
    import sklearn
    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC, LinearSVC
    from sklearn import preprocessing
    from sklearn_pandas import DataFrameMapper
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
    from scipy import interp
    from itertools import cycle
    import seaborn as sns
    lw = 2
    multi_target_classifier = MultiOutputClassifier(rf, n_jobs=1)
    print('\nRunning 5fold Cross Validation')
    cv_scores = cross_val_score(estimator=multi_target_classifier,
                              X=model_data,
                              y=Y_ml,
                              cv=sklearn.model_selection.KFold(shuffle=True, n_splits=5))
    
    print('\nAccuracy of model on cross validation dataset while training')
    print("Accuracy: %0.6f (+/- %0.6f)" % (cv_scores.mean(), cv_scores.std() * 2))
    print('\nRunning 5fold Cross Validated Prediction')
    preds_multilable = cross_val_predict(estimator=multi_target_classifier,
                                  X=model_data,
                                  y=Y_ml,
                                  cv=sklearn.model_selection.KFold(shuffle=True, n_splits=5))
    
    Taxa_prediction = ml.inverse_transform(preds_multilable)
    Taxa_observed = ml.inverse_transform(Y_ml)
    
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
    pre_recall = precision_recall_fscore_support(preds_multilable, Y_ml)
    
    validation_matrix =pd.DataFrame({'precision':pre_recall[0], 'recall':pre_recall[1], 'fscore':pre_recall[2],'support': pre_recall[3]},  index=list(ml.classes_))
    
    print ('PREDICTING Probabilities for various ORDERS & ROC analysis')
    
    n_classes = len(np.unique(Y_ml))
    preds_multilable_prob = cross_val_predict(estimator=multi_target_classifier,
                                  X=model_data,
                                  y=Y_ml,
                                  cv=sklearn.model_selection.KFold(shuffle=True, n_splits=5), method='predict_proba')
    
    print (preds_multilable_prob.shape)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, preds_multilable_prob[:, i], pos_label = clf.classes_[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    cmap = sns.color_palette("Set2", 12)

    plt.rcParams['figure.figsize'] = (9,9)
    plt.figure()
    
    colors = sns.color_palette("husl", n_classes)
    colors = colors.as_hex()
    for i, color, s in zip(range(n_classes), colors, clf.classes_):
        #name = 'ROC ' + clf.classes_[i]+'(area = {1:0.2f})'
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'
                 ''.format(s, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.savefig(data_path+'/ROC_curve.png', dpi = 600)
    plt.show()
    
    return validation_matrix, Taxa_prediction, Taxa_observed, preds_multilable_prob



#######################################################################################################
#######################################################################################################
    
    
def calculate_features_taxa_level(data_frame, network, taxa_level, Species_file_name, data_path, virus_df, long = False):
    print('this function is in function file 2nd function')
    print ('calculating topographical features')

    ################################################################################################################################
    ################################################################################################################################
    if long:
    
    ############################################################################################################################
    ################################################################################################################################
        
    ################################################################################################################################
    ################################################################################################################################
        IUCN = pd.read_csv(data_path+ Species_file_name)
        IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
        IUCN.loc[IUCN.ScientificName== 'Homo sapiens', 'Order'] = 'Humans'
        

    ################################################################################################################################
    ################################################################################################################################
        print ('getting Order and Family values for shared hosts')
        def getOrders (c):
            orderlist = []
            if len(c.shared_hosts) > 0:
                for h in (c.shared_hosts):
                    try:
                        orderlist.append(IUCN.loc[IUCN[taxa_level] == h, 'Order'].iloc[0])
                    except:
                        orderlist.append('MatchNotFound')
            return orderlist
        data_frame['orders'] = data_frame.apply(getOrders, axis=1)

    ################################################################################################################################
    ################################################################################################################################
        def getFamily (c):
            orderlist = []
            if len(c.shared_hosts) > 0:
                for h in (c.shared_hosts):
                    try:
                        orderlist.append(IUCN.loc[IUCN[taxa_level] == h, 'Family'].iloc[0])
                    except:
                        orderlist.append('MatchNotFound')
            return orderlist
        data_frame['families'] = data_frame.apply(getFamily, axis=1)

    ################################################################################################################################
    ################################################################################################################################
        def OrderRichness (c):
            return len(set(c.orders))

        def FamilyRichness (c):
            return len(set(c.families))

        data_frame['OrderRichness'] = data_frame.apply(OrderRichness, axis=1)
        data_frame['FamilyRichness'] = data_frame.apply(FamilyRichness, axis=1)
        print ('richness calculations complete')

    ################################################################################################################################
    ################################################################################################################################
        print ('calculating ShannonH index of diversity for shared Orders and Familes of taxa')
        def shannon_order(c):
            total = len(c.orders)
            counts = pd.Series(c.orders).value_counts().tolist()
            h = sum(map(lambda x:abs(np.log(x/float(total)))*(x/float(total)), counts))
            return h

        data_frame['Order_H'] = data_frame.apply(shannon_order, axis=1)

    ################################################################################################################################
    ################################################################################################################################
        def shannon_family(c):
            total = len(c.families)
            counts = pd.Series(c.families).value_counts().tolist()
            h = sum(map(lambda x:abs(np.log(x/float(total)))*(x/float(total)), counts))
            return h

        data_frame['Familiy_H'] = data_frame.apply(shannon_family, axis=1)

    
    ################################################################################################################################
    ################################################################################################################################
    print ('calculating adamic/adar index')
    def adar (c):
        return sorted(nx.adamic_adar_index(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["adamic_adar"] = data_frame.apply(adar, axis=1)
        
    ################################################################################################################################
    ################################################################################################################################
    print ('calculating Resource coefficients')
    def resource (c):
        return sorted(nx.resource_allocation_index(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["resource"] = data_frame.apply(resource, axis=1)
        
    ################################################################################################################################
    ################################################################################################################################
    print ('calculating preferential attachment coefficients')
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
    print ('calculating Jaccard coefficients')
    def jaccard (c):
        return sorted(nx.jaccard_coefficient(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["jaccard"] = data_frame.apply(jaccard, axis=1)


    ################################################################################################################################
    ################################################################################################################################
    print ('listing neighbors')
    def neighbors  (c):
        l = sorted(nx.common_neighbors(network, c['Virus1'],c['Virus2']))
        return str(l)[1:-1]
    data_frame["neighbors"] = data_frame.apply(neighbors, axis=1)

    ################################################################################################################################
    ################################################################################################################################
    print ('calculating number of neighbors')
    def neighbors_n  (c):
        return len(sorted(nx.common_neighbors(network, c['Virus1'],c['Virus2'])))
    data_frame["neighbors_n"] = data_frame.apply(neighbors_n, axis=1)

    ################################################################################################################################
    ################################################################################################################################
    print ('calculating difference in betweenness centrality')
    btw = nx.betweenness_centrality(network, 25)
    def betweenDiff(c):
        return abs(btw[c['Virus1']] - btw[c['Virus2']])
    data_frame["betweeness_diff"] = data_frame.apply(betweenDiff, axis=1)  

    ################################################################################################################################
    ################################################################################################################################
    print ('calculating is nodes are in a same cluster')
    import community
    partition = community.best_partition(network)

    ################################################################################################################################
    ################################################################################################################################
    def in_same_cluster(c):
        if(partition[c['Virus1']] == partition[c['Virus2']]):
            return True
        else:
            return False
    data_frame["in_same_cluster"] = data_frame.apply(in_same_cluster, axis=1)    
    data_frame['in_same_cluster'] =np.where(data_frame['in_same_cluster']== True, 1, 0)

    ################################################################################################################################
    ################################################################################################################################
    print ('calculating difference in degree')
    degree = nx.degree(network)
    def degreeDiff(c):
        return abs(degree[c['Virus1']] - degree[c['Virus2']])
    data_frame["degree_diff"] = data_frame.apply(degreeDiff, axis=1) 

    ################################################################################################################################
    ################################################################################################################################
    print ('calculating shortest path length')
    def ShortPathLen(c):
        if c["hasPath"]:
            return nx.shortest_path_length(network, c['Virus1'], c['Virus2'])
        else:
            return np.nan
    data_frame["ShortPathLen"] = data_frame.apply(ShortPathLen, axis=1)
    data_frame['ShortPathLen'].fillna(0, inplace = True)

    ################################################################################################################################
    ################################################################################################################################
    print ('Matching Virus Families')
    data_frame = pd.merge(data_frame,virus_df[['virus_name','viral_family','PubMed_Search_ln']], left_on='Virus1', right_on='virus_name', how='left')
    data_frame = pd.merge(data_frame,virus_df[['virus_name','viral_family', 'PubMed_Search_ln']], left_on='Virus2', right_on='virus_name', how='left')
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
    data_frame['FamilyMatch'] =np.where(data_frame['FamilyMatch']== 'True', 1, 0)
    ################################################################################################################################
    ################################################################################################################################
    print ('difference in PubMed hits')
    def PubMed_hits(c):
        return abs(c.PubMed_Search_ln1 - c.PubMed_Search_ln2)
    data_frame['PubMed_diff'] = data_frame.apply(PubMed_hits, axis=1)
    ################################################################################################################################
    ################################################################################################################################
    data_frame['Link'] =np.where(data_frame['n_shared_hosts']>= 1, 1, 0)
    return data_frame



#############################################################################################
#############################################################################################

def construct_unipartite_taxa_level_virus_virus_network(dataframe, taxa_level, network_name, 
                                                        layout_func = 'fruchterman_reingold',
                                                        plot= False, filter_file=  False, 
                                                        taxonomic_filter = None,
                                                        return_df = False):
    
    """first construct bipartite network"""
    if filter_file:
        BPnx = construct_bipartite_taxa_virus_network(dataframe = dataframe, taxa_level = taxa_level,  network_name= network_name, 
                                                      plot=False, filter_file= True, taxonomic_filter = taxonomic_filter)
    else:
        BPnx = construct_bipartite_taxa_virus_network(dataframe = dataframe, taxa_level = taxa_level, network_name= network_name, 
                                                      plot=False, filter_file= False, taxonomic_filter = taxonomic_filter)
    
    #if data_filename:
    #    """Importing all the data
    #    data: """
    #    if ".pickle" in data_filename:
    #        data = pd.read_pickle(data_filename,)
    #    else:
    #        data = pd.read_csv(data_filename, encoding='ISO-8859-1', low_memory=False)
    data = dataframe
        
    data[taxa_level] = data[taxa_level].str.replace('[^\x00-\x7F]','')
    if taxonomic_filter:
        data = data[data.viral_family == taxonomic_filter]

    """hosttaxa: creating dataframe of unique hosts and their characteristics to generate nodes"""
    hosttaxa = data.groupby([taxa_level]).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    virus_dataframe = data.groupby(['Virus', 'viral_family']).size().reset_index().rename(columns={0:'count'})
    vlist = data.Virus.dropna().unique().tolist()
    
    """Here we will copllapse the Bipartite network to monopartite
    Nodes will be viruses
    Edges will be hosts they share the virus with"""

    df = pd.DataFrame(list(itertools.combinations(vlist, 2)))
    df.columns = ['Virus1', 'Virus2']

    def get_n_shared_hosts(c):
        return len(list(nx.common_neighbors(BPnx, c['Virus1'],c['Virus2'])))
    df['n_shared_hosts'] = df.apply(get_n_shared_hosts, axis=1)
    
    
    
    #"""removing pairs with 0 shared hosts"""
    #df.drop(df[df.n_shared_hosts == 0].index, inplace=True)
    def addsharedhosts (c):
        return  sorted(nx.common_neighbors(BPnx, c['Virus1'],c['Virus2']))
    
    
    df["shared_hosts"] = df.apply(addsharedhosts, axis=1)
    print 'we have '+str(df.shape[0])+' virus pairs in our model'
    
    """Creating the a network now using the df
    EDGES will be weighted according to number of shared hosts"""


    VS_unx = nx.Graph()

    """"Initiating virus nodes"""

    for index, row in virus_dataframe.iterrows():
        VS_unx.add_node(row['Virus'], type="virus",  
                    ViralFamily = str(row['viral_family']), bipartite = 1)

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    for index, row in df.iterrows():
        if row['n_shared_hosts'] > 0:
            VS_unx.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']))
            

    """Creating positions of the nodes"""
    if layout_func == 'fruchterman_reingold':
        layout = nx.fruchterman_reingold_layout(VS_unx, scale=2 )#k = 0.05, iterations=500
    elif layout_func =='spring':
        layout = nx.spring_layout(VS_unx, k = 0.05, scale=2)
    elif layout_func =='circular':
        layout = nx.circular_layout(VS_unx, scale=1, center=None, dim=2)
    elif layout_func == 'kamada':
        layout = nx.kamada_kawai_layout(VS_unx, scale=1, center=None, dim=2)
    elif layout_func == 'spectral':
        layout = nx.spectral_layout(VS_unx, scale=1, center=None, dim=2)
    else:
        layout = nx.fruchterman_reingold_layout(VS_unx, scale=2 )#k = 0.05, iterations=500
    
    """write graph """
    #nx.write_graphml(VS_unx, network_name+"unipartite.graphml")
    
       
    if plot:
        plot_unipartite_network(title = network_name,network = VS_unx, network_name = network_name, layout_func = layout_func)
    
    if return_df:
        return df, VS_unx
        



#######################################################################################################
#######################################################################################################


def construct_bipartite_taxa_virus_network(dataframe, taxa_level, network_name, plot= False, filter_file=  False, 
                                          taxonomic_filter = None):
    
    
    #if data_filename:
    #    """Importing all the data
    #    data: """
    #    if ".pickle" in data_filename:
    #        data = pd.read_pickle(data_filename,)
    #    else:
    #        data = pd.read_csv(data_filename, encoding='ISO-8859-1', low_memory=False)
    data = dataframe
        
    """ filter data according to viral family """
    if taxonomic_filter:
        data = data[data.viral_family == taxonomic_filter]

    """hosttaxa: creating dataframe of unique hosts and their characteristics to generate nodes"""
    hosttaxa = data.groupby([taxa_level]).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    vlist = data.Virus.unique().tolist()
    
    """Construction of network"""

    from networkx.algorithms import bipartite
    DG=nx.Graph()

    """Initiating host nodes"""

    for index, row in hosttaxa.iterrows():
        DG.add_node(row[taxa_level], type="host", 
                    speciesname = row[taxa_level], bipartite = 0 )

    """Initiating virus nodes"""

    for virus in vlist:
        DG.add_node(virus, type="virus", virusname = virus, bipartite = 1)

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    
    """Iterating through the raw data to add Edges if a virus is found in a host"""
    if filter_file:
        for index, row in data.iterrows():
            if row.ConfirmationResult == 'Positive':
                DG.add_edge(row[taxa_level], row['Virus'], AnimalID = 'AnimalID', weight = 1)
    else:
        for index, row in data.iterrows():
            DG.add_edge(row[taxa_level], row['Virus'], weight = 1)

    """Creating positions of the nodes"""
    #layout = nx.spring_layout(DG, k = 0.05, scale=2) #
    layout = nx.fruchterman_reingold_layout(DG, k = 0.05, iterations=50)
    """write graph """
    #nx.write_graphml(DG, network_name + "_bipartite.graphml")
    
       
    """
    Plotting
    """
    if plot:
        from bokeh.models import ColumnDataSource

        nodes, nodes_coordinates = zip(*layout.items())
        nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
        node_data = dict(x=nodes_xs, y=nodes_ys, name=nodes)
        nd = pd.DataFrame.from_dict(node_data)
        def addNodeType(c):
            if c.name in vlist:
                return 'Virus'
            else:
                return 'Host'
        #nd['node_type'] = nd.apply(addNodeType, axis=1)
        
        virusc = '#ef8a62' # ,'#e05354'
        hostc = '#67a9cf'
        nt = []
        nodecolors = []
        for i in range (nd.shape[0]):
            if nd.name[i] in vlist:
                nt.append('virus')
                nodecolors.append(virusc)
            else:
                nt.append(taxa_level)
                nodecolors.append(hostc)


        nd['node_type'] = nt 
        nd['colors'] = nodecolors
        #nodes_source = ColumnDataSource(nd.to_dict())
        nodes_source = ColumnDataSource(dict(x=nd.x.tolist(), y=nd.y.tolist(),
                              name = nd.name.tolist(),
                              node_type = nd.node_type.tolist(), colors = nd.colors.tolist()))
        from bokeh.plotting import show, figure , output_file
        from bokeh.io import output_notebook
        from bokeh.models import HoverTool
        output_notebook()
        
        """
        Generate the figure
        1. Create tools 
        2. Set plot size and tools

        """

        #hover = HoverTool(tooltips=[('name', '@name'),('type', '@node_type')])
        plot = figure(title=network_name+": Host Genus virus bipartite network", 
                      plot_width=1200, plot_height=1200,
                      tools=['pan','wheel_zoom','reset','box_zoom','tap' ])

        """
        plot main circles

        1. Plot only nodes according to their positions

        """
        r_circles = plot.circle('x', 'y', source=nodes_source, size=10,
                                color= "colors", alpha=0.5, level = 'overlay',)

        """
        Function 

        Get data for generation of edges 

        """
        def get_edges_specs(_network, _layout):
            c = dict(xs=[], ys=[], alphas=[])
            #print d
            weights = [d['weight'] for u, v, d in _network.edges(data=True)]
            max_weight = max(weights)
            calc_alpha = lambda h: 0.1 + 0.6 * (h / max_weight)

            # example: { ..., ('user47', 'da_bjoerni', {'weight': 3}), ... }
            for u, v, data in _network.edges(data=True):
                c['xs'].append([_layout[u][0], _layout[v][0]])
                c['ys'].append([_layout[u][1], _layout[v][1]])
                c['alphas'].append(calc_alpha(data['weight']))
            return c


        """
        get the data for edges

        """
        lines_source = ColumnDataSource(get_edges_specs(DG, layout))

        """
        plot edge lines

        """

        r_lines = plot.multi_line('xs', 'ys', line_width=1.5,
                                  alpha=1 , color='#b3b6b7',
                                  source=lines_source)

        """Centrality """

        centrality = nx.algorithms.centrality.betweenness_centrality(DG)
        """ first element are nodes again """
        _, nodes_centrality = zip(*centrality.items())
        max_centraliy = max(nodes_centrality)
        nodes_source.add([7 + 15 * t / max_centraliy
                          for t in nodes_centrality],
                         'centrality')

        """Communities"""
        import community #python-louvain
        partition = community.best_partition(DG)
        p_, nodes_community = zip(*partition.items())
        nodes_source.add(nodes_community, 'community')
        community_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628',
                            '#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec',
                            '#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d',
                            '#666666']
        nodes_source.add([community_colors[t % len(community_colors)]
                          for t in nodes_community],'community_color')

        """Host Type colour"""

        


        """Update the plot with communities and Centrality"""

        r_circles.glyph.size = 'centrality'
        hover = HoverTool(tooltips=[('', '@name')], renderers=[r_circles])
        plot.add_tools(hover)
        output_file(network_name+"_bipartite.html")
       
        show(plot)
    
    return DG

#######################################################################################################
#######################################################################################################


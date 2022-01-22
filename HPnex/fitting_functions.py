"""Running basic code:
Importing packages, setting working directory, 
printing out date"""

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


plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
sns.set_style('white')
plt.close()


#############################################################################################
#############################################################################################


def get_complete_network_data_for_fitting(Gc, BPnx, data_path, virus_df, Species_file_name, plot = False):
    from HPnex import functions as f
    
    print('Beta  function to for validated model')
    IUCN = pd.read_csv(data_path+ Species_file_name,encoding= 'unicode_escape')
    IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
    IUCN.loc[IUCN.ScientificName== 'Homo sapiens', 'Order'] = 'Humans'
    print('we have ' + str(len(Gc.edges)) + ' edges in complete network')
    ## randomly assigning groups to all edges to remove
    ## Copying Gc to Go
    Go = Gc.copy()
    print('we have ' + str(len(Go.edges)) + ' edges in COMPLETE network ') 
    net_name = 'COMPLETE network'
    print (net_name)
    if plot:
        f.plot_unipartite_network(
            title=net_name,
            network=Go,
            network_name=net_name,
            layout_func='fruchterman_reingold')
    """ Develop Dataset for the COMPLETE NETWORK """
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
    print ('Not removing any edges as this is for COMPLETE network data')
    #ebunch = ((u, v) for u, v, d in Gc.edges(data=True)
    #          if d['remove_group'] == i)
    #to_remove = pd.DataFrame(list(ebunch))
    #to_remove.columns = ['Virus1', 'Virus2']
    #to_remove['n_shared_hosts'] = 0
    #to_remove['shared_hosts'] = [list() for x in range(len(to_remove.index))]
    """STEP 6"""
    #m = pd.merge(d, to_remove, on=['Virus1', 'Virus2'], how='left')
    m = d
    m['n_shared_hosts'] = m['n_shared_hosts_c']
    m.n_shared_hosts.fillna(m.n_shared_hosts_c, inplace=True)
    m['shared_hosts'] = m['shared_hosts_c']
    m.shared_hosts.fillna(m.shared_hosts_c, inplace=True)

    print("\nCalculating topographical features for 'Go'\n")

    Go_data = f.calculate_features(
        data_frame=m,
        network=Go,
        Species_file_name= Species_file_name,
        data_path=data_path,
        virus_df=virus_df)

    #print("\nGenerating model data lables for 'Go'\n")
    #"""STEP 8"""
    Go_data['label'] = np.where(Go_data['n_shared_hosts_c'] > 0, 1, 0)
    Go_data[
        'PubMedSeach_sum'] = Go_data.PubMed_Search_ln1 + Go_data.PubMed_Search_ln2
    return Go_data





#############################################################################################
#############################################################################################


def get_complete_network_data_for_fitting_multiclass(Gc, BPnx, data_path, virus_df, Species_file_name, plot = False):
    
    from HPnex import functions as f
    IUCN = pd.read_csv(data_path+ Species_file_name,encoding= 'unicode_escape')
    IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
    IUCN.loc[IUCN.ScientificName== 'Homo sapiens', 'Order'] = 'Humans'
    print('we have ' + str(len(Gc.edges)) + ' edges in complete network')
    ## randomly assigning groups to all edges to remove
    ## Copying Gc to Go
    Go = Gc.copy()
    print('we have ' + str(len(Go.edges)) + ' edges in observed network ')

    net_name = 'COMPLETE network'
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
    #ebunch = ((u, v) for u, v, d in Gc.edges(data=True)
    #          if d['remove_group'] == i)
    #to_remove = pd.DataFrame(list(ebunch))
    #to_remove.columns = ['Virus1', 'Virus2']
    #to_remove['n_shared_hosts'] = 0
    #to_remove['shared_hosts'] = [list() for x in range(len(to_remove.index))]
    """STEP 6"""
    #m = pd.merge(d, to_remove, on=['Virus1', 'Virus2'], how='left')
    m = d
    m['n_shared_hosts'] = m['n_shared_hosts_c']
    m.n_shared_hosts.fillna(m.n_shared_hosts_c, inplace=True)
    m['shared_hosts'] = m['shared_hosts_c']
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
    Go_data['label'] = np.where(Go_data['n_shared_hosts_c'] > 0, 1, 0)

    Go_data[
        'PubMedSeach_sum'] = Go_data.PubMed_Search_ln1 + Go_data.PubMed_Search_ln2

    return Go_data


#############################################################################################
#############################################################################################













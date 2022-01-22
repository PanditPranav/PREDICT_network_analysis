
# coding: utf-8

# In[1]:


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

def plot_unipartite_network (title,network, network_name, layout_func):
    """Creating positions of the nodes"""
    if layout_func == 'fruchterman_reingold':
        layout = nx.fruchterman_reingold_layout(network, scale=2 )#k = 0.05, iterations=500
    elif layout_func =='spring':
        layout = nx.spring_layout(network, k = 0.05, scale=2)
    elif layout_func =='circular':
        layout = nx.circular_layout(network, scale=1, center=None, dim=2)
    elif layout_func == 'kamada':
        layout = nx.kamada_kawai_layout(network, scale=1, center=None, dim=2)
    elif layout_func == 'spectral':
        layout = nx.spectral_layout(network, scale=1, center=None, dim=2)
    else:
        layout = nx.fruchterman_reingold_layout(network, scale=2 )#k = 0.05, iterations=500
    
    
    
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import show, figure , output_file
    from bokeh.io import output_notebook
    from bokeh.models import HoverTool
    output_notebook()

    nodes, nodes_coordinates = zip(*layout.items())
    
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
    
    #nodes_source = ColumnDataSource(dict(x=nodes_xs, y=nodes_ys,
    #                                     name=nodes,))
    node_data = dict(x=nodes_xs, y=nodes_ys, name=nodes)
    nd = pd.DataFrame.from_dict(node_data).dropna()
    #hostc = '#377eb8'
    nodes_source = ColumnDataSource(dict(x=nd.x.tolist(), y=nd.y.tolist(),
                          name = nd.name.tolist()))
    
    """
    Generate the figure
    1. Create tools 
    2. Set plot size and tools

    """

    #hover = HoverTool(tooltips=[('', '@name')])
    #hover = HoverTool(names=["name"])
    plot = figure(title=title, 
                  plot_width=800, plot_height=800,
                  tools=['pan','wheel_zoom', 'reset','box_zoom','tap' ])

    """
    plot main circles

    1. Plot only nodes according to their positions

    """
    r_circles = plot.circle('x', 'y', source=nodes_source, size=10,
                            color= '#377eb8', alpha=0.5, level = 'overlay',name='name')


    """
    Function 

    Get data for generation of edges 

    """
    def get_edges_specs(_network, _layout):
        c = dict(xs=[], ys=[], alphas=[])
        #print d
        weights = [d['weight'] for u, v, d in _network.edges(data=True)]
        max_weight = max(weights)
        calc_alpha = lambda h: 0.1 + 0.5 * (h / max_weight)

        # example: { ..., ('user47', 'da_bjoerni', {'weight': 3}), ... }
        for u, v, data in _network.edges(data=True):
            c['xs'].append([_layout[u][0], _layout[v][0]])
            c['ys'].append([_layout[u][1], _layout[v][1]])
            c['alphas'].append(calc_alpha(data['weight']))
        return c


    """
    get the data for edges

    """
    lines_source = ColumnDataSource(get_edges_specs(network, layout))

    """
    plot edge lines

    """

    r_lines = plot.multi_line('xs', 'ys', line_width=1.5,
                              alpha=1 , color='#b3b6b7',
                              source=lines_source, )#name = 'edge'

    """Centrality """

    centrality = nx.algorithms.centrality.betweenness_centrality(network)
    """ first element are nodes again """
    _, nodes_centrality = zip(*centrality.items())
    max_centraliy = max(nodes_centrality)
    nodes_source.add([7 + 15 * t / max_centraliy
                      for t in nodes_centrality],
                     'centrality')

    """Communities"""

    from community import community_louvain
    partition = community_louvain.best_partition(network)
    
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
    r_circles.glyph.fill_color = 'community_color'

    hover = HoverTool(tooltips=[('', '@name')], renderers=[r_circles])
    plot.add_tools(hover)
    
    output_file(network_name+"_unipartite.html")
    
    show(plot)


#############################################################################################
#############################################################################################


def construct_bipartite_host_virus_network(dataframe, network_name, plot= False, filter_file=  False, 
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
    hosttaxa = data.groupby(['ScientificName']).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    vlist = data.virus_name.dropna().unique().tolist()
    
    """Construction of network"""

    from networkx.algorithms import bipartite
    DG=nx.Graph()

    """Initiating host nodes"""

    for index, row in hosttaxa.iterrows():
        DG.add_node(row['ScientificName'], type="host", 
                    speciesname = row['ScientificName'], bipartite = 0 )

    """Initiating virus nodes"""

    for virus in vlist:
        DG.add_node(virus, type="virus", virusname = virus, bipartite = 1)

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    
    """Iterating through the raw data to add Edges if a virus is found in a host"""
    if filter_file:
        for index, row in data.iterrows():
            if row.ConfirmationResult == 'Positive':
                DG.add_edge(row['ScientificName'], row['virus_name'], AnimalID = 'AnimalID', weight = 1)
    else:
        for index, row in data.iterrows():
            DG.add_edge(row['ScientificName'], row['virus_name'], weight = 1)

    """Creating positions of the nodes"""
    #layout = nx.spring_layout(DG, k = 0.05, scale=2) #
    layout = nx.fruchterman_reingold_layout(DG, k = 0.05, iterations=50)
    """write graph """
    nx.write_graphml(DG, network_name + "_bipartite.graphml")
    
       
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
                nt.append('host')
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
        plot = figure(title=network_name+": Host virus bipartite network", 
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
        
        import community
        partition = community.best_partition(network)
        #import community #python-louvain
        #partition = community.best_partition(DG)
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



#############################################################################################
#############################################################################################

def construct_unipartite_virus_virus_network(dataframe, network_name,
                                             layout_func = 'fruchterman_reingold',
                                             plot= False, filter_file=  False, 
                                             taxonomic_filter = None,
                                             return_df = False):
    
    """first construct bipartite network"""
    if filter_file:
        BPnx = construct_bipartite_host_virus_network(dataframe = dataframe, network_name= network_name, 
                                                      plot=False, filter_file= True, taxonomic_filter = taxonomic_filter)
    else:
        BPnx = construct_bipartite_host_virus_network(dataframe = dataframe, network_name= network_name, 
                                                      plot=False, filter_file= False, taxonomic_filter = taxonomic_filter)
    
    #if data_filename:
    #    """Importing all the data
    #    data: """
    #    if ".pickle" in data_filename:
    #        data = pd.read_pickle(data_filename,)
    #    else:
    #        data = pd.read_csv(data_filename, encoding='ISO-8859-1', low_memory=False)
    data = dataframe
        
    data['ScientificName'] = data['ScientificName'].str.replace('[^\x00-\x7F]','')
    if taxonomic_filter:
        data = data[data.viral_family == taxonomic_filter]

    """hosttaxa: creating dataframe of unique hosts and their characteristics to generate nodes"""
    hosttaxa = data.groupby(['ScientificName']).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    virus_dataframe = data.groupby(['virus_name', 'viral_family']).size().reset_index().rename(columns={0:'count'})
    vlist = data.virus_name.dropna().unique().tolist()
    
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
    print ('we have '+str(df.shape[0])+' virus pairs in our model')
    
    """Creating the a network now using the df
    EDGES will be weighted according to number of shared hosts"""


    VS_unx = nx.Graph()

    """Initiating virus nodes"""

    for index, row in virus_dataframe.iterrows():
        VS_unx.add_node(row['virus_name'], type="virus",  
                    ViralFamily = str(row['viral_family']), bipartite = 1)
        
        
    #for virus in pd.unique(df[['Virus1', 'Virus2']].values.ravel()).tolist():
    #    VS_unx.add_node(virus, type="virus", virusname = virus, bipartite = 1)

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
    


def calculate_features(data_frame, network, Species_file_name, data_path, virus_df, long = False):
    print('calculate_features function is in function file 1st function')
    print ('calculating topographical features')

    

    ################################################################################################################################
    ################################################################################################################################
        
    ################################################################################################################################
    ################################################################################################################################
    print ('calculating Jaccard coefficients')
    def jaccard (c):
        return sorted(nx.jaccard_coefficient(network, [(c['Virus1'],c['Virus2'])]))[0][2]
    data_frame["jaccard"] = data_frame.apply(jaccard, axis=1)
    
    ################################################################################################################################
    ################################################################################################################################
    def hasShortestPath (c):
        return nx.has_path(network, c['Virus1'], c['Virus2'])
    data_frame["hasPath"] = data_frame.apply(hasShortestPath, axis=1)
    
    print ('calculating shortest path length')
    def ShortPathLen(c):
        if c["hasPath"]:
            return nx.shortest_path_length(network, c['Virus1'], c['Virus2'])
        else:
            return np.nan
    data_frame["ShortPathLen"] = data_frame.apply(ShortPathLen, axis=1)
    
    
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
    if long:
        

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
    print ('calculating node clusters')
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
    print ('calculating if nodes are in a same cluster')
    def in_same_cluster(c):
        if(partition[c['Virus1']] == partition[c['Virus2']]):
            return True
        else:
            return False
    data_frame["in_same_cluster"] = data_frame.apply(in_same_cluster, axis=1)      

    ################################################################################################################################
    ################################################################################################################################
    print ('calculating difference in degree')
    degree = nx.degree(network)
    def degreeDiff(c):
        return abs(degree[c['Virus1']] - degree[c['Virus2']])
    data_frame["degree_diff"] = data_frame.apply(degreeDiff, axis=1) 

    ################################################################################################################################
    ################################################################################################################################
    
    if long:
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
                        orderlist.append(IUCN.loc[IUCN['ScientificName'] == h, 'Order'].iloc[0])
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
                        orderlist.append(IUCN.loc[IUCN['ScientificName'] == h, 'Family'].iloc[0])
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
    ################################################################################################################################
    ################################################################################################################################
    print ('difference in PubMed hits')
    def PubMed_hits(c):
        return abs(c.PubMed_Search_ln1 - c.PubMed_Search_ln2)
    data_frame['PubMed_diff'] = data_frame.apply(PubMed_hits, axis=1)
    ################################################################################################################################
    ################################################################################################################################
    data_frame['hasPath'] = np.where(data_frame['hasPath']== True, 1, 0)
    data_frame['in_same_cluster'] =np.where(data_frame['in_same_cluster']== True, 1, 0)
    data_frame['FamilyMatch'] =np.where(data_frame['FamilyMatch']== 'True', 1, 0)
    data_frame['ShortPathLen'].fillna(0, inplace = True)
    data_frame['Link'] =np.where(data_frame['n_shared_hosts']>= 1, 1, 0)
    print (data_frame.shape)
    return data_frame


#######################################################################################################
#######################################################################################################

def interactive_plot(network, network_name, layout_func = 'fruchterman_reingold'):
    plot = Plot(plot_width=800, plot_height=800,
                x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    
    plot.title.text = network_name

    plot.add_tools(HoverTool( tooltips=[('','@index')]),TapTool(),
                   BoxSelectTool(), BoxZoomTool(), 
                  ResetTool(),  PanTool(), WheelZoomTool())
    if layout_func == 'fruchterman_reingold':
        graph_renderer = graphs.from_networkx(network, nx.fruchterman_reingold_layout, scale=1, center=(0,0))
        
    elif layout_func =='spring':
        graph_renderer = graphs.from_networkx(network, nx.spring_layout, scale=1, center=(0,0))
        
    elif layout_func =='circular':
        graph_renderer = graphs.from_networkx(network, nx.circular_layout, scale=1, center=(0,0))
        
    elif layout_func == 'kamada':
        graph_renderer = graphs.from_networkx(network, nx.kamada_kawai_layout, scale=1, center=(0,0))
        
    elif layout_func == 'spectral':
        graph_renderer = graphs.from_networkx(network, nx.spectral_layout, scale=1, center=(0,0))
        
    else:
        graph_renderer = graphs.from_networkx(network, nx.fruchterman_reingold_layout, scale=1, center=(0,0))
    
    centrality = nx.algorithms.centrality.betweenness_centrality(network)
    """ first element are nodes again """
    _, nodes_centrality = zip(*centrality.items())
    max_centraliy = max(nodes_centrality)
    c_centrality = [7 + 15 * t / max_centraliy
                      for t in nodes_centrality]
    
    import community #python-louvain
    partition = community.best_partition(network)
    p_, nodes_community = zip(*partition.items())
    
    community_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628',
                        '#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec',
                        '#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d',
                        '#666666']
    colors = [community_colors[t % len(community_colors)] for t in nodes_community]
    

    
    graph_renderer.node_renderer.data_source.add(c_centrality, 'centrality')
    graph_renderer.node_renderer.data_source.add(colors, 'colors')
    graph_renderer.node_renderer.glyph = Circle(size='centrality', fill_color='colors')
    graph_renderer.node_renderer.selection_glyph = Circle(size='centrality', fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#757474", line_alpha=0.2, line_width=2)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=3)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=1)

    graph_renderer.selection_policy = graphs.NodesAndLinkedEdges()
    graph_inspection_policy = graphs.NodesOnly() 
    #graph_renderer.inspection_policy = graphs.EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)

    #output_file("interactive_graphs.html")
    return plot
    
#######################################################################################################
#######################################################################################################


def get_observed_network_data(Gc, BPnx, i, data_path, virus_df, Species_file_name):
    
    IUCN = pd.read_csv(data_path+ Species_file_name,)
    IUCN["ScientificName"] = IUCN["Genus"].map(str) +' '+IUCN["Species"]
    IUCN.loc[IUCN.ScientificName== 'Homo sapiens', 'Order'] = 'Humans'
    
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
    print (net_name)
    plot_unipartite_network(
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

    Go_data = calculate_features(
        data_frame=m,
        network=Go,
        Species_file_name= Species_file_name,
        data_path=data_path,
        virus_df=virus_df)

    print("\nGenerating model data lables for 'Go'\n")
    """STEP 8"""
    model_data = Go_data[Go_data.n_shared_hosts == 0] ## Crucial step: Go_data has all pairs but those pairs from removed edges will have zero as their n_shared host along with known negatives
    model_data['label'] = np.where(model_data['n_shared_hosts_c'] > 0, 1, 0)
    model_data[
        'PubMedSeach_sum'] = model_data.PubMed_Search_ln1 + model_data.PubMed_Search_ln2
    return model_data

##########################################################################################################################
##########################################################################################################################



def construct_unipartite_virus_virus_network_order(dataframe, network_name,IUCN, 
                                             layout_func = 'fruchterman_reingold',
                                             plot= False, filter_file=  False, 
                                             taxonomic_filter = None,
                                             return_df = False):
    
    """first construct bipartite network"""
    print('this function is essenstial to generate species level sharing network but to add order data to the edges attributes')
    if filter_file:
        BPnx = construct_bipartite_host_virus_network(dataframe = dataframe, network_name= network_name, 
                                                      plot=False, filter_file= True, taxonomic_filter = taxonomic_filter)
    else:
        BPnx = construct_bipartite_host_virus_network(dataframe = dataframe, network_name= network_name, 
                                                      plot=False, filter_file= False, taxonomic_filter = taxonomic_filter)
    
    #if data_filename:
    #    """Importing all the data
    #    data: """
    #    if ".pickle" in data_filename:
    #        data = pd.read_pickle(data_filename,)
    #    else:
    #        data = pd.read_csv(data_filename, encoding='ISO-8859-1', low_memory=False)
    data = dataframe
        
    data['ScientificName'] = data['ScientificName'].str.replace('[^\x00-\x7F]','')
    if taxonomic_filter:
        data = data[data.viral_family == taxonomic_filter]

    """hosttaxa: creating dataframe of unique hosts and their characteristics to generate nodes"""
    hosttaxa = data.groupby(['ScientificName']).size().reset_index().rename(columns={0:'count'})

    """vlist: creating list of unique viruses to generate nodes"""
    virus_dataframe = data.groupby(['virus_name', 'viral_family']).size().reset_index().rename(columns={0:'count'})
    vlist = data.virus_name.dropna().unique().tolist()
    
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
    print ('we have '+str(df.shape[0])+' virus pairs in our model')
    
    def add_hosts_orders (c):
        order_list = IUCN[IUCN.ScientificName.isin(c['shared_hosts'])]['Order'].unique().tolist()
        return order_list
    df["shared_orders"] = df.apply(add_hosts_orders, axis=1)
    
    """Creating the a network now using the df
    EDGES will be weighted according to number of shared hosts"""


    VS_unx = nx.Graph()

    """Initiating virus nodes"""

    for index, row in virus_dataframe.iterrows():
        VS_unx.add_node(row['virus_name'], type="virus",  
                    ViralFamily = str(row['viral_family']), bipartite = 1)
        
        
    #for virus in pd.unique(df[['Virus1', 'Virus2']].values.ravel()).tolist():
    #    VS_unx.add_node(virus, type="virus", virusname = virus, bipartite = 1)

    """Iterating through the raw data to add Edges if a virus is found in a host"""
    for index, row in df.iterrows():
        if row['n_shared_hosts'] > 0:
            VS_unx.add_edge(row['Virus1'], row['Virus2'], weight = row['n_shared_hosts'], hosts = ','.join(row['shared_hosts']), 
                           orders = ','.join(row['shared_orders']))

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


    
    
    

    
    
    
    
    
    
    
    


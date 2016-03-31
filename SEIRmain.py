
# coding: utf-8

# In[1]:

# basic maths capabilities
import math

# more advanced maths capabilities
import numpy
import numpy.random as rnd
from random import randint
import networkx as nx
# timing how long things take
import time

# analysis and presentation
import pandas
# graphing and animation tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib import animation
import seaborn as sns

# use high-performance SVG graphics in the notebook
#get_ipython().magic(u'matplotlib inline')
#get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")


# In[2]:

class HCgraph(nx.Graph):
    def __init__(self, household_size, community_size, number_of_communities):
        print "here"
        self._household_size = household_size
        self._community_size = community_size
        self._number_of_communities = number_of_communities
        G = self.create_community(community_size)
        for i in range(1, number_of_communities):
            C = self.create_community(community_size-1)
            G = self.add_community(G, C)
        last_community = len(G.nodes())/household_size-1
        for i in range(0, household_size*(community_size-1)/household_size):
            G.add_edge(self.get_random_node(last_community), self.get_random_node(i))
        nx.Graph.__init__(self,G)

    def create_household(self):
        G=nx.complete_graph(self._household_size)
        return G

    def create_community(self, community_size):
        households = []
        for i in range(community_size):
            H =self.create_household()
            households.insert(0, H)
        G = households[0]
        for i in range(1, len(households)):
            G = nx.disjoint_union(G, households[i])
        for i in range(len(households)-1):
            for j in range(i+1, len(households)):
                G.add_edge(self.get_node_with_highest_degree(i, G), self.get_node_with_highest_degree(j, G))
        return G

    def add_community(self, G, community):
        last_community = len(G.nodes())/self._household_size-1
        G=nx.disjoint_union(G, community)
        self.get_node_with_highest_degree(last_community, G)
        for i in range(last_community+1, last_community+len(community.nodes())/self._household_size+1):
            G.add_edge(self.get_random_node(last_community), self.get_random_node(i))
        return G

    def get_random_node(self, household_number):
        low = self._household_size*household_number
        return rnd.randint(low, low+self._household_size-1)
    
    def get_node_with_highest_degree(self, household_number, G):
        low = self._household_size*household_number
        max_node = low
        max_degree = G.degree(low)
        for i in range(low, low+self._household_size-1):
            cur_degree = G.degree(i)
            if cur_degree>max_degree:
                max_degree = cur_degree
                max_node = i
        return max_node
    
    def get_nodes_in_community(self, s_node):
        intervals = {}
        household_number = s_node/self._household_size
        s_comm = s_node/((self._community_size-1)*self._household_size)
        s_from = s_comm*((self._community_size-1)*self._household_size)
        s_to = s_from+self._household_size*self._community_size-1
        #deal with overflow
        num_of_nodes = self.order()
        if s_to>num_of_nodes:
            s_to = s_to-num_of_nodes
        intervals[0] = [s_from, s_to]
        if household_number%(self._community_size-1)==0:
            if s_comm==0:
                s_from1 = num_of_nodes-self._household_size
                s_to1 = num_of_nodes-1
            else:
                s_comm1 = s_comm-1
                s_from1 = s_comm1*((self._community_size-1)*self._household_size)
                s_to1 = s_from1+self._household_size*self._community_size-1
            intervals[1] = [s_from1, s_to1]
        return intervals
    


# In[3]:

class GraphWithDynamics(HCgraph):
    '''A NetworkX undirected network with an associated dynamics. This
    class combines two sets of entwined functionality: a network, and
    the dynamical process being studied. This is the abstract base class
    for studying different kinds of dynamics.'''

    # keys for node and edge attributes
    OCCUPIED = 'occupied'     # edge has been used to transfer infection or not
    DYNAMICAL_STATE = 'seidr'   # dynamical state of a node

    def __init__( self, household_size=0, community_size=0, number_of_communities=0, g = None ):
        '''Create a graph, optionally with nodes and edges copied from
        the graph given.
        
        g: graph to copy (optional)'''
        if household_size!=0 and community_size!=0 and number_of_communities!=0:
            HCgraph.__init__(self, household_size, community_size, number_of_communities)
            print "here as well"
        else:
            nx.Graph.__init__(self, g)
            if g is not None:
                self.copy_from(g)
        
    def copy_from( self, g ):
        '''Copy the nodes and edges from another graph into us.
        
        g: the graph to copy from
        returns: the graph'''
        
        # copy in nodes and edges from source network
        self.add_nodes_from(g.nodes_iter())
        self.add_edges_from(g.edges_iter())
        
        # remove self-loops
        es = self.selfloop_edges()
        self.remove_edges_from(es)
        
        return self
    
    def remove_all_nodes( self ):
        '''Remove all nodes and edges from the graph.'''
        self.remove_nodes_from(self.nodes())

    def at_equilibrium( self, t ):
        '''Test whether the model is an equilibrium. The default runs for
        20000 steps and then stops.
        
        t: the current simulation timestep
        returns: True if we're done'''
        return (t >= 20000)

    def before( self ):
        '''Placeholder to be run ahead of simulation, Defaults does nothing.'''
        pass

    def after( self ):
        '''Placeholder to be run after simulation, Defaults does nothing.'''
        pass
    
    def _dynamics( self ):
        '''Internal function defining the way the dynamics works.

        returns: a dict of properties'''
        raise NotYetImplementedError('_dynamics()')
        
    def dynamics( self ):
        '''Run a number of iterations of the model over the network. The
        default doesn't add anything to the basic parameters captured 
        
        returns: a dict of properties'''
        return self._dynamics()

    def skeletonise( self ):
        '''Remove unoccupied edges from the network.
        
        returns: the network with unoccupied edges removed'''
        
        # find all unoccupied edges
        edges = []
        for n in self.nodes_iter():
            for (np, m, data) in self.edges_iter(n, data = True):
                if (self.OCCUPIED not in data.keys()) or (data[self.OCCUPIED] != True):
                    # edge is unoccupied, mark it to be removed
                    # (safe because there are no parallel edges)
                    edges.insert(0, (n, m))
                    
        # remove them
        self.remove_edges_from(edges)
        return self
    
    def populations( self ):
        '''Return a count of the number of nodes in each dynamical state.
        
        returns: a dict'''
        pops = dict()
        for n in self.nodes_iter():
            s = self.node[n][self.DYNAMICAL_STATE]
            if s not in pops.keys():
                pops[s] = 1
            else:
                pops[s] = pops[s] + 1
        return pops


# In[4]:

class GraphWithSynchronousDynamics(GraphWithDynamics):
    '''A graph with a dynamics that runs synchronously,
    applying the dynamics to each node in the network.'''
        
    def __init__( self, household_size=0, community_size=0, number_of_communities=0, g = None ):
        '''Create a graph, optionally with nodes and edges copied from
        the graph given.
        
        g: graph to copy (optional)'''
        GraphWithDynamics.__init__(self,household_size, community_size, number_of_communities, g)
        
    def model( self, n ):
        '''The dynamics function that's run over the network. This
        is a placeholder to be re-defined by sub-classes.
        
        n: the node being simulated
        returns: the number of events that happened in this timestep'''
        raise NotYetImplementedError('model()')
    
    def _dynamics_step( self, t ):
        '''Run a single step of the model over the network.
        
        t: timestep being simulated
        returns: the number of dynamic events that happened in this timestep'''
        events = 0
        for i in self.node.keys():
            events = events + self.model(i)
        return events

    def _dynamics( self ):
        '''Synchronous dynamics. We apply _dynamics_step() at each timestep
        and then check for completion using at_equilibrium().
        
        returns: a dict of simulation properties'''
        rc = dict()

        rc['start_time'] = time.clock()
        self.before()
        t = 0
        events = 0
        eventDist = dict()
        timestepEvents = 0
        while True:
            # run a step
            nev = self._dynamics_step(t)
            if nev > 0:
                events = events + nev
                timestepEvents = timestepEvents + 1
                eventDist[t] = nev
        
            # test for termination
            if self.at_equilibrium(t):
                break
            
            t = t + 1
        self.after()
        rc['end_time'] = time.clock()
        
        # return the simulation-level results
        rc['elapsed_time'] = rc['end_time'] - rc['start_time']
        rc['timesteps'] = t
        rc['events'] = events
        rc['event_distribution'] = eventDist
        rc['timesteps_with_events'] = timestepEvents
        return rc


# In[5]:

class SEIRSynchronousDynamics(GraphWithSynchronousDynamics):
    '''A graph with a particular SEIR dynamics. We use probabilities
    to express infection and recovery per timestep, and run the system
    using synchronous dynamics.'''
    
    # the possible dynamics states of a node for SEIR dynamics
    SUSCEPTIBLE = 'S'
    EXPOSED = 'E'
    INFECTED = 'I'
    REMOVED = 'R'
    
    # list of infected nodes, the sites of all the dynamics
    _infected = []
    
    # list of exposed nodes
    _exposed = []
    
    # list of dynamic states captured during a simulation
    _states = dict()
    
    _populations = dict()
    
    def __init__( self, household_size=0, community_size=0, number_of_communities=0,
                 beta = 0.0, gamma = 1.0, pInfected = 0.0, eta = 1.0, delta = 0.0, epsilon = 1.0, zeta = 1.0, g = None ):
        '''Generate a graph with dynamics for the given parameters.
        
        beta: infection probability (defaults to 0.0)
        gamma: probability of recovery (defaults to 1.0)
        pInfected: initial infection probability (defaults to 0.0)
        g: the graph to copy from (optional)'''
        GraphWithSynchronousDynamics.__init__(self,household_size, community_size, number_of_communities, g)
        self._beta = beta
        self._gamma = gamma
        self._pInfected = pInfected
        self._eta = eta
        self._delta = delta
        self._epsilon = epsilon
        self._zeta = zeta
            
    def before( self ):
        '''Seed the network with infected nodes, and mark all edges
        as unoccupied by the dynamics.'''
        self._infected = []       # in case we re-run from a dirty intermediate state
        self._exposed = []
        for n in self.node.keys():
            if numpy.random.random() <= self._pInfected:
                self._infected.insert(0, n)
                self.node[n][self.DYNAMICAL_STATE] = self.INFECTED
            else:
                self.node[n][self.DYNAMICAL_STATE] = self.SUSCEPTIBLE
        for (n, m, data) in self.edges_iter(data = True):
            data[self.OCCUPIED] = False

    def _dynamics_step( self, t ):
        '''Optimised per-step dynamics that only runs the dynamics at infected
        nodes, since they're the only places where state changes originate. At the
        end of each timestep we re-build the infected node list.
        
        t: timestep being simulated
        returns: the number of events that happened in this timestep'''
        events = 0
        
        # run model dynamics on all infected nodes
        for n in self._infected:
            events = events + self.model(n)
    
        # re-build the infected list if we need to
        if events > 0:
            self._infected = [ n for n in self._infected if self.node[n][self.DYNAMICAL_STATE] == self.INFECTED ]
            
#         # re-build the exposed list if we need to
#         if events > 0 and not self._exposed:
#             self._exposed = [ n for n in self._exposed if self.node[n][self.DYNAMICAL_STATE] == self.EXPOSED ]

        events1 = 0
            
        for n in self._exposed:
            events1 = events1 + self.end_latent(n)
            
        # re-build the exposed list if we need to
        if events1 > 0:
            self._exposed = [ n for n in self._exposed if self.node[n][self.DYNAMICAL_STATE] == self.EXPOSED ]
        
        events = events+events1
        # if the state has changed, capture it
        if (t == 0) or (events > 0):
            ss = dict()
            for n in self.nodes_iter():
                ss[n] = self.node[n][self.DYNAMICAL_STATE]
            self._states[t] = ss
            self._populations[t] = self.populations()
            
        return events
    
    def model( self, n ):
        '''Apply the SEIR dynamics to node n. From the re-definition of _dynamics_step()
        we already know this node is infected.

        n: the node
        returns: the number of changes made'''
        events = 0
        
        # infect susceptible neighbours with probability beta
        for (_, m, data) in self.edges_iter(n, data = True):
            if self.node[m][self.DYNAMICAL_STATE] == self.SUSCEPTIBLE:
                if numpy.random.random() <= self._beta:
                    events = events + 1
                    
                    # infect the node
                    self.node[m][self.DYNAMICAL_STATE] = self.EXPOSED
                    self._exposed.insert(0, m)
                        
                    # label the edge we traversed as occupied
                    data[self.OCCUPIED] = True
    
        # recover with probability gamma
        if numpy.random.random() <= self._gamma:
            # recover the node
            events = events + 1
            self.node[n][self.DYNAMICAL_STATE] = self.REMOVED
              
        return events
    
    def end_latent( self, n ):
        '''Apply the SEIR dynamics to node n. From the re-definition of _dynamics_step()
        we already know this node is exposed.

        n: the node
        returns: the number of changes made'''
        events = 0

        # end latent perioud with probability eta
        if numpy.random.random() <= self._eta:
            # make the node infected
            events = events + 1
            self.node[n][self.DYNAMICAL_STATE] = self.INFECTED
            self._infected.insert(0, n)
        return events
    
    def at_equilibrium( self, t ):
        '''SEIR dynamics is at equilibrium if there are no more
        infected nodes left in the network or if we've exceeded
        the default simulation length.
        
        returns: True if the model has stopped'''
        if t >= 20000:
            return True
        else:
            return ((len(self._infected) == 0) and len(self._exposed)==0)
            
    def dynamics( self ):
        '''Returns statistics of outbreak sizes. This skeletonises the
        network, so it can't have any further dynamics run on it.
        
        returns: a dict of statistical properties'''
        
        # run the basic dynamics
        rc = self._dynamics()
        
        # compute the limits and means
        cs = sorted(nx.connected_components(self.skeletonise()), key = len, reverse = True)
        max_outbreak_size = len(cs[0])
        max_outbreak_proportion = (max_outbreak_size + 0.0) / self.order()
        mean_outbreak_size = numpy.mean([ len(c) for c in cs ])
        
        # add parameters and metrics for this simulation run
        rc['pInfected' ] = self._pInfected,
        rc['gamma'] = self._gamma,
        rc['beta'] = self._beta,
        rc['eta'] = self._eta,
        rc['N'] = self.order(),
        rc['mean_outbreak_size'] = mean_outbreak_size,
        rc['max_outbreak_size'] = max_outbreak_size,
        rc['max_outbreak_proportion'] = max_outbreak_proportion
        rc['evolution'] = self._states
        rc['populations'] = self._populations
        return rc


# In[6]:

def show_changes(G, syn_dyn):
    pos=nx.spring_layout(G) # positions for all nodes
    # print G.nodes()
    states = syn_dyn['evolution']
    susceptible_nodes_time = {}
    infected_nodes_time = {}
    exposed_nodes_time = {}
    removed_nodes_time = {}
    changes = sorted(states.keys())
    for t in changes:
        susceptible_nodes = []
        infected_nodes = []
        exposed_nodes = []
        removed_nodes = []
        for node in states[t]:
            if states[t][node]=='S':
                susceptible_nodes.insert(0, node)
            else:
                if states[t][node]=='I':
                    infected_nodes.insert(0, node)
                else:
                    if states[t][node]=='E':
                        exposed_nodes.insert(0, node)
                    else:
                        if states[t][node]=='R':
                            removed_nodes.insert(0, node)
                        else:
                            print "Error"
        susceptible_nodes_time[t] = susceptible_nodes
        infected_nodes_time[t] = infected_nodes
        exposed_nodes_time[t] = exposed_nodes
        removed_nodes_time[t] = removed_nodes

    fig = plt.figure(figsize=(10,100))
    plot_number = 1
    node_size = 100
    alpha = 0.8
    for t in changes:
        plt.subplot(len(changes)+1,1,plot_number)
        plot_number=plot_number+1
        nx.draw_networkx_nodes(G,pos, ax=None,nodelist=susceptible_nodes_time[t],
                               node_color='y',
                               node_size=node_size,
                           alpha=alpha)
        nx.draw_networkx_nodes(G,pos,ax=None,nodelist=infected_nodes_time[t],
                               node_color='r',
                               node_size=node_size,
                           alpha=alpha)
        nx.draw_networkx_nodes(G,pos,ax=None,nodelist=exposed_nodes_time[t],
                               node_color='b',
                               node_size=node_size,
                           alpha=alpha)
        nx.draw_networkx_nodes(G,pos,ax=None,nodelist=removed_nodes_time[t],
                               node_color='orange',
                               node_size=node_size,
                           alpha=alpha)
        nx.draw_networkx_edges(G,pos,ax=None, width=1.0,alpha=0.5)
        nx.draw_networkx_labels(G,pos)
        plt.title('t = {0}'.format(t))
        plt.axis('off')
    # fig.tight_layout()
    plt.show()


# In[7]:

delta = 0.25
epsilon = 0.3
zeta = 0.1
household_size =5
community_size = 10
number_of_communities = 100
number_of_nodes = 5000
p_edge_creation = 0.002
# syn = SEIDRSynchronousDynamics(household_size, community_size, number_of_communities, pInfected = 0.01,
#                                           beta = 0.128, gamma = 0.01038, eta = 0.01, 
#                                           delta =delta, epsilon = epsilon, zeta = zeta)
syn = SEIRSynchronousDynamics(pInfected = 0.00136557,
                                          beta = 0.3151, gamma = 0.06851662, eta = 0.083333, 
                                          g = nx.erdos_renyi_graph(number_of_nodes, p_edge_creation))
syn_dyn = syn.dynamics()


# In[9]:

import io
import os
SEPARATOR = ', '
file_num = 1
if os.path.isfile('seir-experiment'+str(file_num)+'.csv'):
    file = open('seir-experiment'+str(file_num)+'.csv', 'a')
else:
    file = open('seir-experiment'+str(file_num)+'.csv', 'w')
#file.write('p_edge_creation, p_infected, gamma, beta, delta, epsilon, zeta, eta, N, elapsed_time, timesteps, events, timesteps_with_events,')
#file.write('mean_outbreak_size, max_outbreak_size, max_outbreak_proportion, exposed_from_infected, exposed_from_dead, rewire_degree\n')
file.write(str(p_edge_creation)+ SEPARATOR + str(syn_dyn['pInfected' ][0]) + SEPARATOR + str(syn_dyn['gamma'][0]) + SEPARATOR + str(syn_dyn['beta'][0])+ 
           SEPARATOR + str(syn_dyn['eta'][0]) + SEPARATOR + str(syn_dyn['N'][0]) + SEPARATOR + str(syn_dyn['elapsed_time'])+ 
           SEPARATOR + str(syn_dyn['timesteps']) + SEPARATOR + str(syn_dyn['events']) + SEPARATOR + str(syn_dyn['timesteps_with_events']) +
           SEPARATOR + str(syn_dyn['mean_outbreak_size'][0]) + SEPARATOR + str(syn_dyn['max_outbreak_size'][0]) + 
           SEPARATOR + str(syn_dyn['max_outbreak_proportion'])+'\n')
file.close()

if os.path.isfile('seir-experiment'+str(file_num)+'_1.csv'):
    file1 = open('seir-experiment'+str(file_num)+'_1.csv', 'a')
else:
    file1 = open('seir-experiment'+str(file_num)+'_1.csv', 'w')
event_distr = syn_dyn['event_distribution']
changes = sorted(event_distr.keys())
timesteps = ''
events = ''
for i in changes:
    timesteps = timesteps +  str(i) + SEPARATOR
    events = events +  str(event_distr[i]) + SEPARATOR
timesteps = 'timesteps, ' + timesteps[:-2] + '\n'
events = 'events, '+ events[:-2] + '\n'
populations =  syn_dyn['populations']
sus = ''
exp = ''
inf = ''
rem = ''
for t in changes:
    s = 0
    e = 0
    i = 0
    r = 0
    for j in populations[t]:
        if j=='S':
            s = populations[t][j]
        else: 
            if j=='E':
                e = populations[t][j]
            else:
                if j=='I':
                    i = populations[t][j]
                else:
                    if j=='R':
                        r = populations[t][j]
    sus = sus +  str(s) + SEPARATOR
    exp = exp +  str(e) + SEPARATOR
    inf = inf +  str(i) + SEPARATOR
    rem = rem +  str(r) + SEPARATOR
sus = 'sus, '+sus[:-2] + '\n'
exp = 'exp, '+exp[:-2] + '\n'
inf = 'inf, '+inf[:-2] + '\n'
rem = 'rem, '+rem[:-2] + '\n'

file1.write(timesteps+events+sus+exp+inf+rem)
file1.close()


# In[ ]:




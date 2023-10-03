import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class PerishableInventorySupplyChain(gym.Env):

    """ 
    PerishableInventorySupplyChain
    lead time is set to the same number for all the facilities.
    demand: FIFO
    """

    def __init__(self, sc_network, number_of_layers, warming_up=50, num_periods=365):
        super(PerishableInventorySupplyChain, self).__init__()
        
        self.warming_up = warming_up
        self.num_periods = num_periods + warming_up
        self.total_lost_sale = np.zeros((self.num_periods, ))
        
        self.lifetime = 10
        self.max_lead_time = 1
        self.max_quantity = 200
        self.sigma_leadtime = 0.3

        self.order_cost = 2
        self.perish_cost = 2
        self.underage_cost = 20
        self.holding_cost = 1
        self.transport_cost = 1
        self.price = 0

        self.initial_inventory = 0
        self.initial_transport = 0

        self.number_of_layers = number_of_layers

        self.sc = sc_network

        self.number_of_facilities = self.sc.number_of_nodes()

        # get elements in each layer
        self.layers = [[] for i in range(self.number_of_layers)]

        for i in range(self.number_of_facilities):
            temp = self.sc.nodes[i]['layer']
            self.layers[temp].append(i) # [[0], [1, 2], [3, 4, 5, 6], [7, 8, 9, 10]]

        # main facilities: not invlude the first layer (supplier) and the last layer (retailer)
        self.main_facilities = self.layers[1:-1]
        self.number_of_main_facilities = sum(
            len(i) for i in self.main_facilities)

        # order edges: supplier -> distributer -> retailer
        # get edge list [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        self.order_list = [
            e for e in self.sc.edges if 'Lead_time' in self.sc.edges[e]]  
        self.number_order_list = len(self.order_list)
        
        # supply edges and distributor edges
        # [(0, 1), (0, 2)]
        self.supply_edges = [e for e in self.sc.edges if e[0] in self.layers[0]]
        # [(1, 3), (1, 4), (2, 5), (2, 6)]
        self.distri_edges = [e for e in self.order_list if e not in self.supply_edges]

        # retail edges: retailer->customer
        self.retail_edges = [e for e in self.sc.edges(self.layers[-2])]
        self.demand = [0] * len(self.retail_edges)
        self.demand = dict(zip(self.retail_edges, self.demand))
        # self.retailer_list = [x[0]-1 for x in self.retail_edges]
        # self.customer_list = [x[1]-1 for x in self.retail_edges]
        
        # number of supplier
        self.number_of_supplier = len(self.layers[0])

        # record
        self.onhand_inventory = np.zeros((self.number_of_main_facilities, self.num_periods))
        self.inventory_position = np.zeros((self.number_of_main_facilities, self.num_periods))

        self.reset()

        self.obs_dim = len(self.inventory_state.flatten()) + \
            len(self.transport_state.flatten())

        # action range
        action_low = np.zeros(self.number_order_list)
        action_high = np.ones(self.number_order_list) * (self.max_quantity)
        self.range = action_high - action_low

        # action 
        self.action_space = spaces.Box(low=-np.ones(self.number_order_list, dtype=np.float32),
                                       high=np.ones(self.number_order_list, dtype=np.float32),
                                       dtype=np.float32)
        # obs
        self.observation_space = spaces.Box(low=np.ones(self.obs_dim)*np.iinfo(np.int32).min,
                                            high=np.ones(
                                                self.obs_dim)*np.iinfo(np.int32).max,
                                            dtype=np.int32)

    def step(self, action):
        
        next_inventory = self.inventory_state.copy()
        # age inventory
        perished_inventory = next_inventory[:,-1]
        next_inventory = np.delete(next_inventory, -1, 1) # delete the last column
        next_inventory = np.append(np.zeros((self.number_of_main_facilities,1), dtype=np.int32), next_inventory, axis=1) # add one column
        total_inventory = np.sum(next_inventory)

        # age transport inventory
        perished_transport = self.transport_state[:, :, -1]
        next_transport = self.transport_state.copy()
        next_transport = np.delete(next_transport, -1, 2) # delete the last column
        next_transport = np.append(np.zeros((self.number_order_list,self.max_lead_time,1), dtype=np.int32), next_transport, axis=2) # add one column
        total_transport = np.sum(next_transport)

        # update transportation
        next_inventory = next_transport[:,0,:] + next_inventory
        self.inventory_state = next_inventory.copy()

        # generate demand
        for i in range(len(self.retail_edges)):
            e = self.retail_edges[i]
            mu = self.sc.get_edge_data(*e)['mu_demand']
            sigma = self.sc.get_edge_data(*e)['sigma_demand']
            self.demand[e] = np.around(np.random.normal(mu, sigma))

        # meet demand (FIFO)      
        number_of_lost_sale = 0
        for (i, j), d in self.demand.items():
            for k in range(self.lifetime):
                if d == 0: break
                next_inventory[i-self.number_of_supplier, -k-1] = max(self.inventory_state[i-self.number_of_supplier, -k-1] - d, 0)
                d = max(d - self.inventory_state[i-self.number_of_supplier, -k-1], 0)
            number_of_lost_sale = number_of_lost_sale + d

        # convert to action to dict
        action = self.convert_action(action)
        action_dict = {key: action[i] for i, key in enumerate(self.order_list)}
        
        # make order, inventory - action, FIFO
        modified_action = np.zeros((self.number_order_list,self.lifetime))
        modified_action_temp = np.zeros((self.number_order_list,self.lifetime))
        for idx, ((i, j), q) in enumerate(action_dict.items()): # i: source, j: target, q: quantity           
            if i in self.layers[0]: # supplier has untimate inventory
                modified_action[idx, 0] = q
                continue
            elif q == 0: continue
            elif q <= next_inventory[i-self.number_of_supplier, -1]:
                modified_action[idx, -1] = q
                next_inventory[i-self.number_of_supplier, -1] = next_inventory[i-self.number_of_supplier, -1] - q
            elif q >= np.sum(next_inventory[i-self.number_of_supplier, :]): # when quantity > total inventory
                modified_action[idx, :] = next_inventory[i-self.number_of_supplier, :]
                next_inventory[i-self.number_of_supplier, :] = next_inventory[i-self.number_of_supplier, :] - modified_action[idx, :]
                continue
            for k in range(self.lifetime):
                if q > np.sum(next_inventory[i-self.number_of_supplier, -k-1:]): continue
                modified_action_temp[idx, -k:] = next_inventory[i-self.number_of_supplier, -k:]
                modified_action_temp[idx, -k-1] = q - np.sum(next_inventory[i-self.number_of_supplier, -k:])
                next_inventory[i-self.number_of_supplier] = next_inventory[i-self.number_of_supplier] - modified_action_temp[idx]
                modified_action = modified_action + modified_action_temp
                modified_action_temp = np.zeros((self.number_order_list,self.lifetime))
                break
       
        # make order, transportation + action
        next_transport = np.append(next_transport, np.zeros((self.number_order_list,1,self.lifetime), dtype=np.int32), axis=1) # add one row
        next_transport[:,-1,:] = modified_action + next_transport[:,-1,:] #!!!

        # update transportation 
        next_transport = np.delete(next_transport, 0, 1) # delete the first row

        # service level, number of perished products, 
        
        # cost: order cost, perish cost, underage cost, holding cost
        total_order_cost = np.sum(action) * self.order_cost
        total_perish_cost = (np.sum(perished_inventory) + np.sum(perished_transport)) * self.perish_cost
        total_underage_cost = number_of_lost_sale * self.underage_cost
        total_holding_cost = total_inventory * self.holding_cost
        total_transportation_cost = total_transport * self.transport_cost
        total_revenue = (sum(self.demand.values()) - number_of_lost_sale) * self.price
        reward = -(total_order_cost + total_perish_cost + total_underage_cost + total_holding_cost + total_transportation_cost) + total_revenue

        # lost sales
        self.total_lost_sale[self.periods] = number_of_lost_sale

        # update state
        self.inventory_state = next_inventory
        self.transport_state = next_transport
        self.state = np.hstack((self.inventory_state.flatten() , self.transport_state.flatten()))

        # transportation to dict, expected transport
        transport_dict = {key: self.transport_state[i,:,:] for i, key in enumerate(self.order_list)}
        self.expected_transport = np.zeros((self.number_order_list, ))
        for ((i, j), q) in transport_dict.items():
            self.expected_transport[j-1] = np.sum(q) + self.expected_transport[j-1]

        # record
        self.onhand_inventory[:, self.periods] = np.sum(self.inventory_state, axis=1)
        self.inventory_position[:, self.periods] = np.sum(self.inventory_state, axis=1) + np.sum(np.sum(self.transport_state, axis=1),axis=1)

        # update period
        self.periods = self.periods + 1
        if self.periods < self.num_periods:
            done = False
        else:
            done = True

        return self.state, reward, done, {}

    def reset(self):

        self.periods = 0
        
        # inventory (number of node, lifetime)
        # example
        #     0   1  2  3  4  5  6  7  8  9
        # 1 [[10 10 10 10 10 10 10 10 10 10] 
        # 2  [10 10 10 10 10 10 10 10 10 10]
        # 3  [10 10 10 10 10 10 10 10 10 10]
        # 4  [10 10 10 10 10 10 10 10 10 10]
        # 5  [10 10 10 10 10 10 10 10 10 10]
        # 6  [10 10 10 10 10 10 10 10 10 10]]
        self.inventory_state = np.ones(
            (self.number_of_main_facilities, self.lifetime), dtype=np.int32) * self.initial_inventory
        
        # transport (edge, leadtime, lifetime)
        # edges [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        # example (one edge)
        #     1 2 3 4 5 6 7 8 9 10
        # 1 [[2 2 2 2 2 2 2 2 2 2]
        # 2  [2 2 2 2 2 2 2 2 2 2]
        # 3  [2 2 2 2 2 2 2 2 2 2]]
        self.transport_state = np.ones(
            (self.number_order_list, self.max_lead_time, self.lifetime), dtype=np.int32) * self.initial_transport
        self.state = np.hstack((self.inventory_state.flatten() , self.transport_state.flatten()))

        self.expected_transport = np.zeros((self.number_order_list, ))
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def convert_action(self, action):
        action = np.around((action + 1) * self.range /2)
        action = np.array(action, dtype=np.int32)
        return action
    
    def normalize_action(self, action):
        action = ((action / self.range) * 2) - 1
        return action
 
    def plot_sc_net(self):
        pos = nx.multipartite_layout(self.sc, subset_key="layer")
        pos = {key: (x*2, y) for key, (x, y) in pos.items()}
        plt.figure(figsize=(8, 8))
        nx.draw(self.sc, pos, with_labels=True, node_color='lightgrey', node_size=1000)
        plt.axis("equal")
        plt.show()

    def plot_inventory_record(self):
        # fig, axs = plt.subplots(self.number_of_main_facilities, sharex=True, sharey=True,figsize=[10, 15])
        fig, axs = plt.subplots(self.number_of_main_facilities, sharex=True, sharey=True,figsize=[10, 7])
        for i in range(self.number_of_main_facilities):
            axs[i].plot(self.onhand_inventory[i,self.warming_up:],'--',dashes=(10, 1),label='On-hand inventory')
            axs[i].plot(self.inventory_position[i,self.warming_up:],label='Inventory position')
            axs[i].set_xlim(0, 365)
            axs[i].set_title('Node ' + str(i+1))
            # axs[i].set(ylabel='Node ' + str(i+1))

        for ax in axs.flat:
            # ax.set(xlabel='Time (day)', ylabel='Inventory')
            ax.set(xlabel='Time (day)')
            
            # legend
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', ncol=2)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        
        plt.tight_layout()
        # plt.show()
        plt.savefig("sample.svg")

    def plot_lost_sales(self):
        plt.plot(self.total_lost_sale)
        plt.show()

    def ss_policy(self, order_policy):
        '''order_policy: (s1, s2, ..., S1, S2, ...)'''
        n_facility = self.number_of_main_facilities

        # decoding
        reorder_value = order_policy[0:n_facility ]
        order_up_to_value = order_policy[n_facility:2*n_facility]

        # convert
        reorder_value = self.convert_action(reorder_value)
        order_up_to_value = self.convert_action(order_up_to_value)

        # get policy
        total_inventory = np.sum(self.inventory_state, axis=1) + self.expected_transport
        action = np.where(total_inventory<=reorder_value, order_up_to_value-total_inventory, 0)

        # normalize
        action = self.normalize_action(action)
        return action
    
    def bsp_low_policy(self, order_policy):
        '''order_policy: (b, b, ..., S2, S2, S2, ..., S1, S1, ...)'''
        n_facility = self.number_of_main_facilities

        # decoding
        break_point = order_policy[0:n_facility ]
        s2 = order_policy[n_facility:2*n_facility]
        s1 = order_policy[2*n_facility:3*n_facility]

        # convert
        break_point = np.maximum(self.convert_action(break_point), 1)
        s2 = self.convert_action(s2)
        s1 = self.convert_action(s1)
        alpha = (s1 - s2 + break_point)/break_point

        # get policy
        total_inventory = np.sum(self.inventory_state, axis=1) + self.expected_transport
        action = np.where(total_inventory<break_point, s1-np.around(alpha*total_inventory), s2-total_inventory)
        action = np.maximum(0, action)

        # normalize
        action = self.normalize_action(action)
        return action

if __name__ == '__main__':
    case = 2
    if case == 1:
        sc = nx.DiGraph()
        sc.add_node(0, layer=0)  # supplier

        sc.add_node(1, layer=1)  # distributer
        sc.add_node(2, layer=1)  # distributer

        sc.add_node(3, layer=2)  # retailer
        sc.add_node(4, layer=2)  # retailer
        sc.add_node(5, layer=2)  # retailer
        sc.add_node(6, layer=2)  # retailer

        sc.add_node(7, layer=3)  # customer
        sc.add_node(8, layer=3)  # customer
        sc.add_node(9, layer=3)  # customer
        sc.add_node(10, layer=3) # customer

        # supplier to distributer
        sc.add_edge(0, 1, Lead_time=1)
        sc.add_edge(0, 2, Lead_time=2)

        # distributer to retailer
        sc.add_edge(1, 3, Lead_time=1)
        sc.add_edge(1, 4, Lead_time=2)
        sc.add_edge(2, 5, Lead_time=2)
        sc.add_edge(2, 6, Lead_time=1)

        # retailer to customer
        sc.add_edge(3, 7, mu_demand=2, sigma_demand=0.7)
        sc.add_edge(4, 8, mu_demand=4, sigma_demand=1)
        sc.add_edge(5, 9, mu_demand=4, sigma_demand=1.5)
        sc.add_edge(6, 10, mu_demand=6, sigma_demand=1.5)

    if case == 2:

        sc = nx.DiGraph()

        sc.add_node(0, layer=0)  # supplier

        sc.add_node(1, layer=1)  # distributer
        sc.add_node(2, layer=1)  
        sc.add_node(3, layer=1)  
        sc.add_node(4, layer=1)  

        # retailer
        sc.add_node(5, layer=2)  
        sc.add_node(6, layer=2)  
        sc.add_node(7, layer=2)  
        sc.add_node(8, layer=2)  
        sc.add_node(9, layer=2)  
        sc.add_node(10, layer=2) 
        sc.add_node(11, layer=2) 
        sc.add_node(12, layer=2) 

        # customer
        sc.add_node(13, layer=3)
        sc.add_node(14, layer=3)
        sc.add_node(15, layer=3)
        sc.add_node(16, layer=3)
        sc.add_node(17, layer=3)
        sc.add_node(18, layer=3)
        sc.add_node(19, layer=3)
        sc.add_node(20, layer=3)


        # supplier to distributer
        sc.add_edge(0, 1, Lead_time=1)
        sc.add_edge(0, 2, Lead_time=2)
        sc.add_edge(0, 3, Lead_time=2)
        sc.add_edge(0, 4, Lead_time=2)

        # distributer to retailer
        sc.add_edge(1, 5, Lead_time=1)
        sc.add_edge(1, 6, Lead_time=2)
        sc.add_edge(1, 7, Lead_time=2)

        sc.add_edge(2, 8, Lead_time=1)
        sc.add_edge(2, 9, Lead_time=1)

        sc.add_edge(3, 10, Lead_time=1)

        sc.add_edge(4, 11, Lead_time=1)
        sc.add_edge(4, 12, Lead_time=1)

        # retailer to customer
        sc.add_edge(5, 13, mu_demand=2, sigma_demand=0.7)
        sc.add_edge(6, 14, mu_demand=4, sigma_demand=1)
        sc.add_edge(7, 15, mu_demand=4, sigma_demand=1.5)
        sc.add_edge(8, 16, mu_demand=6, sigma_demand=1.5)
        sc.add_edge(9, 17, mu_demand=2, sigma_demand=0.7)
        sc.add_edge(10, 18, mu_demand=4, sigma_demand=1)
        sc.add_edge(11, 19, mu_demand=4, sigma_demand=1.5)
        sc.add_edge(12, 20, mu_demand=6, sigma_demand=1.5)

    env = PerishableInventorySupplyChain(sc_network=sc,number_of_layers=4,warming_up=50, num_periods=365)

    obs = env.reset()

    total_reward = 0
    env.plot_sc_net()

    num_timesteps = 365
    for t in range(num_timesteps):

        random_action = env.action_space.sample()

        next_state, reward, done, info = env.step(random_action)
        
        total_reward = total_reward + reward
        
        # print(env.inventory_state)

        if done:
            print(total_reward)
            env.plot_inventory_record()
            break

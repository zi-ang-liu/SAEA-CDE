from PerishableInventorySupplyChain import PerishableInventorySupplyChain
import networkx as nx

def evaluate(x, policy, record=False, warming_up=50, num_periods=365):
    case = 1

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

    env = PerishableInventorySupplyChain(sc_network=sc,number_of_layers=4,warming_up=warming_up, num_periods=num_periods)
    env.reset()
    done = False
    total_reward = 0

    dispatcher={'ss_policy':env.ss_policy, 'bsp_low_policy':env.bsp_low_policy}

    def call_func(x, func):
        try:
            return dispatcher[func](x)
        except:
            return "Invalid function"

    for i in range(env.warming_up):
        action = call_func(x, policy)
        next_state, reward, done, info = env.step(action)
    
    while not done:
        action = call_func(x, policy)
        next_state, reward, done, info = env.step(action)
        total_reward = total_reward + reward
    
    if record == True:
        env.plot_inventory_record()
        # env.plot_lost_sales()
    
    return -total_reward    
    
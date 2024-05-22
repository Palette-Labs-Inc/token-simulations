import networkx as nx

import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np

from nosh.buyer_agent import BuyerAgent
from nosh.producer_agent import SellerAgent

class NoshGraphSimulation:
    def __init__(
            self, 
            initial_buyers_kwargs, 
            initial_sellers_kwargs, 
            new_buyer_kwargs,
            new_seller_kwargs,
            min_buyers=2, 
            min_sellers=2, 
            seed=None, 
            weight_update_fn=None
    ):
        self.graph = nx.Graph()
        self.buyer_agents = {}
        self.seller_agents = {}
        self.min_buyers = min_buyers
        self.min_sellers = min_sellers
        self.rng = np.random.default_rng(seed)
        
        self.graph_evolution_metrics = []
        self.new_buyer_kwargs = new_buyer_kwargs
        self.new_seller_kwargs = new_seller_kwargs

        self.current_time_step = 0
        self.total_supply = 0
        self.buyer_agent_id_counter = 1
        self.seller_agent_id_counter = 1

        if weight_update_fn is None:
            self.compute_updated_weight = self.default_weight_update_fn
        else:
            self.compute_updated_weight = weight_update_fn

        for buyer_kwargs in initial_buyers_kwargs:
            buyer_agent = BuyerAgent(
                agent_id=self.buyer_agent_id_counter, 
                rng=self.rng,
                **buyer_kwargs
            )
            self.buyer_agents[buyer_agent.agent_id] = buyer_agent
            self.graph.add_node(buyer_agent, bipartite=0, type='buyer')
            self.buyer_agent_id_counter += 1

        for seller_kwargs in initial_sellers_kwargs:
            seller_agent = SellerAgent(
                agent_id=self.seller_agent_id_counter, 
                rng=self.rng,
                **seller_kwargs
            )
            self.seller_agents[seller_agent.agent_id] = seller_agent
            self.graph.add_node(seller_agent, bipartite=1, type='seller')
            self.seller_agent_id_counter += 1

    def default_weight_update_fn(self, current_weight, transaction_value):
        return current_weight + transaction_value
    
    def evolve_graph(self, add_delete_maintain_probs):
        add_prob, delete_prob, maintain_prob = add_delete_maintain_probs
        
        # Add or delete nodes
        prob = self.rng.random()
        if prob < add_prob:
            self.add_node()
        elif prob < add_prob + delete_prob:
            self.delete_node()

        # Simulate transactions and update weights
        # for each buyer and seller, determine if they will transact:
        for buyer_agent in self.buyer_agents.values():
            for seller_agent in self.seller_agents.values():
                if buyer_agent.will_transact() and seller_agent.will_transact():
                    transaction_value = buyer_agent.make_transaction([seller_agent.agent_id])
                    if transaction_value:
                        edge = (buyer_agent, seller_agent)
                        if self.graph.has_edge(*edge):
                            self.graph.edges[edge]['weight'] = self.compute_updated_weight(self.graph.edges[edge]['weight'], transaction_value)
                        else:
                            self.graph.add_edge(*edge, weight=self.compute_updated_weight(0, transaction_value))

        # simulate whether any sellers exchange some graph weight for tokens
        for seller_agent in self.seller_agents.values():
            if seller_agent.will_unstake():
                # get all the buyers that the seller has an edge with and associated weights in a dictionary
                # NOTE: edges are added in the format (buyer, seller, weight=...)
                #       However, when querying the graph for edges, the format is
                #       (querying_node, found_node, data)
                #       This isn't documented clearly (atleast to me) in NetworkX, so
                #       to be safe, I check the returned edges and pop the querying node
                #       from the tuple to ensure I got the counterparty node
                buyer_edges = self.graph.edges(seller_agent, data=True)
                
                buyer2weight = {}
                for query_agent, counterparty_agent, edge_data in buyer_edges:
                    # Read the note above - this is extra overhead to ensure that
                    # we get the correct counterparty node
                    if query_agent == seller_agent:
                        buyer_agent = counterparty_agent
                    elif counterparty_agent == seller_agent:
                        buyer_agent = query_agent
                    # get the transaction value from the edge data
                    buyer2weight[buyer_agent] = edge_data['weight']
                if len(buyer2weight) > 0:
                    buyer2burn = seller_agent.get_unstake_amount(buyer2weight)
                    # buyer2burn is a dictionary of how much to burn from each buyer that this seller is connected to
                    total_burned = 0
                    for buyer_to_burn, amount in buyer2burn.items():
                        burn_amount = amount
                        self.graph.edges[(buyer_to_burn, seller_agent)]['weight'] -= burn_amount
                        
                        total_burned += burn_amount
                    
                    # update the token balance of the seller
                    tokens_added = self.compute_minting(seller_agent, total_burned)
                    # The seller exchanges some graph weight for tokens
                    # TODO: need to have a callback to update the seller's token balance and compute it in the correct way
                    seller_agent.num_tokens += tokens_added
                    self.total_supply += tokens_added

    def add_node(self):
        
        # determine whether to add a buyer or seller node
        if self.rng.random() < 0.5:
            new_agent = BuyerAgent(
                agent_id=self.buyer_agent_id_counter, 
                rng=self.rng,
                **self.new_buyer_kwargs
            )
            self.graph.add_node(new_agent, bipartite=0, type='buyer')
            self.buyer_agents[new_agent.agent_id] = new_agent
            self.buyer_agent_id_counter += 1
        else:
            new_agent = SellerAgent(
                agent_id=self.seller_agent_id_counter, 
                rng=self.rng,
                **self.new_seller_kwargs
            )
            self.graph.add_node(new_agent, bipartite=1, type='seller')
            self.seller_agents[new_agent.agent_id] = new_agent
            self.seller_agent_id_counter += 1

    def delete_node(self):
        # Determine randomly whether to delete a buyer or seller node
        delete_buyer = self.rng.random() < 0.5

        if delete_buyer:
            possible_nodes_to_remove = list(self.buyer_agents.values())
            if len(possible_nodes_to_remove) < self.min_buyers:
                return
        else:
            possible_nodes_to_remove = list(self.seller_agents.values())
            if len(possible_nodes_to_remove) < self.min_sellers:
                return

        node_to_remove = self.rng.choice(possible_nodes_to_remove)
        self.graph.remove_node(node_to_remove)
        if delete_buyer:
            self.buyer_agents.pop(node_to_remove.agent_id, None)
        else:
            self.seller_agents.pop(node_to_remove.agent_id, None)

    def visualize(self, ax):
        # Create a layout for the bipartite graph
        buyer_nodes = [node for node, data in self.graph.nodes(data=True) if data['bipartite'] == 0]
        pos = nx.bipartite_layout(self.graph, buyer_nodes)
        
        # Extract agent IDs for labeling
        agent_labels = {node: node.agent_id for node in self.graph.nodes()}
        
        # Extract edge weights for labeling
        edge_labels = {(u, v): self.graph.edges[u, v]['weight'] for u, v in self.graph.edges()}
        
        # Get the nodes involved in edges
        nodes_in_edges = set(sum(self.graph.edges(), ()))

        # Filter out nodes without edges
        nodes_to_draw = [node for node in self.graph.nodes() if node in nodes_in_edges]

        # Draw the graph with agent IDs as labels and edge weights as labels
        nx.draw(self.graph, pos, ax=ax, nodelist=nodes_to_draw, labels=agent_labels, node_color='skyblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=ax, font_color='red')
        ax.set_title("Bipartite Graph Visualization")

    def run(self, num_time_steps, add_delete_maintain_probs, create_video=False):
        frame_folder = 'frames'
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder, exist_ok=True)

        for ii in range(num_time_steps):
            self.record_graph_metrics()
            self.evolve_graph(add_delete_maintain_probs)
            self.current_time_step += 1

            if create_video:
                fig, ax = plt.subplots(figsize=(10, 6))
                self.visualize(ax)
                frame_filename = os.path.join(frame_folder, f'frame_{ii:03d}.png')
                plt.savefig(frame_filename)
                
                # Close the figure to release resources
                plt.close(fig)
            
        if create_video:
            # Call FFmpeg to combine frames into a video
            subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', f'{frame_folder}/frame_%03d.png', '-c:v', 'libx264', '-r', '30', '-pix_fmt', 'yuv420p', 'output.mp4'])
            print("Video created: output.mp4")

        return self.graph_evolution_metrics

    def compute_minting(self, producer_agent, total_burned):
        # TODO: update based on the function we choose
        return total_burned

    def compute_seller2buyer_weights(self):
        """
        Get a dictionary mapping each seller's ID to a dictionary of buyer IDs and the weight of the edge connecting them.
        """
        seller_buyer_weights = {}

        for seller_agent in self.seller_agents.values():
            seller_id = seller_agent.agent_id
            seller_buyer_weights[seller_id] = {}

            for buyer_agent in self.buyer_agents.values():
                buyer_id = buyer_agent.agent_id
                edge_weight = self.graph.get_edge_data(seller_agent, buyer_agent, default={'weight': 0})['weight']
                seller_buyer_weights[seller_id][buyer_id] = edge_weight

        return seller_buyer_weights
    
    def record_graph_metrics(self):
        # TODO: implement this function
        # for now, the value is defined as the sum of the eigenvector centrality and the total weight of the edges for a particular seller node
        
        # compute eigenvector centrality for all the nodes in the graph
        try:
            ec_full = nx.eigenvector_centrality(self.graph)
        except nx.PowerIterationFailedConvergence:
            ec_full = {node: 0 for node in self.graph.nodes()}
        

        seller_agents = list(self.seller_agents.values())
        seller2value = {}
        seller2weight = {}
        seller2totalweight = {}
        ec = {}
        for seller_agent in seller_agents:
            buyer_edges = self.graph.edges(seller_agent, data=True)
            buyer2weight = {}
            total_weight = 0
            for query_agent, counterparty_agent, edge_data in buyer_edges:
                # Read the note above - this is extra overhead to ensure that
                # we get the correct counterparty node
                if query_agent == seller_agent:
                    buyer_agent = counterparty_agent
                elif counterparty_agent == seller_agent:
                    buyer_agent = query_agent
                # get the transaction value from the edge data
                w = edge_data['weight']
                buyer2weight[buyer_agent] = w
                total_weight += w

            seller2weight[seller_agent] = buyer2weight
            seller2totalweight[seller_agent] = total_weight
            seller2value[seller_agent] = np.mean([total_weight, ec_full[seller_agent]])
            ec[seller_agent] = ec_full[seller_agent]
            # TODO: once we have a notion of total "value", we can compute it iteratively for each time-step
            # seller2value[seller_id] = ec[seller_agent]

        self.graph_evolution_metrics.append({
            'time_step': self.current_time_step,
            'seller2value': seller2value,
            'seller2eigenvectorcentrality': ec,  # only keep the seller agent's eigenvector centrality
            'seller2weight': seller2weight,
            'seller2totalweight': seller2totalweight,
            'total_supply': self.total_supply,
            'buyer_agent_id_counter': self.buyer_agent_id_counter,
            'seller_agent_id_counter': self.seller_agent_id_counter
        })
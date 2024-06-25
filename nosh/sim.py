from typing import Union, Callable
import networkx as nx

import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from tqdm.auto import tqdm
import scipy.stats as stats
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import powerlaw
import scipy.stats

from nosh.buyer_agent import BuyerAgent
from nosh.producer_agent import SellerAgent

def mt_linear(gt, t, m=1):
    return gt * (t*m)

def mt_exp(gt, t, alpha=0.05):
    return gt * np.exp(alpha*t)

def kld(p,q):
    assert len(p) == len(q)
    # assume they are defined on the same sample space
    kl = 0
    for ii in range(len(p)):
        if p[ii] > 0 and q[ii] > 0:
            kl += p[ii] * np.log(p[ii]/q[ii])
    return kl

# def network_state_fn(degrees):
#     try:
#         fit = powerlaw.Fit(degrees+1, discrete=True, xmin=1)
#         # get distance between fit and theoretical
#         dist = fit.power_law.KS()
#         return dist
#     except Exception as e:
#         print(e)
#         return 1
# def network_state_fn(G):
#     degree_distribution = nx.degree_histogram(G)
#     try:
#         # print(degree_distribution)
#         fit = powerlaw.Fit(degree_distribution, discrete=True, xmin=1)
#         # print(fit.power_law.alpha, fit.power_law.KS())
#         # get distance between fit and theoretical
#         dist = fit.power_law.KS()
#         if dist == np.nan:
#             return 1
#         return dist
#     except Exception as e:
#         # print(e)
#         # print(degree_distribution)
#         return 1
def network_state_fn(G):
    # compute kurtosis as a proxy for network state
    degrees = np.asarray(list(dict(G.degree()).values()))
    kurtosis_value = scipy.stats.kurtosis(degrees)
    return kurtosis_value


class NoshGraphSimulation:
    def __init__(
            self, 
            initial_buyers_kwargs, 
            initial_sellers_kwargs, 
            new_buyer_kwargs,
            new_seller_kwargs,
            mt_callback=mt_linear,
            min_buyers=2, 
            min_sellers=2, 
            seed=None, 
            weight_update_fn=None
    ):
        self.graph = nx.Graph()
        self.buyer_agents = {}
        self.seller_agents = {}
        self.mt_callback = mt_callback
        self.min_buyers = min_buyers
        self.min_sellers = min_sellers
        self.rng = np.random.default_rng(seed)
        
        self.graph_evolution_metrics = []
        self.new_buyer_kwargs = new_buyer_kwargs
        self.new_seller_kwargs = new_seller_kwargs

        self.current_time_step = 0
        self.total_minted_tokens = 0
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
    
    def step_all_agents(self):
        for buyer_agent in self.buyer_agents.values():
            buyer_agent.step()
        for seller_agent in self.seller_agents.values():
            seller_agent.step()
            
    def evolve_graph(self, add_delete_maintain_probs):
        add_prob, delete_prob, maintain_prob = add_delete_maintain_probs
        
        # Add or delete nodes
        prob = self.rng.random()
        if prob < add_prob:
            self.add_node()
        elif prob < add_prob + delete_prob:
            self.delete_node()

        # update internal state of all agents
        self.step_all_agents()

        # Simulate transactions and update weights
        # for each buyer and seller, determine if they will transact:
        # Note that the value of the transaction is symmetric in this current representation,
        #  but we can make it asymmetric later on.
        for buyer_agent in self.buyer_agents.values():
            for seller_agent in self.seller_agents.values():
                if buyer_agent.will_transact() and seller_agent.will_transact():
                    transaction_value = buyer_agent.make_transaction([seller_agent.agent_id])
                    if transaction_value:
                        edge = (buyer_agent, seller_agent)
                        if self.graph.has_edge(*edge):
                            weight_update_value = self.compute_updated_weight(self.graph.edges[edge]['seller_weight'], transaction_value)
                            self.graph.edges[edge]['seller_weight'] = weight_update_value
                            self.graph.edges[edge]['buyer_weight'] = weight_update_value
                        else:
                            weight_update_value = self.compute_updated_weight(0, transaction_value)
                            self.graph.add_edge(
                                *edge, 
                                seller_weight=weight_update_value,
                                buyer_weight=weight_update_value
                            )

        # simulate whether any sellers exchange some graph seller_weight for tokens
        for seller_agent in self.seller_agents.values():
            if seller_agent.will_unstake():
                # get all the buyers that the seller has an edge with and associated weights in a dictionary
                # NOTE: edges are added in the format (buyer, seller, seller_weight=...)
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
                    buyer2weight[buyer_agent] = edge_data['seller_weight']
                if len(buyer2weight) > 0:
                    buyer2burn = seller_agent.get_unstake_amount(buyer2weight)
                    # buyer2burn is a dictionary of how much to burn from each buyer that this seller is connected to
                    total_burned = 0
                    for buyer_to_burn, amount in buyer2burn.items():
                        burn_amount = amount
                        self.graph.edges[(buyer_to_burn, seller_agent)]['seller_weight'] -= burn_amount
                        
                        total_burned += burn_amount
                    
                    # update the token balance of the seller
                    tokens_added = self.mt_callback(total_burned, self.current_time_step)
                    # The seller exchanges some graph seller_weight for tokens
                    # TODO: need to have a callback to update the seller's token balance and compute it in the correct way
                    seller_agent.num_tokens += tokens_added
                    self.total_minted_tokens += tokens_added
            
            # TODO: can add unstaking with buyers also, now that we have a concept of buyers being able to mint tokens

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

        # convert the node's graph value to tokens, and add it to the total supply
        if delete_buyer:
            # TODO: include weight removal for buyer ... but currently sim is focused on seller
            pass
        else:
            total_seller_weight = 0
            buyer_edges = self.graph.edges(node_to_remove, data=True)
            for query_agent, counterparty_agent, edge_data in buyer_edges:
                total_seller_weight += edge_data['seller_weight']
            # update the token balance of the seller
            tokens_added = self.mt_callback(total_seller_weight, self.current_time_step)
            # no need to update the sellers token balance, since they will be removed!    
            self.total_minted_tokens += tokens_added

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
        # TODO: can add in buyer weight here for viz
        edge_labels = {(u, v): self.graph.edges[u, v]['seller_weight'] for u, v in self.graph.edges()}
        
        # Get the nodes involved in edges
        nodes_in_edges = set(sum(self.graph.edges(), ()))

        # Filter out nodes without edges
        nodes_to_draw = [node for node in self.graph.nodes() if node in nodes_in_edges]

        # Draw the graph with agent IDs as labels and edge weights as labels
        nx.draw(self.graph, pos, ax=ax, nodelist=nodes_to_draw, labels=agent_labels, node_color='skyblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=ax, font_color='red')
        ax.set_title("Bipartite Graph Visualization")

    def run(
            self, 
            num_time_steps: int,
            add_delete_maintain_probs: np.ndarray,
            weight_alpha_vec: np.ndarray,
            ec_alpha_vec_or_fn: Union[np.ndarray, Callable, str],
            reputation_alpha_vec: np.ndarray,
            create_video:bool=False,
            verbose:bool=False
        ):
        assert add_delete_maintain_probs.shape == (num_time_steps,3), "add_delete_maintain_probs must be a 2D array of length 3"
        assert weight_alpha_vec.shape == (num_time_steps,), "weight_alpha_vec must be a 1D array of length num_time_steps"
        if isinstance(ec_alpha_vec_or_fn, np.ndarray):
            assert ec_alpha_vec_or_fn.shape == (num_time_steps,), "ec_alpha_vec must be a 1D array of length num_time_steps"
        assert reputation_alpha_vec.shape == (num_time_steps,), "reputation_alpha_vec must be a 1D array of length num_time_steps"

        if create_video:
            frame_folder = 'frames'
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder, exist_ok=True)

        for ii in tqdm(range(num_time_steps), disable = not verbose):
            if callable(ec_alpha_vec_or_fn):
                if len(self.graph_evolution_metrics) > 0:
                    ec_alpha = ec_alpha_vec_or_fn(self.graph_evolution_metrics[-1])
                else:
                    ec_alpha = ec_alpha_vec_or_fn(None)
            elif isinstance(ec_alpha_vec_or_fn, np.ndarray):
                ec_alpha = ec_alpha_vec_or_fn[ii]
            elif isinstance(ec_alpha_vec_or_fn, str):
                if ec_alpha_vec_or_fn == 'dynamic':
                    ec_alpha = 'dynamic'
                else:
                    raise ValueError("ec_alpha_vec must be a 1D array or a callable function or string[dynamic]")
            else:
                raise ValueError("ec_alpha_vec must be a 1D array or a callable function or string")
            self.record_graph_metrics(weight_alpha_vec[ii], ec_alpha, reputation_alpha_vec[ii])
            self.evolve_graph(add_delete_maintain_probs[ii,:])
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
    
    def record_graph_metrics(self, weight_alpha, ec_alpha, reputation_alpha):

        def alpha(ec):
            return 1-ec

        # compute eigenvector centrality for all the nodes in the graph
        try:
            # NOTE: this computes the EC using an adjacency matrix with binary entries
            #       since the weight argument is not provided!
            # NOTE: the max_iter and tol arguments are set to avoid the PowerIterationFailedConvergence exception
            ec_full = nx.eigenvector_centrality(
                self.graph,
                max_iter=3000,
                tol=1e-1,
                weight='seller_weight'
            )
            
            # do min/max normalization to set EC values between 0 and 1
            if ec_alpha == 'dynamic':
                min_ec = min(ec_full.values())
                max_ec = max(ec_full.values())
                for node in ec_full:
                    if np.isclose(min_ec, max_ec):
                        ec_full[node] = 0
                    else:
                        ec_full[node] = (ec_full[node] - min_ec) / (max_ec - min_ec)
        except nx.PowerIterationFailedConvergence:
            print('Convergence Failed! Setting all EC values to 0')
            ec_full = {node: 0 for node in self.graph.nodes()}
        
        total_graph_value = 0

        seller_agents = list(self.seller_agents.values())
        seller2value = {}
        seller2weight = {}
        seller2totalweight = {}
        seller2ec = {}
        seller2degree = {}
        for seller_agent in seller_agents:
            buyer_edges = self.graph.edges(seller_agent, data=True)
            sellers_buyer2weight = {}
            total_weight = 0
            for query_agent, counterparty_agent, edge_data in buyer_edges:
                # Read the note above - this is extra overhead to ensure that
                # we get the correct counterparty node
                if query_agent == seller_agent:
                    buyer_agent = counterparty_agent
                elif counterparty_agent == seller_agent:
                    buyer_agent = query_agent
                # get the transaction value from the edge data
                w = edge_data['seller_weight']
                sellers_buyer2weight[buyer_agent] = w
                total_weight += w

            seller_ec = ec_full[seller_agent] + 1  # min-value=1
            seller2weight[seller_agent] = sellers_buyer2weight
            seller2totalweight[seller_agent] = total_weight
            
            # tv = (total_weight ** weight_alpha) * (seller_ec ** ec_alpha) * (seller_agent.get_current_reputation() ** reputation_alpha)
            aa = alpha(seller_ec)
            tv = (total_weight ** (1-aa)) * (seller_ec ** aa) * (seller_agent.get_current_reputation() ** reputation_alpha)
            
            seller2value[seller_agent] = tv
            seller2ec[seller_agent] = ec_full[seller_agent]
            seller2degree[seller_agent] = self.graph.degree(seller_agent)
            total_graph_value += tv

        buyer_agents = list(self.seller_agents.values())
        buyer2value = {}
        buyer2weight = {}
        buyer2totalweight = {}
        buyer2ec = {}
        buyer2degree = {}
        for buyer_agent in buyer_agents:
            seller_edges = self.graph.edges(buyer_agent, data=True)
            buyers_seller2weight = {}
            total_weight = 0
            for query_agent, counterparty_agent, edge_data in seller_edges:
                # Read the note above - this is extra overhead to ensure that
                # we get the correct counterparty node
                if query_agent == buyer_agent:
                    seller_agent = counterparty_agent
                elif counterparty_agent == buyer_agent:
                    seller_agent = query_agent
                # get the transaction value from the edge data
                w = edge_data['buyer_weight']
                buyers_seller2weight[buyer_agent] = w
                total_weight += w

            buyer_ec = ec_full[buyer_agent] + 1  # min-value=1
            buyer2weight[buyer_agent] = buyers_seller2weight
            buyer2totalweight[buyer_agent] = total_weight
            
            # tv = (total_weight ** weight_alpha) * (buyer_ec ** ec_alpha) * (buyer_agent.get_current_reputation() ** reputation_alpha)
            aa = alpha(seller_ec)
            tv = (total_weight ** (1-aa)) * (buyer_ec ** aa) * (buyer_agent.get_current_reputation() ** reputation_alpha)

            buyer2value[buyer_agent] = tv
            buyer2ec[buyer_agent] = ec_full[buyer_agent]
            buyer2degree[buyer_agent] = self.graph.degree(buyer_agent)
            total_graph_value += tv

        num_total_buyers = len(self.buyer_agents)
        num_total_sellers = len(self.seller_agents)

        # compute power-law fit and distance from fit to empirical distribution
        # for the seller degree distribution
        # seller_degrees = np.asarray(list(seller2degree.values()))
        # seller_powerlaw_dist = fit_powerlaw_compute_distance(seller_degrees)
        # seller_powerlaw_dist = fit_powerlaw_compute_distance(self.graph)
        network_state = network_state_fn(self.graph)

        self.graph_evolution_metrics.append({
            'time_step': self.current_time_step,
            'seller2value': seller2value,
            'seller2eigenvectorcentrality': seller2ec,
            'seller2weight': seller2weight,
            'seller2totalweight': seller2totalweight,
            'seller2degree': seller2degree,
            'network_state': network_state,
            'buyer2value': buyer2value,
            'buyer2eigenvectorcentrality': buyer2ec,
            'buyer2weight': buyer2weight,
            'buyer2totalweight': buyer2totalweight,
            'buyer2degree': buyer2degree,
            'total_supply': self.total_minted_tokens,
            'buyer_agent_id_counter': self.buyer_agent_id_counter,
            'seller_agent_id_counter': self.seller_agent_id_counter,
            'num_total_buyers': num_total_buyers,
            'num_total_sellers': num_total_sellers,
            'weight_alpha': weight_alpha,
            'ec_alpha': ec_alpha,
            'reputation_alpha': reputation_alpha,
            'alpha': aa,
            'total_graph_value': total_graph_value
        })

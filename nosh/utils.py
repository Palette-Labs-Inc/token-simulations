import numpy as np

def eigenvector_centrality_metric_fn(graph_evolution_metrics):
    # rank the sellers by eigenvector centrality
    max_num_sellers = graph_evolution_metrics[-1]['seller_agent_id_counter']
    max_t = len(graph_evolution_metrics)

    seller2ec_by_t = np.ones((max_num_sellers, max_t)) * -1  # a -1 means no seller at that index for a given time
    for t, metric in enumerate(graph_evolution_metrics):
        seller2ec = metric['seller2eigenvectorcentrality']
        for seller, ec in seller2ec.items():
            seller_id = seller.agent_id
            seller2ec_by_t[seller_id, t] = ec

    return seller2ec_by_t

def total_weight_metric_fn(graph_evolution_metrics):
    # rank the sellers by total weight
    max_num_sellers = max_num_sellers = graph_evolution_metrics[-1]['seller_agent_id_counter']
    max_t = len(graph_evolution_metrics)

    seller2weight_by_t = np.ones((max_num_sellers, max_t)) * -1  # a -1 means no seller at that index for a given time
    for t, metric in enumerate(graph_evolution_metrics):
        seller2weight = metric['seller2totalweight']
        for seller, weight in seller2weight.items():
            seller_id = seller.agent_id
            seller2weight_by_t[seller_id, t] = weight

    return seller2weight_by_t


def get_seller_by_quantile(graph_evolution_metrics, metric_fn=None, quantile=0.5):
    # TODO: get the matrix of seller2metric by time
    #       aggregate & then get the seller corresponding to the desired quantile
    pass
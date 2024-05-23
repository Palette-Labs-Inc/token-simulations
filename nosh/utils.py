import numpy as np

def _get_seller2metric_by_t(graph_evolution_metrics, metric_fn):
    max_num_sellers = graph_evolution_metrics[-1]['seller_agent_id_counter']
    max_t = len(graph_evolution_metrics)

    seller2metric_by_t = np.ones((max_num_sellers, max_t)) * np.nan
    for t, metric in enumerate(graph_evolution_metrics):
        seller2metric = metric_fn(metric)
        for seller, metric in seller2metric.items():
            seller_id = seller.agent_id
            seller2metric_by_t[seller_id, t] = metric

    return seller2metric_by_t

def eigenvector_centrality_metric_fn(graph_evolution_metrics):
    return _get_seller2metric_by_t(graph_evolution_metrics, lambda metric: metric['seller2eigenvectorcentrality'])

def total_weight_metric_fn(graph_evolution_metrics):
    return _get_seller2metric_by_t(graph_evolution_metrics, lambda metric: metric['seller2totalweight'])

def seller_totalvalue_metric_fn(graph_evolution_metrics):
    return _get_seller2metric_by_t(graph_evolution_metrics, lambda metric: metric['seller2value'])

def get_seller_by_quantile(graph_evolution_metrics, metric_fn=None, quantile=0.5):
    # TODO: get the matrix of seller2metric by time
    #       aggregate & then get the seller corresponding to the desired quantile
    pass
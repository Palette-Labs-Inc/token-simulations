import numpy as np

class Agent:
    def __init__(
            self, 
            agent_id, 
            rng, 
            num_tokens=0, 
            transact_probability=0.5,
            reputation_process_kwargs=None,
            dt=1
    ):
        self.agent_id = agent_id
        self.transact_probability = transact_probability
        self.num_tokens = num_tokens

        self.rng = rng

        if reputation_process_kwargs is None:
            reputation_process_kwargs = {
                'mu': 1,
                'sigma': 0.1,
                'start_value': 0.5,
            }
        else:
            assert 'mu' in reputation_process_kwargs
            assert 'sigma' in reputation_process_kwargs
            assert 'start_value' in reputation_process_kwargs
        # seed reputation process
        self.reputation_vector = [reputation_process_kwargs['start_value']]
        self.reputation_process_kwargs = reputation_process_kwargs
        self.dt = dt

    def will_transact(self):
        return self.rng.random() < self.transact_probability
    
    def make_transaction(self, counterparty_node):
        return self.compute_transaction_value(counterparty_node)
    
    def step(self):
        new_reputation = self.reputation_vector[-1] + \
            self.reputation_process_kwargs['mu'] * self.dt + \
            self.reputation_process_kwargs['sigma'] * np.sqrt(self.dt) * self.rng.normal()
        self.reputation_vector.append(new_reputation)
    
    def get_current_reputation(self):
        return self.reputation_vector[-1]
    
    def compute_transaction_value(self, counterparty_node):
        # Example: The transaction value is based on some logic
        return 10  # Placeholder value
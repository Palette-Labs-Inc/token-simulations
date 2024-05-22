import numpy as np

class Agent:
    def __init__(self, agent_id, rng, num_tokens=0, transact_probability=0.5):
        self.agent_id = agent_id
        self.transact_probability = transact_probability
        self.num_tokens = num_tokens

        self.rng = rng

    def will_transact(self):
        return self.rng.random() < self.transact_probability
    
    def make_transaction(self, counterparty_node):
        return self.compute_transaction_value(counterparty_node)
    
    def compute_transaction_value(self, counterparty_node):
        # Example: The transaction value is based on some logic
        return 10  # Placeholder value
from nosh.agent import Agent

class BuyerAgent(Agent):
    def __init__(self, agent_id, rng, starting_tokens=0, transact_probability=0.5):
        super().__init__(agent_id, rng, num_tokens=starting_tokens, transact_probability=transact_probability)
        # print('BuyerAgent initialized with agent_id:', self.agent_id, 'num_tokens:', self.num_tokens, 'transact_probability:', self.transact_probability)
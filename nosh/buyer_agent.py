from nosh.agent import Agent

class BuyerAgent(Agent):
    def __init__(self, agent_id, rng, transact_probability=0.5):
        super().__init__(agent_id, rng, transact_probability)
from nosh.agent import Agent

class BuyerAgent(Agent):
    def __init__(
            self, 
            agent_id, 
            rng, 
            starting_tokens=0, 
            transact_probability=0.5,
            reputation_process_kwargs=None,
            dt=1,
        ):
        super().__init__(
            agent_id, 
            rng, 
            num_tokens=starting_tokens, 
            transact_probability=transact_probability,
            reputation_process_kwargs=reputation_process_kwargs,
            dt=dt
        )

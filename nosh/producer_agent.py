from nosh.agent import Agent

class SellerAgent(Agent):
    def __init__(
            self, 
            agent_id, 
            rng, 
            starting_tokens=0, 
            transact_probability=0.5, 
            unstake_probability=0.2,
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
        self.unstake_probability = unstake_probability

    def will_unstake(self):
        return self.rng.random() < self.unstake_probability
    
    def get_unstake_amount(self, buyer2weight):
        # burn a random amount from each edge
        buyer2burn = {}
        for b, w in buyer2weight.items():
            burn_fraction = self.rng.random()
            buyer2burn[b] = w * burn_fraction
        return buyer2burn

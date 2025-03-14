"""
SocialMediaAgent.py - Base class for agents in social media simulation
using Mesa 3.1.4
"""

import mesa
import random
from datetime import date, timedelta

class SocialMediaAgent(mesa.Agent):
    """Base class for all agents in the social media simulation."""

    def __init__(self, unique_id, model, post_type=None, post_frequency=None):
        # In Mesa 3.1.4, Agent.__init__ no longer accepts positional arguments
        # We need to explicitly pass unique_id and model as kwargs
        super().__init__(unique_id=unique_id, model=model)

        self.creation_date = date(2022, 1, 1) + timedelta(model.steps)
        self.deactivation_date = None
        self.active = True
        self.connections = set()  # Set of connected agents
        self.post_frequency = post_frequency  # Posts per day
        self.last_post_date = self.creation_date
        self.popularity = model.random.uniform(0, 1)  # Use model's RNG for reproducibility
        self.posted_today = False
        self.post_type = post_type

    @staticmethod
    def get_current_date(model):
        return date(2022, 1, 1) + timedelta(model.steps)

    def step(self):
        """Base step function to be overridden by child classes."""
        if not self.active:
            return

    def get_agent_by_id(self, id):
        """Retrieve an agent by their unique ID from the model's agents collection."""
        return self.model.agents.get_agent_by_id(id)

    def should_post(self):
        """Determine if the agent should post based on their post frequency."""
        return self.model.random.random() < self.post_frequency  # Use model's RNG for reproducibility

    def deactivate(self):
        """Deactivate the agent."""
        self.active = False
        self.deactivation_date = self.get_current_date(self.model)

    def add_connection(self, other_agent):
        """Add a connection to another agent."""
        self.connections.add(other_agent.unique_id)
        other_agent.connections.add(self.unique_id)

    def remove_connection(self, other_agent):
        """Remove a connection to another agent."""
        if other_agent.unique_id in self.connections:
            self.connections.remove(other_agent.unique_id)
        if self.unique_id in other_agent.connections:
            other_agent.connections.remove(self.unique_id)

    def get_connections(self):
        """Return a list of connected agent instances."""
        return [self.get_agent_by_id(agent_id) for agent_id in self.connections
                if self.get_agent_by_id(agent_id) is not None]
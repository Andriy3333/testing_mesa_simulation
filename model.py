"""
model.py - Implementation of a small world network model for social media simulation
using Mesa 3.1.4
"""

import mesa
import numpy as np
import networkx as nx
from datetime import date, timedelta
from mesa.datacollection import DataCollector

from HumanAgent import HumanAgent
from BotAgent import BotAgent
import constants  # Assuming constants file exists

class SmallWorldNetworkModel(mesa.Model):
    """Small world network model for social media simulation."""

    def __init__(
        self,
        num_initial_humans=100,
        num_initial_bots=20,
        human_creation_rate=0.1,  # New humans per step
        bot_creation_rate=0.05,   # New bots per step
        connection_rewiring_prob=0.1,  # For small world network
        topic_shift_frequency=30,  # Steps between major topic shifts
        dimensions=5,  # Dimensions in topic space
        seed=None,  # Added seed parameter
    ):
        # Pass the seed to super().__init__() as required in Mesa 3.0+
        super().__init__(seed=seed)

        # Create a numpy random generator with the same seed
        self.np_random = np.random.RandomState(seed)

        self.num_initial_humans = num_initial_humans
        self.num_initial_bots = num_initial_bots
        self.human_creation_rate = human_creation_rate
        self.bot_creation_rate = bot_creation_rate
        self.connection_rewiring_prob = connection_rewiring_prob
        self.topic_shift_frequency = topic_shift_frequency
        self.dimensions = dimensions

        # Initialize counters and trackers
        # Note: self.steps is now automatically managed by Mesa 3.0+
        self.next_id = 0
        self.active_humans = 0
        self.active_bots = 0
        self.deactivated_humans = 0
        self.deactivated_bots = 0
        self.avg_human_satisfaction = 0

        # Initialize data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Active Humans": lambda m: m.active_humans,
                "Active Bots": lambda m: m.active_bots,
                "Deactivated Humans": lambda m: m.deactivated_humans,
                "Deactivated Bots": lambda m: m.deactivated_bots,
                "Average Human Satisfaction": self.get_avg_human_satisfaction,
            },
            agent_reporters={
                "Satisfaction": lambda a: getattr(a, "satisfaction", 0),
                "Agent Type": lambda a: getattr(a, "agent_type", ""),
                "Active": lambda a: getattr(a, "active", False),
                "Connections": lambda a: len(getattr(a, "connections", [])),
            }
        )

        # Create initial network topology
        self.create_network()

        # Create initial agents
        self.create_initial_agents()

    def get_next_id(self):
        """Get next unique ID and increment counter."""
        # Note: In Mesa 3.0+, unique_id is automatically assigned
        # This method is kept for backward compatibility
        next_id = self.next_id
        self.next_id += 1
        return next_id

    def create_network(self):
        """Create a small world network for agent connections."""
        # We'll use a NetworkX small world graph as the basis
        # This will just store the structure, agents are stored in the schedule

        # Start with a ring lattice
        n = self.num_initial_humans + self.num_initial_bots
        k = 4  # Each node connected to k nearest neighbors

        # Use the model's random number generator for reproducibility
        self.network = nx.watts_strogatz_graph(
            n,
            k,
            self.connection_rewiring_prob,
            seed=self.random.randint(0, 2 ** 32 - 1)  # Use the model's RNG
        )

    def create_initial_agents(self):
        """Create initial human and bot agents."""
        # Create humans
        for i in range(self.num_initial_humans):
            agent = HumanAgent(model=self)  # Updated for Mesa 3.1.4
            self.active_humans += 1

        # Create bots
        for i in range(self.num_initial_bots):
            agent = BotAgent(model=self)  # Updated for Mesa 3.1.4
            self.active_bots += 1

        # Create initial connections based on network topology
        self.update_agent_connections()

    def update_agent_connections(self):
        """Update agent connections based on current network topology."""
        # Reset all connections
        for agent in self.agents:
            agent.connections = set()

        # Get all active agents
        active_agents = [agent for agent in self.agents if agent.active]

        # Create connections based on network edges
        for edge in self.network.edges():
            source_idx, target_idx = edge

            # Skip if indices are out of range
            if source_idx >= len(active_agents) or target_idx >= len(active_agents):
                continue

            # Get the agents using their indices in the active_agents list
            source_agent = active_agents[source_idx]
            target_agent = active_agents[target_idx]

            # Add connection between the agents
            if source_agent and target_agent:
                source_agent.add_connection(target_agent)

    def rewire_network(self):
        """Rewire the network connections to simulate changing interests."""
        # Create a new small world network
        n = self.active_humans + self.active_bots
        k = 4  # Each node connected to k nearest neighbors

        if n > k:  # Make sure we have enough nodes
            self.network = nx.watts_strogatz_graph(
                n,
                k,
                self.connection_rewiring_prob,
                seed=self.random.randint(0, 2 ** 32 - 1)  # Use the model's RNG for seed
            )

            # Update agent connections
            self.update_agent_connections()

    def get_nearby_agents(self, agent, threshold=0.5):
        """Get agents that are nearby in topic space."""
        nearby_agents = []

        for other in self.agents:
            if other.unique_id != agent.unique_id and other.active:
                # Calculate topic similarity
                similarity = self.calculate_topic_similarity(agent, other)
                if similarity > threshold:
                    nearby_agents.append(other)

        return nearby_agents

    def calculate_topic_similarity(self, agent1, agent2):
        """Calculate similarity in topic space between two agents."""
        # Cosine similarity between topic vectors
        dot_product = np.dot(agent1.topic_position, agent2.topic_position)
        norm1 = np.linalg.norm(agent1.topic_position)
        norm2 = np.linalg.norm(agent2.topic_position)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0

        similarity = dot_product / (norm1 * norm2)

        # Convert from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2

    def create_new_agents(self):
        """Create new agents based on creation rates."""
        # Create new humans - Use np_random for Poisson distribution
        num_new_humans = self.np_random.poisson(self.human_creation_rate)
        for _ in range(num_new_humans):
            agent = HumanAgent(model=self)  # Updated for Mesa 3.1.4
            self.active_humans += 1

        # Create new bots - Use np_random for Poisson distribution
        num_new_bots = self.np_random.poisson(self.bot_creation_rate)
        for _ in range(num_new_bots):
            agent = BotAgent(model=self)  # Updated for Mesa 3.1.4
            self.active_bots += 1

    def update_agent_counts(self):
        """Update counters for active and deactivated agents."""
        active_humans = 0
        active_bots = 0
        deactivated_humans = 0
        deactivated_bots = 0

        for agent in self.agents:
            if getattr(agent, "agent_type", "") == "human":
                if agent.active:
                    active_humans += 1
                else:
                    deactivated_humans += 1
            elif getattr(agent, "agent_type", "") == "bot":
                if agent.active:
                    active_bots += 1
                else:
                    deactivated_bots += 1

        self.active_humans = active_humans
        self.active_bots = active_bots
        self.deactivated_humans = deactivated_humans
        self.deactivated_bots = deactivated_bots

    def get_avg_human_satisfaction(self):
        """Calculate average satisfaction of active human agents."""
        satisfactions = [
            agent.satisfaction for agent in self.agents
            if getattr(agent, "agent_type", "") == "human" and agent.active
        ]

        if satisfactions:
            return sum(satisfactions) / len(satisfactions)
        return 0

    def step(self):
        """Advance the model by one step."""
        # Execute agent steps (using the Mesa 3.1.4 syntax)
        self.agents.shuffle_do("step")

        # Create new agents
        self.create_new_agents()

        # Periodically rewire the network to simulate changing trends
        if self.steps % self.topic_shift_frequency == 0:
            self.rewire_network()

        # Update agent counters
        self.update_agent_counts()

        # Update data collector
        self.datacollector.collect(self)

        # Note: In Mesa 3.0+, steps counter is automatically incremented
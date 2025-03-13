from SocialMediaAgent import SocialMediaAgent
import mesa
import random
import numpy as np
import constants
from datetime import date, timedelta

from datetime import date, timedelta


class HumanAgent(SocialMediaAgent):
    """Human agent in the social media simulation."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, post_type = "normal")
        self.agent_type = "human"

        # Satisfaction ranges from 0 (negative) to 100 (positive)
        self.satisfaction = 100

        self.move_in_topic_space()

        # Personality parameters that affect interactions
        self.irritability = random.uniform(0.5, 2)  # More irritable users will lose satisfaction quicker
        self.authenticity = random.uniform(0.5, 2)  # Higher number indicates distaste for misinformation
        self.activeness = random.random()  # Determines exposure to posts

        # Position in the topic space (for network formation)
        self.topic_position = np.random.normal(0, 0.5, size=5)  # 5-dimensional topic space

        self.post_frequency = random.uniform(0.1, 0.6)

        self.popularity = random.uniform(0.3, 0.95)


    def step(self):
        """Human agent behavior during each step."""
        super().step()

        self.human_post() # Posts if necessary

        # Check if satisfaction is too low to continue
        if self.satisfaction <= 0:
            self.deactivate()
            return

        # Interact with nearby agents
        self.react_to_posts()

    def human_post(self):
        if self.should_post():
            self.last_post_date = self.get_current_date(self.model)
            self.posted_today = True
        else:
            self.posted_today = False

    def react_to_all_posts(self):
        """Interact with connected or nearby agents."""

        # First, try interacting with connected agents
        connected_agents = self.get_connections()

        # Also consider nearby agents in topic space that aren't connected yet
        nearby_agents = self.model.get_nearby_agents(self)

        # Combine the lists, prioritizing connections
        potential_interactions = connected_agents + [a for a in nearby_agents if a not in connected_agents]

        # Filter out thos who didnt post
        potential_interactions = [a for a in potential_interactions if a.posted_today]

        interaction_targets = potential_interactions.copy() if potential_interactions else []


        for target in interaction_targets:
            self.react_to_post(target)

    def react_to(self, other_agent):
        """Interact with another agent and update satisfaction accordingly."""
        # Different interaction types based on the other agent
        if other_agent.post_type() == "normal":
            self.satisfaction += 1
        elif other_agent.post_type() == "misinformation":
            self.satisfcation -= 1 * self.authenticity * self.irritability
        elif other_agent.post_type() == "astroturfing":
            self.satisfcation -= 1 * self.authenticity * self.irritability
        elif other_agent.post_type() == "spam":
            self.satisfcation -= 1 * self.irritability
        else:
            return

    def update_connection_probability(self, other_agent, satisfaction_change):
        """Update probability of maintaining connection based on interaction."""
        # If interaction was positive, increase chance to interact again
        # If negative, decrease chance

        # For human-to-human, might form echo chambers
        if other_agent.agent_type == "human":
            if satisfaction_change > 0:
                # Positive interaction strengthens connection
                if other_agent.unique_id not in self.connections:
                    # 20% chance to form new connection after positive interaction
                    if random.random() < 0.2:
                        self.add_connection(other_agent)
            else:
                # Negative interaction might break connection
                if other_agent.unique_id in self.connections:
                    # 10% chance to break connection after negative interaction
                    if random.random() < 0.1:
                        self.remove_connection(other_agent)

        # For human-to-bot, negative interactions might lead to blocking
        elif other_agent.agent_type == "bot":
            if satisfaction_change < -1.5:
                # Very negative interactions with bots might lead to blocking
                if other_agent.unique_id in self.connections:
                    # 30% chance to block bot after very negative interaction
                    if random.random() < 0.3:
                        self.remove_connection(other_agent)

    # def calculate_topic_similarity(self, other_agent):
    #     """Calculate similarity in topic space between this agent and another."""
    #     # Cosine similarity between topic vectors
    #     dot_product = np.dot(self.topic_position, other_agent.topic_position)
    #     norm_product = np.linalg.norm(self.topic_position) * np.linalg.norm(other_agent.topic_position)
    #     similarity = dot_product / (norm_product + 1e-8)  # Add small epsilon to avoid division by zero
    #
    #     # Convert from [-1, 1] to [0, 1] range
    #     return (similarity + 1) / 2

    # def shift_interests(self):
    #     """Shift interests in the topic space to simulate changing user preferences."""
    #     # Small random shift in topic position
    #     shift = np.random.normal(0, 0.1, size=5)
    #     self.topic_position = self.topic_position + shift
    #
    #     # Normalize to stay in similar bounds
    #     norm = np.linalg.norm(self.topic_position)
    #     if norm > 2.0:  # Prevent too extreme positions
    #         self.topic_position = self.topic_position * (2.0 / norm)

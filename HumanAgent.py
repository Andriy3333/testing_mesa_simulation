"""
HumanAgent.py - Implementation of human agents for social media simulation
using Mesa 3.1.4
"""

from SocialMediaAgent import SocialMediaAgent
import numpy as np
from datetime import date, timedelta


class HumanAgent(SocialMediaAgent):
    """Human agent in the social media simulation."""

    def __init__(self, unique_id, model):
        # Pass keyword arguments to super().__init__
        super().__init__(unique_id=unique_id, model=model, post_type="normal")
        self.agent_type = "human"

        # Satisfaction ranges from 0 (negative) to 100 (positive)
        self.satisfaction = 100

        # Personality parameters that affect interactions - use model's RNG
        self.irritability = model.random.uniform(0.5, 2)  # More irritable users will lose satisfaction quicker
        self.authenticity = model.random.uniform(0.5, 2)  # Higher number indicates distaste for misinformation
        self.activeness = model.random.random()  # Determines exposure to posts

        # Position in the topic space (for network formation)
        # Use model's numpy RNG wrapper
        self.topic_position = model.random.normal(0, 0.5, size=5)  # 5-dimensional topic space

        self.post_frequency = model.random.uniform(0.1, 0.6)
        self.popularity = model.random.uniform(0.3, 0.95)

        # Initialize position in topic space
        self.move_in_topic_space()

    def step(self):
        """Human agent behavior during each step."""
        super().step()

        if not self.active:
            return

        # Move in topic space
        self.move_in_topic_space()

        # Post with some probability
        self.human_post()

        # React to posts from connected agents
        self.react_to_posts()

        # Check if satisfaction is too low to continue
        if self.satisfaction <= 0:
            self.deactivate()

    def human_post(self):
        """Create a post with some probability."""
        if self.should_post():
            self.last_post_date = self.get_current_date(self.model)
            self.posted_today = True
        else:
            self.posted_today = False

    def move_in_topic_space(self):
        """Move in topic space to simulate changing interests."""
        # Small random shift in topic position - use model's numpy RNG wrapper
        shift = self.model.random.normal(0, 0.1, size=5)  # 5-dimensional topic space
        self.topic_position = self.topic_position + shift

        # Normalize to stay in similar bounds
        norm = np.linalg.norm(self.topic_position)
        if norm > 2.0:  # Prevent too extreme positions
            self.topic_position = self.topic_position * (2.0 / norm)

    def react_to_posts(self):
        """React to posts from connected or nearby agents."""
        # Get all active connected agents
        connected_agents = [
            self.get_agent_by_id(agent_id) for agent_id in self.connections
            if self.get_agent_by_id(agent_id) is not None and self.get_agent_by_id(agent_id).active
        ]

        # Also consider nearby agents in topic space that aren't connected yet
        nearby_agents = self.model.get_nearby_agents(self)

        # Combine the lists, prioritizing connections
        potential_interactions = connected_agents + [a for a in nearby_agents if a.unique_id not in self.connections]

        # Filter out those who didn't post
        potential_interactions = [a for a in potential_interactions if getattr(a, "posted_today", False)]

        # Randomly select a subset to interact with based on activeness
        num_interactions = max(1, int(len(potential_interactions) * self.activeness))
        if potential_interactions and num_interactions > 0:
            # Use model's random for reproducibility
            interaction_targets = self.model.random.sample(
                potential_interactions,
                min(num_interactions, len(potential_interactions))
            )

            # React to each target's post
            for target in interaction_targets:
                self.react_to_post(target)

    def react_to_post(self, other_agent):
        """React to a post from another agent."""
        # Base satisfaction change
        satisfaction_change = 0

        # Human-to-human interaction
        if other_agent.agent_type == "human":
            # Humans generally have positive interactions with each other
            # with some randomness - use model's RNG
            base_change = self.model.random.uniform(-0.5, 2.0)  # Slightly favored positive

            # Topic similarity increases positive satisfaction
            similarity = self.model.calculate_topic_similarity(self, other_agent)
            topic_effect = (similarity * 2) - 0.5  # Range from -0.5 to 1.5

            satisfaction_change = base_change + topic_effect

        # Human-to-bot interaction
        elif other_agent.agent_type == "bot":
            # Mostly negative interactions with bots
            if other_agent.post_type == "normal":
                # Occasionally bots post normal content - use model's RNG
                satisfaction_change = self.model.random.uniform(-0.2, 0.5)
            elif other_agent.post_type == "misinformation":
                # Misinformation is more negative for users who value authenticity
                satisfaction_change = -1.0 * self.authenticity * self.irritability
            elif other_agent.post_type == "astroturfing":
                # Astroturfing is similarly negative
                satisfaction_change = -0.8 * self.authenticity * self.irritability
            elif other_agent.post_type == "spam":
                # Spam is universally annoying
                satisfaction_change = -1.0 * self.irritability

        # Apply satisfaction change
        self.satisfaction += satisfaction_change

        # Cap satisfaction between 0 and 100
        self.satisfaction = max(0, min(100, self.satisfaction))

        # Update connection probability based on interaction
        self.update_connection_probability(other_agent, satisfaction_change)

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
                    # Use model's RNG for reproducibility
                    if self.model.random.random() < 0.2:
                        self.add_connection(other_agent)
            else:
                # Negative interaction might break connection
                if other_agent.unique_id in self.connections:
                    # 10% chance to break connection after negative interaction
                    # Use model's RNG for reproducibility
                    if self.model.random.random() < 0.1:
                        self.remove_connection(other_agent)

        # For human-to-bot, negative interactions might lead to blocking
        elif other_agent.agent_type == "bot":
            if satisfaction_change < -1.5:
                # Very negative interactions with bots might lead to blocking
                if other_agent.unique_id in self.connections:
                    # 30% chance to block bot after very negative interaction
                    # Use model's RNG for reproducibility
                    if self.model.random.random() < 0.3:
                        self.remove_connection(other_agent)
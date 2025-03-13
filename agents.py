"""
agents.py - Implementation of human and bot agents for social media simulation
using Mesa 3.1.4
"""

import mesa
import random
import numpy as np
import constants
from datetime import date, timedelta

class SocialMediaAgent(mesa.Agent):
    """Base class for all agents in the social media simulation."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.creation_date = date(2022, 1, 1) + timedelta(model.steps)
        self.deactivation_date = None
        self.active = True
        self.connections = set()  # Set of connected agents
        self.post_frequency = random.uniform(0.05, 0.8)  # Posts per day
        self.last_post_date = self.creation_date

    @staticmethod
    def get_current_date(model):
        return date(2022, 1, 1) + timedelta(model.steps)

    def step(self):
        """Base step function to be overridden by child classes."""
        if not self.active:
            return

        # Determine if agent posts today
        if self.should_post():
            self.post()
            self.last_post_date = self.get_current_date(self.model)


    def get_agent_by_id(self, id):
        """Retrieve an agent by their unique ID from the model's schedule."""
        return self.model._agents.get(id, None)

    def should_post(self):
        """Determine if the agent should post based on their post frequency."""
        days_since_last_post = (self.get_current_date(self.model) - self.last_post_date).days
        post_probability = min(1.0, self.post_frequency * days_since_last_post)
        return random.random() < post_probability

    def post(self):
        """Create a post that might be interacted with by other agents."""
        # Posts will be handled by the model's interaction system
        self.model.register_post(self)

    def deactivate(self):
        """Deactivate the agent."""
        self.active = False
        self.deactivation_date = self.get_current_date(self.model)
        self.remove()

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


class HumanAgent(SocialMediaAgent):
    """Human agent in the social media simulation."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agent_type = "human"
        # Satisfaction ranges from 0 (negative) to 100 (positive)
        self.satisfaction = 100
        self.irritability = random.random()

        # Personality parameters that affect interactions
        self.openness = random.uniform(0.1, 1.0)  # Openness to new connections
        self.echo_chamber_tendency = random.uniform(0.3, 0.8)  # Tendency to seek similar views
        self.bot_skepticism = random.uniform(0.2, 0.9)  # Ability to recognize bots, maybe change this to misinfo skepticism?
        # Position in the topic space (for network formation)
        self.topic_position = np.random.normal(0, 0.5, size=5)  # 5-dimensional topic space
        # Interaction history
        self.interaction_history = {}  # {agent_id: [interaction_score_1, xinteraction_score_2, ...]}

    def step(self):
        """Human agent behavior during each step."""
        if not self.active:
            return

        super().step()

        # Check if satisfaction is too low to continue
        if self.satisfaction <= 0:
            self.deactivate()
            return

        # Interact with nearby agents
        self.interact_with_agents()

        # Potentially move in the topic space
        if random.random() < 0.1:  # 10% chance to shift interests each step
            self.shift_interests()

    def interact_with_agents(self):
        """Interact with connected or nearby agents."""
        # First, try interacting with connected agents
        connected_agents = self.get_connections()

        # Also consider nearby agents in topic space that aren't connected yet
        nearby_agents = self.model.get_nearby_agents(self, limit=5)

        # Combine the lists, prioritizing connections
        potential_interactions = connected_agents + [a for a in nearby_agents if a not in connected_agents]

        # Filter out inactive agents
        potential_interactions = [a for a in potential_interactions if a.active]

        # Limit interactions per step
        max_interactions = min(len(potential_interactions), 3)
        interaction_targets = random.sample(potential_interactions, max_interactions) if potential_interactions else []

        for target in interaction_targets:
            self.interact_with(target)

    def interact_with(self, other_agent):
        """Interact with another agent and update satisfaction accordingly."""
        # Different interaction types based on the other agent
        if isinstance(other_agent, HumanAgent):
            interaction_score = self.human_to_human_interaction(other_agent)
        elif isinstance(other_agent, BotAgent):
            interaction_score = self.human_to_bot_interaction(other_agent)
        else:
            return  # Unknown agent type

        # Update satisfaction based on the interaction
        self.update_satisfaction(interaction_score)

        # Update interaction history
        if other_agent.unique_id not in self.interaction_history:
            self.interaction_history[other_agent.unique_id] = []
        self.interaction_history[other_agent.unique_id].append(interaction_score)

        # Update connections based on interaction
        # Positive interactions may create new connections
        if interaction_score > 0 and other_agent.unique_id not in self.connections:
            if random.random() < 0.3 + (interaction_score / 100) * 0.4:
                self.add_connection(other_agent)

        # Negative interactions may break connections
        elif interaction_score < 0 and other_agent.unique_id in self.connections:
            if random.random() < 0.1 + (abs(interaction_score) / 100) * 0.3:
                self.remove_connection(other_agent)

    def human_to_human_interaction(self, other_human):
        """Model human-to-human interaction.

        Returns:
            float: Score between -50 (very negative) and 50 (very positive)
        """
        # Base likelihood of positive interaction
        positive_bias = 0.65  # Slightly favored positive interactions

        # Adjust based on topic similarity (echo chamber effect)
        topic_similarity = self.calculate_topic_similarity(other_human)

        # Echo chamber effect - similar topics lead to more positive interactions
        echo_effect = topic_similarity * self.echo_chamber_tendency * 20

        # Calculate the final score with some randomness
        base_score = random.uniform(-40, 40)  # Base randomness
        adjusted_score = base_score + (positive_bias * 20) + echo_effect

        # Ensure the score is within bounds
        interaction_score = max(-50, min(50, adjusted_score))

        return interaction_score

    def human_to_bot_interaction(self, bot):
        """Model human-to-bot interaction.

        Returns:
            float: Score between -50 (very negative) and 50 (very positive)
        """
        # Base likelihood - mostly negative
        negative_bias = 0.7

        # Calculate bot recognition probability based on skepticism
        recognized_as_bot = random.random() < self.bot_skepticism * bot.detection_difficulty

        # If recognized as bot, interaction is generally negative
        if recognized_as_bot:
            base_score = random.uniform(-50, -10)
        else:
            # If not recognized, the bot might successfully manipulate
            topic_similarity = self.calculate_topic_similarity(bot)
            echo_effect = topic_similarity * (1 - self.echo_chamber_tendency) * 20
            base_score = random.uniform(-30, 30) - (negative_bias * 20) + echo_effect

        # Ensure the score is within bounds
        interaction_score = max(-50, min(50, base_score))

        return interaction_score

    def calculate_topic_similarity(self, other_agent):
        """Calculate similarity in topic space between this agent and another."""
        # Cosine similarity between topic vectors
        dot_product = np.dot(self.topic_position, other_agent.topic_position)
        norm_product = np.linalg.norm(self.topic_position) * np.linalg.norm(other_agent.topic_position)
        similarity = dot_product / (norm_product + 1e-8)  # Add small epsilon to avoid division by zero

        # Convert from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2

    def update_satisfaction(self, interaction_score):
        """Update satisfaction based on an interaction."""
        # Scale the impact based on current satisfaction to create a non-linear effect
        # Users with high satisfaction are more resilient to negative interactions
        impact_factor = 0.5 * (1 - (self.satisfaction / 100)) + 0.5

        # Update satisfaction
        self.satisfaction += interaction_score * impact_factor * 0.2

        # Ensure satisfaction stays within bounds [0, 100]
        self.satisfaction = max(0, min(100, self.satisfaction))

    def shift_interests(self):
        """Shift interests in the topic space to simulate changing user preferences."""
        # Small random shift in topic position
        shift = np.random.normal(0, 0.1, size=5)
        self.topic_position = self.topic_position + shift

        # Normalize to stay in similar bounds
        norm = np.linalg.norm(self.topic_position)
        if norm > 2.0:  # Prevent too extreme positions
            self.topic_position = self.topic_position * (2.0 / norm)


class BotAgent(SocialMediaAgent):
    """Bot agent in the social media simulation."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agent_type = "bot"
        self.bot_type = random.choice(["spam", "misinformation", "astroturfing"])

        # Bot characteristics
        self.detection_difficulty = random.uniform(0.3, 0.9)  # Higher means harder to detect
        self.topic_position = np.random.normal(0, 1, size=5)  # Fixed topic position
        self.topic_mobility = random.uniform(0.0, 0.3)  # Lower mobility than humans

        # Bot network parameters
        self.connection_strategy = random.choice(["random", "targeted", "broadcast", "echo_chamber"])

        # Bot activity parameters
        self.post_frequency = random.uniform(0.5, 2.0)  # Higher posting rate than humans
        self.ban_probability = 0.01  # Daily probability of being banned

        # Set specific parameters based on bot type
        self.configure_bot_type()

    def get_connection_strategy(self):
        return self.connection_strategy

    def configure_bot_type(self):
        """Configure bot parameters based on type."""
        if self.bot_type == "misinformation":
            self.post_frequency = random.uniform(1.0, 3.0)
            self.detection_difficulty = random.uniform(0.5, 0.8)
            self.connection_strategy = random.choice(["targeted", "echo_chamber", "random", "broadcast"])
        elif self.bot_type == "spam":
            self.post_frequency = random.uniform(2.0, 5.0)
            self.detection_difficulty = random.uniform(0.1, 0.4)
            self.connection_strategy = random.choice(["broadcast", "random"])
            self.ban_probability = 0.03
        elif self.bot_type == "astroturfing":
            self.post_frequency = random.uniform(0.8, 2.0)
            self.detection_difficulty = random.uniform(0.7, 0.9)
            self.connection_strategy = random.choice(["targeted", "echo_chamber", "random", "broadcast"])

    def step(self):
        """Bot agent behavior during each step."""
        if not self.active:
            return

        super().step()

        # Check if the bot gets banned
        if random.random() < self.ban_probability:
            self.deactivate()
            return

        # Target humans for interaction
        self.target_humans()

        # Possibly shift topic position slightly (limited mobility)
        if random.random() < 0.05:  # 5% chance to shift
            self.shift_topic()

    def target_humans(self):
        """Target human agents for interaction based on bot strategy."""
        # Different targeting strategies
        if self.connection_strategy == "random":
            # Random targeting
            humans = [agent for agent in self.model.schedule.agents
                     if isinstance(agent, HumanAgent) and agent.active]
            if humans:
                target_count = min(len(humans), random.randint(1, 5))
                targets = random.sample(humans, target_count)
                for target in targets:
                    self.attempt_interaction(target)

        elif self.connection_strategy == "targeted":
            # Target humans based on topic similarity
            humans = self.model.get_nearby_agents(self, agent_type="human", limit=10)
            for human in humans:
                similarity = self.calculate_topic_similarity(human)
                # Higher probability of interaction with similar topic positions
                if random.random() < similarity * 0.8:
                    self.attempt_interaction(human)

        elif self.connection_strategy == "broadcast":
            # Broadcast to many humans with low success rate
            humans = [agent for agent in self.model.schedule.agents
                     if isinstance(agent, HumanAgent) and agent.active]
            if humans:
                target_count = min(len(humans), random.randint(5, 15))
                targets = random.sample(humans, target_count)
                for target in targets:
                    # Low success rate for spam bots
                    if random.random() < 0.2:
                        self.attempt_interaction(target)

        elif self.connection_strategy == "echo_chamber":
            # Target humans that already have connections to other bots
            bot_connected_humans = []
            for agent in self.model.schedule.agents:
                if isinstance(agent, HumanAgent) and agent.active:
                    bot_connections = [a for a in agent.get_connections()
                                     if isinstance(a, BotAgent)]
                    if bot_connections:
                        bot_connected_humans.append((agent, len(bot_connections)))

            # Sort by number of bot connections (descending)
            bot_connected_humans.sort(key=lambda x: x[1], reverse=True)

            # Target humans with most bot connections first
            for human, _ in bot_connected_humans[:5]:  # Top 5
                self.attempt_interaction(human)

    def attempt_interaction(self, human):
        """Attempt to interact with a human agent."""
        # Only interact if active
        if not human.active or not self.active:
            return

        # Check if already connected
        if human.unique_id in self.connections:
            # Higher chance to interact with connected humans
            interact_probability = 0.6
        else:
            # Lower chance to interact with new humans
            interact_probability = 0.3

        # Determine if interaction happens
        if random.random() < interact_probability:
            # This will trigger the human's human_to_bot_interaction method
            human.interact_with(self)

            # Possibly form a connection if not already connected
            if human.unique_id not in self.connections:
                connection_probability = 0.2
                if random.random() < connection_probability:
                    self.add_connection(human)

    def calculate_topic_similarity(self, other_agent):
        """Calculate similarity in topic space between this bot and another agent."""
        # Cosine similarity between topic vectors
        dot_product = np.dot(self.topic_position, other_agent.topic_position)
        norm_product = np.linalg.norm(self.topic_position) * np.linalg.norm(other_agent.topic_position)
        similarity = dot_product / (norm_product + 1e-8)

        # Convert from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2

    def shift_topic(self):
        """Slightly shift the bot's topic position based on its mobility."""
        # Bots have limited mobility compared to humans
        shift = np.random.normal(0, self.topic_mobility, size=5)
        self.topic_position = self.topic_position + shift

        # Normalize to stay in reasonable bounds
        norm = np.linalg.norm(self.topic_position)
        if norm > 2.0:
            self.topic_position = self.topic_position * (2.0 / norm)

    def bot_to_bot_interaction(self, other_bot):
        """Bot-to-bot interaction - mostly networking effects."""
        # Bot networks will have a higher chance of being banned
        # This increases ban probability for both bots
        self.ban_probability += 0.005
        other_bot.ban_probability += 0.005

        # Bots in the same network might adopt similar topic positions
        if random.random() < 0.3:
            # Move topic positions slightly closer together
            midpoint = (self.topic_position + other_bot.topic_position) / 2
            self.topic_position = self.topic_position * 0.8 + midpoint * 0.2
            other_bot.topic_position = other_bot.topic_position * 0.8 + midpoint * 0.2

        # Connect the bots to form bot networks
        if other_bot.unique_id not in self.connections:
            self.add_connection(other_bot)
"""
BotAgent.py - Implementation of bot agents for social media simulation
using Mesa 3.1.4
"""

from SocialMediaAgent import SocialMediaAgent
import numpy as np
from datetime import date, timedelta


class BotAgent(SocialMediaAgent):
    """Bot agent in the social media simulation."""

    def __init__(self, model):
        # Use model's RNG for reproducibility
        self.bot_type = model.random.choice(["spam", "misinformation", "astroturfing"])

        # Pass only model to super().__init__ in Mesa 3.1.4
        super().__init__(model=model, post_type=self.bot_type)
        self.agent_type = "bot"

        # Bot characteristics - use model's RNG
        self.detection_difficulty = model.random.uniform(0.3, 0.9)  # Higher means harder to detect
        self.malicious_post_rate = model.random.uniform(0.01, 1)
        # Use model's numpy RNG wrapper
        self.topic_position = model.random.normal(0, 1, size=5)  # Fixed topic position
        self.topic_mobility = model.random.uniform(0.0, 0.3)  # Lower mobility than humans

        # Set specific parameters based on bot type
        self.configure_bot_type()

    def configure_bot_type(self):
        """Configure bot parameters based on type."""
        if self.bot_type == "misinformation":
            self.post_frequency = self.model.random.uniform(0.2, 0.8)
            self.detection_difficulty = self.model.random.uniform(0.4, 0.7)
            # self.connection_strategy = model.random.choice(["targeted", "echo_chamber", "random", "broadcast"])
        elif self.bot_type == "spam":
            self.post_frequency = self.model.random.uniform(0.5, 0.99)
            self.detection_difficulty = self.model.random.uniform(0.1, 0.2)
            # self.connection_strategy = model.random.choice(["broadcast", "random"])
        elif self.bot_type == "astroturfing":
            self.post_frequency = self.model.random.uniform(0.2, 0.8)
            self.detection_difficulty = self.model.random.uniform(0.4, 0.7)
            # self.connection_strategy = model.random.choice(["targeted", "echo_chamber", "random", "broadcast"])

    def step(self):
        """Bot agent behavior during each step."""
        super().step()

        if not self.active:
            return

        # Post with some probability
        self.bot_post()

        # Check if bot gets banned
        self.check_ban()

        # Possibly shift topic position slightly (limited mobility)
        if self.model.random.random() < 0.05:  # 5% chance to shift
            self.shift_topic()

    def check_ban(self):
        """Check if the bot gets banned."""
        # Use model's RNG for reproducibility
        if self.model.random.random() < (1 - self.detection_difficulty) / 10:
            self.deactivate()

    def bot_post(self):
        """Create a post with some probability."""
        # Use model's RNG
        if self.model.random.random() < self.post_frequency:
            self.posted_today = True
            # Decide if post should be malicious
            if self.model.random.random() < self.malicious_post_rate:
                self.create_malicious_post()
            else:
                self.attempt_normal_post()
        else:
            self.posted_today = False

    def create_malicious_post(self):
        """Create a malicious post based on bot type."""
        self.post_type = self.bot_type

    def attempt_normal_post(self):
        """Create a normal post to avoid detection."""
        self.post_type = "normal"

    def shift_topic(self):
        """Slightly shift the bot's topic position (with limited mobility)."""
        # Small random shift in topic position - use model's numpy RNG wrapper
        shift = self.model.random.normal(0, self.topic_mobility, size=5)
        self.topic_position = self.topic_position + shift

        # Normalize to stay in reasonable bounds
        norm = np.linalg.norm(self.topic_position)
        if norm > 2.0:
            self.topic_position = self.topic_position * (2.0 / norm)
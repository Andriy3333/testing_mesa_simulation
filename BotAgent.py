import mesa
from SocialMediaAgent import SocialMediaAgent
import random
import numpy as np
import constants
from datetime import date, timedelta


class BotAgent(SocialMediaAgent):
    """Bot agent in the social media simulation."""

    def __init__(self, unique_id, model):
        self.bot_type = random.choice(["spam", "misinformation", "astroturfing"])

        super().__init__(unique_id, model, post_type = self.bot_type)
        self.agent_type = "bot"

        # Bot characteristics
        self.detection_difficulty = random.uniform(3, 9)  # Higher means harder to detect
        self.malicious_post_rate = random.uniform(0.01, 1)
        self.topic_position = np.random.normal(0, 1, size=5)  # Fixed topic position
        self.topic_mobility = random.uniform(0.0, 0.3)  # Lower mobility than humans

        # Bot network parameters. Implement later
        #self.connection_strategy = random.choice(["random", "targeted", "broadcast", "echo_chamber"])

        # Bot activity parameters
        #self.post_frequency = random.uniform(0.3, 0.95)  # Higher posting rate than humans
        #self.popularity = random.uniform(0.01, 0.5)

        # Set specific parameters based on bot type
        self.configure_bot_type()


    def configure_bot_type(self):
        """Configure bot parameters based on type."""
        if self.bot_type == "misinformation":
            self.post_frequency = random.uniform(0.2, 0.8)
            self.detection_difficulty = random.uniform(0.4, 0.7)
            self.popularity
           #self.connection_strategy = random.choice(["targeted", "echo_chamber", "random", "broadcast"])
        elif self.bot_type == "spam":
            self.post_frequency = random.uniform(0.5, 0.99)
            self.detection_difficulty = random.uniform(0.1, 0.2)
            #self.connection_strategy = random.choice(["broadcast", "random"])
        elif self.bot_type == "astroturfing":
            self.post_frequency = random.uniform(0.2, 0.8)
            self.detection_difficulty = random.uniform(0.4, 0.7)
           #self.connection_strategy = random.choice(["targeted", "echo_chamber", "random", "broadcast"])

    def step(self):
        """Bot agent behavior during each step."""
        super().step()

        self.check_ban()

        # Possibly shift topic position slightly (limited mobility)
        if random.random() < 0.05:  # 5% chance to shift
            self.shift_topic()

    def check_ban(self):
        if random.random() < (1 - self.detection_difficulty) / 10:
            self.deactivate()
            return


    def bot_post(self):
        if random.random() < self.post_frequency:
            self.posted_today = True
            if random.random() < self.malicious_rate:
                self.create_malicious_post()
            else:
                return
        else:
            self.posted_today = False

    def create_malicious_post(self):
        self.post_type = self.bot_type

    def attempt_normal_post(self):
        self.post_type = "normal"


    # def calculate_topic_similarity(self, other_agent):
    #     """Calculate similarity in topic space between this bot and another agent."""
    #     # Cosine similarity between topic vectors
    #     dot_product = np.dot(self.topic_position, other_agent.topic_position)
    #     norm_product = np.linalg.norm(self.topic_position) * np.linalg.norm(other_agent.topic_position)
    #     similarity = dot_product / (norm_product + 1e-8)
    #
    #     # Convert from [-1, 1] to [0, 1] range
    #     return (similarity + 1) / 2


    # def bot_to_bot_interaction(self, other_bot):
    #     """Bot-to-bot interaction - mostly networking effects."""
    #     # Bot networks will have a higher chance of being banned
    #     # This increases ban probability for both bots
    #     self.ban_probability += 0.005
    #     other_bot.ban_probability += 0.005
    #
    #     # Bots in the same network might adopt similar topic positions
    #     if random.random() < 0.3:
    #         # Move topic positions slightly closer together
    #         midpoint = (self.topic_position + other_bot.topic_position) / 2
    #         self.topic_position = self.topic_position * 0.8 + midpoint * 0.2
    #         other_bot.topic_position = other_bot.topic_position * 0.8 + midpoint * 0.2
    #
    #     # Connect the bots to form bot networks
    #     if other_bot.unique_id not in self.connections:
    #         self.add_connection(other_bot)
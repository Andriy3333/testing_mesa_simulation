"""
visualization.py - Solara visualization for social media simulation
using Mesa 3.1.4 and Solara 1.44.1
"""

import solara
import numpy as np
import networkx as nx
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import threading

from model import SmallWorldNetworkModel
from mesa.visualization import SolaraViz, make_space_component, make_plot_component


# Function to visualize the network
def network_visualization(model):
    """Creates a network visualization of the social media model"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a graph from the agent connections
    G = nx.Graph()

    # Add nodes
    active_agents = [agent for agent in model.agents if agent.active]
    for agent in active_agents:
        G.add_node(agent.unique_id,
                   agent_type=agent.agent_type,
                   satisfaction=getattr(agent, "satisfaction", 0))

    # Add edges from connections
    for agent in active_agents:
        for connection_id in agent.connections:
            if G.has_node(connection_id):  # Make sure the connection exists
                G.add_edge(agent.unique_id, connection_id)

    # Position nodes using a layout algorithm
    pos = nx.spring_layout(G, seed=model.random.randint(0, 2**32-1))

    # Get node colors based on agent type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        agent_type = G.nodes[node]['agent_type']
        if agent_type == 'human':
            node_colors.append('blue')
            node_sizes.append(50)
        else:  # bot
            node_colors.append('red')
            node_sizes.append(50)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)

    # Add legend
    ax.plot([0], [0], 'o', color='blue', label='Human')
    ax.plot([0], [0], 'o', color='red', label='Bot')
    ax.legend()

    ax.set_title(f"Social Network (Step {model.steps})")
    ax.axis('off')

    return fig


# Function to visualize satisfaction distribution
def satisfaction_histogram(model):
    """Creates a histogram of human satisfaction levels"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Get satisfaction values from active human agents
    satisfaction_values = [
        agent.satisfaction for agent in model.agents
        if getattr(agent, "agent_type", "") == "human" and agent.active
    ]

    if satisfaction_values:
        # Create histogram
        ax.hist(satisfaction_values, bins=10, range=(0, 100), alpha=0.7, color='green')
        ax.set_title(f"Human Satisfaction Distribution (Step {model.steps})")
        ax.set_xlabel("Satisfaction Level")
        ax.set_ylabel("Number of Humans")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No active human agents",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)

    return fig


# Create a Solara dashboard component
@solara.component
def SocialMediaDashboard():
    # Define sliders for adjustable parameters with reactive values
    num_initial_humans = solara.use_reactive(100)
    num_initial_bots = solara.use_reactive(20)
    human_creation_rate = solara.use_reactive(0.1)
    bot_creation_rate = solara.use_reactive(0.05)
    connection_rewiring_prob = solara.use_reactive(0.1)
    topic_shift_frequency = solara.use_reactive(30)

    # Additional parameters from constants
    human_human_positive_bias = solara.use_reactive(0.7)
    human_bot_negative_bias = solara.use_reactive(0.8)
    human_satisfaction_init = solara.use_reactive(100)

    # Simulation control values
    is_running = solara.use_reactive(False)
    step_size = solara.use_reactive(1)
    seed = solara.use_reactive(42)

    # Reactive model to update visualization
    model = solara.use_reactive(None)

    # Store model data as a list of dictionaries instead of directly as a DataFrame
    # to prevent re-render loops
    model_data_list = solara.use_reactive([])

    # Update counter for triggering re-renders
    update_counter = solara.use_reactive(0)

    # Function to initialize the model
    def initialize_model():
        # Override certain constants based on slider values
        import constants
        constants.HUMAN_HUMAN_POSITIVE_BIAS = human_human_positive_bias.value
        constants.HUMAN_BOT_NEGATIVE_BIAS = human_bot_negative_bias.value
        constants.DEFAULT_HUMAN_SATISFACTION_INIT = human_satisfaction_init.value

        # Create the model
        new_model = SmallWorldNetworkModel(
            num_initial_humans=num_initial_humans.value,
            num_initial_bots=num_initial_bots.value,
            human_creation_rate=human_creation_rate.value,
            bot_creation_rate=bot_creation_rate.value,
            connection_rewiring_prob=connection_rewiring_prob.value,
            topic_shift_frequency=topic_shift_frequency.value,
            seed=seed.value
        )

        # Reset data
        model_data_list.value = []

        return new_model

    # Initialize the model
    if model.value is None:
        model.value = initialize_model()

    # Function to run a single step
    def step():
        if model.value:
            # Step the model
            model.value.step()

            # Get current data as a dictionary
            df_row = model.value.datacollector.get_model_vars_dataframe().iloc[-1:].to_dict('records')[0]
            df_row['step'] = model.value.steps

            # Update data list using a new list to avoid reference issues
            model_data_list.value = model_data_list.value + [df_row]

            # Increment update counter to trigger re-renders
            update_counter.value += 1

    # Function to run multiple steps
    def run_steps():
        for _ in range(step_size.value):
            step()

    # Reset button function
    def reset():
        is_running.value = False
        model.value = initialize_model()
        update_counter.value += 1

    # Background auto-stepping using a side effect
    def auto_step_effect():
        # Only set up auto-stepping when running is true
        if not is_running.value:
            return None

        # Function to execute in a timer
        def timer_callback():
            # This runs in a background thread
            if is_running.value:
                # Use solara.patch to safely update from a background thread
                solara.patch(lambda: run_steps())

                # Schedule the next step after a delay
                threading.Timer(1.0, timer_callback).start()

        # Start the timer
        threading.Timer(1.0, timer_callback).start()

        # Return a cleanup function
        return lambda: None  # No cleanup needed

    # Set up the auto-stepping effect when is_running changes
    solara.use_effect(auto_step_effect, [is_running.value])

    # Convert model_data_list to DataFrame for plotting
    def get_model_dataframe():
        if model_data_list.value:
            return pd.DataFrame(model_data_list.value)
        return pd.DataFrame()

    # Create the dashboard layout
    with solara.Column():
        with solara.Row():
            # Parameter sliders column - INCREASED WIDTH FROM 1/4 to 1/3
            with solara.Column(classes=["w-1/3"]):
                with solara.Card(title="Simulation Parameters", subtitle="Adjust parameters for the model"):
                    solara.SliderInt(label="Initial Humans", value=num_initial_humans, min=10, max=500)
                    solara.SliderInt(label="Initial Bots", value=num_initial_bots, min=0, max=200)
                    solara.SliderFloat(label="Human Creation Rate", value=human_creation_rate, min=0.0, max=1.0, step=0.01)
                    solara.SliderFloat(label="Bot Creation Rate", value=bot_creation_rate, min=0.0, max=1.0, step=0.01)
                    solara.SliderFloat(label="Connection Rewiring Probability", value=connection_rewiring_prob, min=0.0, max=1.0, step=0.01)
                    solara.SliderInt(label="Topic Shift Frequency (steps)", value=topic_shift_frequency, min=1, max=100)
                    solara.SliderFloat(label="Human-Human Positive Bias", value=human_human_positive_bias, min=0.0, max=1.0, step=0.01)
                    solara.SliderFloat(label="Human-Bot Negative Bias", value=human_bot_negative_bias, min=0.0, max=1.0, step=0.01)
                    solara.SliderInt(label="Initial Human Satisfaction", value=human_satisfaction_init, min=0, max=100)
                    solara.SliderInt(label="Random Seed", value=seed, min=0, max=1000)

            # Social Network Graph - ADJUSTED WIDTH FROM 2/4 to 2/5
            with solara.Column(classes=["w-2/5"]):
                if model.value:
                    solara.FigureMatplotlib(network_visualization(model.value))

            # Satisfaction Histogram - ADJUSTED TO MATCH PROPORTIONS
            with solara.Column(classes=["w-4/15"]):
                if model.value:
                    solara.FigureMatplotlib(satisfaction_histogram(model.value))

        # Center-aligned row for controls and state
        with solara.Row(justify="center"):
            # Simulation controls
            with solara.Column(classes=["w-3/5"]):
                with solara.Card(title="Simulation Controls"):
                    # First row of controls
                    with solara.Row():
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(label="Initialize", on_click=reset)
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(label="Step", on_click=step)
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(
                                label="Run" if not is_running.value else "Pause",
                                on_click=lambda: setattr(is_running, "value", not is_running.value)
                            )

                    # Second row with just the step slider
                    with solara.Row():
                        solara.SliderInt(label="Steps per Click", value=step_size, min=1, max=30)

            # Small spacer column
            with solara.Column(classes=["w-1/20"]):
                pass

            # Current state display
            with solara.Column(classes=["w-1/4"]):
                if model.value:
                    with solara.Card(title="Current State"):
                        with solara.Row():
                            solara.Info(
                                f"Step: {model.value.steps} | "
                                f"Active Humans: {model.value.active_humans} | "
                                f"Active Bots: {model.value.active_bots} | "
                                f"Avg Satisfaction: {model.value.get_avg_human_satisfaction():.1f}"
                            )

        # Create time series plots
        df = get_model_dataframe()

        with solara.Columns([1, 1]):
            if model.value and not df.empty:
                # Line plots of key metrics
                with solara.Card(title="Population Over Time"):
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(df['step'], df['Active Humans'], label='Humans')
                    ax1.plot(df['step'], df['Active Bots'], label='Bots')
                    ax1.set_xlabel('Step')
                    ax1.set_ylabel('Count')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    solara.FigureMatplotlib(fig1)

                # Satisfaction over time
                with solara.Card(title="Satisfaction Over Time"):
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(df['step'], df['Average Human Satisfaction'], color='green')
                    ax2.set_xlabel('Step')
                    ax2.set_ylabel('Satisfaction Level')
                    ax2.set_ylim(0, 100)
                    ax2.grid(True, alpha=0.3)
                    solara.FigureMatplotlib(fig2)


# Main app
@solara.component
def Page():
    SocialMediaDashboard()


# When running with `solara run visualization.py`, this will be used
if __name__ == "__main__":
    # No need to call solara.run() as the CLI will handle that
    pass
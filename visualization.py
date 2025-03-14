"""
visualization.py - Solara visualization for social media simulation
using Mesa 3.1.4
"""

import solara
import numpy as np
import networkx as nx
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

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


# Create interval handler outside of the component to avoid the nested function warning
@solara.component
def IntervalHandler(callback, interval_ms=1000):
    interval = solara.use_interval(interval_ms)  # noqa: SH104

    if interval.value:
        callback()


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

    # Data for plots
    model_data = solara.use_reactive(pd.DataFrame())

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
        model_data.value = pd.DataFrame()

        return new_model

    # Initialize the model
    if model.value is None:
        model.value = initialize_model()

    # Function to run a single step
    def step():
        if model.value:
            # Step the model
            model.value.step()

            # Update data
            current_data = model.value.datacollector.get_model_vars_dataframe().iloc[-1:]

            if model_data.value.empty:
                model_data.value = current_data
            else:
                model_data.value = pd.concat([model_data.value, current_data])

    # Function to run multiple steps
    def run_steps():
        for _ in range(step_size.value):
            step()

    # Reset button function
    def reset():
        is_running.value = False
        model.value = initialize_model()

    # Create the dashboard layout
    with solara.Column():
        with solara.Row():
            # Parameter sliders column
            with solara.Column(classes=["w-1/4"]):
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

            # Social Network Graph
            with solara.Column(classes=["w-2/4"]):
                if model.value:
                    solara.FigureMatplotlib(network_visualization(model.value))

            # Satisfaction Histogram
            with solara.Column(classes=["w-1/4"]):
                if model.value:
                    solara.FigureMatplotlib(satisfaction_histogram(model.value))

        # Center-aligned row for controls and state
        with solara.Row(justify="center"):
            # Simulation controls
            with solara.Column(classes=["w-2/5"]):
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
            with solara.Column(classes=["w-2/5"]):
                if model.value:
                    with solara.Card(title="Current State"):
                        with solara.Row():
                            solara.Info(
                                f"Step: {model.value.steps} | "
                                f"Active Humans: {model.value.active_humans} | "
                                f"Active Bots: {model.value.active_bots} | "
                                f"Avg Satisfaction: {model.value.get_avg_human_satisfaction():.1f}"
                            )

        # Add Auto Run component using the separate handler
        if is_running.value:
            IntervalHandler(run_steps, 1000)

        # Create time series plots
        with solara.Columns([1, 1]):
            if model.value and not model_data.value.empty:
                # Line plots of key metrics
                with solara.Card(title="Population Over Time"):
                    solara.FigurePlotly(
                        pd.DataFrame({
                            'Step': model_data.value.index,
                            'Humans': model_data.value["Active Humans"],
                            'Bots': model_data.value["Active Bots"]
                        }).plot(
                            x='Step',
                            y=['Humans', 'Bots'],
                            labels={"value": "Count", "variable": "Agent Type"}
                        )
                    )

                # Satisfaction over time
                with solara.Card(title="Satisfaction Over Time"):
                    solara.FigurePlotly(
                        pd.DataFrame({
                            'Step': model_data.value.index,
                            'Satisfaction': model_data.value["Average Human Satisfaction"]
                        }).plot(
                            x='Step',
                            y='Satisfaction',
                            labels={"value": "Satisfaction Level"}
                        )
                    )


# Main app
@solara.component
def Page():
    SocialMediaDashboard()


# When running with `solara run visualization.py`, this will be used
if __name__ == "__main__":
    # No need to call solara.run() as the CLI will handle that
    pass
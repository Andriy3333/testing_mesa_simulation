"""
visualization.py - Visualization of social media simulation
using Mesa 3.1.4 SolaraViz API with Solara

Run with: solara run visualization.py
"""

import mesa
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
import networkx as nx
import solara
import solara.components as sc
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from model import SmallWorldNetworkModel
import math
import pandas as pd


def agent_portrayal(agent):
    """
    Define how agents are drawn in the network visualization.

    Parameters:
    -----------
    agent : SocialMediaAgent
        The agent to portray

    Returns:
    --------
    dict : A dictionary with portrayal properties
    """
    portrayal = {
        "shape": "circle",
        "filled": True,
        "r": 5,
    }

    # Use different colors for different agent types and states
    if getattr(agent, "agent_type", "") == "human":
        if agent.active:
            # Color by satisfaction level
            satisfaction = getattr(agent, "satisfaction", 50)
            # Gradient from red (0) to green (100)
            color_val = min(1.0, max(0.0, satisfaction / 100))
            portrayal["color"] = f"rgb({int(255 * (1 - color_val))}, {int(255 * color_val)}, 0)"
            portrayal["r"] = 5
        else:  # Inactive human
            portrayal["color"] = "gray"
            portrayal["r"] = 3
    elif getattr(agent, "agent_type", "") == "bot":
        if agent.active:
            # Different colors for different bot types
            bot_type = getattr(agent, "bot_type", "unknown")
            if bot_type == "spam":
                portrayal["color"] = "purple"
            elif bot_type == "misinformation":
                portrayal["color"] = "red"
            elif bot_type == "astroturfing":
                portrayal["color"] = "orange"
            else:
                portrayal["color"] = "blue"
            portrayal["r"] = 7  # Larger than humans
        else:  # Inactive bot
            portrayal["color"] = "darkgray"
            portrayal["r"] = 4

    # Add information for tooltips
    portrayal["tooltip"] = f"ID: {agent.unique_id}<br>"
    portrayal["tooltip"] += f"Type: {getattr(agent, 'agent_type', 'Unknown')}<br>"

    if getattr(agent, "agent_type", "") == "human":
        portrayal["tooltip"] += f"Satisfaction: {getattr(agent, 'satisfaction', 0):.1f}<br>"
    elif getattr(agent, "agent_type", "") == "bot":
        portrayal["tooltip"] += f"Bot Type: {getattr(agent, 'bot_type', 'Unknown')}<br>"

    portrayal["tooltip"] += f"Connections: {len(getattr(agent, 'connections', []))}<br>"
    portrayal["tooltip"] += f"Active: {getattr(agent, 'active', False)}"

    return portrayal


def network_space_portrayal(G):
    """
    Define the network space portrayal for SolaraViz.
    Uses a force-directed layout algorithm.

    Parameters:
    -----------
    G : NetworkX Graph
        The graph being drawn

    Returns:
    --------
    dict : A dictionary with the portrayal specifications
    """
    # Use force-directed layout
    portrayal = {
        "nodes": [],
        "edges": [],
    }

    # Compute positions using NetworkX's spring layout
    # Note: G might be empty in some edge cases
    if G and len(G.nodes) > 0:
        pos = nx.spring_layout(G, iterations=100, seed=42)  # Fixed seed for consistent layout

        # Add nodes
        for node_id, position in pos.items():
            portrayal["nodes"].append({
                "id": node_id,
                "x": (position[0] + 1) / 2,  # Scale to [0, 1]
                "y": (position[1] + 1) / 2,  # Scale to [0, 1]
            })

        # Add edges
        for source, target in G.edges():
            portrayal["edges"].append({
                "source": source,
                "target": target,
            })

    return portrayal


@solara.component
def agent_count_chart(model):
    """
    Create an agent count chart component.

    Parameters:
    -----------
    model : SmallWorldNetworkModel
        The model instance

    Returns:
    --------
    sc.Figure : Solara figure component with the chart
    """
    # Convert model data to pandas DataFrame
    model_df = model.datacollector.get_model_vars_dataframe()

    # If we don't have enough data yet, return empty figure
    if len(model_df) < 2:
        fig = plt.figure(figsize=(6, 3))
        plt.title("Waiting for data...")
        return sc.Figure(fig=fig)

    # Create chart
    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()

    model_df[["Active Humans", "Active Bots"]].plot(ax=ax)
    ax.set_xlabel("Steps (Days)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return sc.Figure(fig=fig)


@solara.component
def satisfaction_chart(model):
    """
    Create a satisfaction chart component.

    Parameters:
    -----------
    model : SmallWorldNetworkModel
        The model instance

    Returns:
    --------
    sc.Figure : Solara figure component with the chart
    """
    # Convert model data to pandas DataFrame
    model_df = model.datacollector.get_model_vars_dataframe()

    # If we don't have enough data yet, return empty figure
    if len(model_df) < 2:
        fig = plt.figure(figsize=(6, 3))
        plt.title("Waiting for data...")
        return sc.Figure(fig=fig)

    # Create chart
    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()

    model_df["Average Human Satisfaction"].plot(ax=ax, color="green")
    ax.set_xlabel("Steps (Days)")
    ax.set_ylabel("Satisfaction Level")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    return sc.Figure(fig=fig)


@solara.component
def deactivated_chart(model):
    """
    Create a deactivated agents chart component.

    Parameters:
    -----------
    model : SmallWorldNetworkModel
        The model instance

    Returns:
    --------
    sc.Figure : Solara figure component with the chart
    """
    # Convert model data to pandas DataFrame
    model_df = model.datacollector.get_model_vars_dataframe()

    # If we don't have enough data yet, return empty figure
    if len(model_df) < 2:
        fig = plt.figure(figsize=(6, 3))
        plt.title("Waiting for data...")
        return sc.Figure(fig=fig)

    # Create chart
    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()

    model_df[["Deactivated Humans", "Deactivated Bots"]].plot(ax=ax)
    ax.set_xlabel("Steps (Days)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return sc.Figure(fig=fig)


@solara.component
def stats_panel(model):
    """
    Create a panel with current stats.

    Parameters:
    -----------
    model : SmallWorldNetworkModel
        The model instance

    Returns:
    --------
    solara.Column : A column with stats
    """
    with solara.Card(title="Current Statistics") as main:
        with solara.Column(gap="5px"):
            solara.Markdown(f"**Steps (Days):** {model.steps}")
            solara.Markdown(f"**Active Humans:** {model.active_humans}")
            solara.Markdown(f"**Active Bots:** {model.active_bots}")
            solara.Markdown(f"**Deactivated Humans:** {model.deactivated_humans}")
            solara.Markdown(f"**Deactivated Bots:** {model.deactivated_bots}")
            solara.Markdown(f"**Avg Human Satisfaction:** {model.get_avg_human_satisfaction():.2f}")

    return main


# Function is no longer needed as we're using the Page component directly
# It's replaced by the Page component at the module level


# Initialize model as a reactive value so it can be updated by components
# This needs to be at the module level for Solara
default_params = {
    "num_initial_humans": 100,
    "num_initial_bots": 20,
    "human_creation_rate": 0.1,
    "bot_creation_rate": 0.05,
    "topic_shift_frequency": 30,
    "seed": 42
}

# Create a reactive model instance
model_instance = solara.reactive(SmallWorldNetworkModel(**default_params))

# User-adjustable model parameters
num_humans = solara.reactive(100)
num_bots = solara.reactive(20)
human_rate = solara.reactive(0.1)
bot_rate = solara.reactive(0.05)
topic_shift = solara.reactive(30)
seed = solara.reactive(42)

# Function to reset model with current parameters
def reset_model():
    model_instance.value = SmallWorldNetworkModel(
        num_initial_humans=num_humans.value,
        num_initial_bots=num_bots.value,
        human_creation_rate=human_rate.value,
        bot_creation_rate=bot_rate.value,
        topic_shift_frequency=topic_shift.value,
        seed=seed.value
    )

# Function to step the model
def step_model():
    model_instance.value.step()

# Function to run multiple steps
def run_steps(steps=30):
    for _ in range(steps):
        model_instance.value.step()


@solara.component
def Page():
    """
    Main Solara Page component for the visualization.
    This is what Solara will look for when running with 'solara run'.
    """
    # Use grid layout for the page
    with solara.Column(gap="10px") as main:
        solara.Title("Social Media Network Simulation")

        # Parameter controls with collapsible section using standard components
        show_params = solara.reactive(False)

        with solara.Row(justify="center"):
            solara.Checkbox("Show Simulation Parameters", value=show_params)

        # Only show parameters if checkbox is checked
        if show_params.value:
            with solara.Card():
                with solara.Row(gap="20px", classes=["controls"]):
                    with solara.Column(gap="10px", classes=["param-group"]):
                        solara.SliderInt(
                            label="Initial Human Agents",
                            value=num_humans,
                            min=10,
                            max=300,
                            step=10
                        )
                        solara.SliderInt(
                            label="Initial Bot Agents",
                            value=num_bots,
                            min=0,
                            max=100,
                            step=5
                        )

                    with solara.Column(gap="10px", classes=["param-group"]):
                        solara.SliderFloat(
                            label="Human Creation Rate",
                            value=human_rate,
                            min=0.0,
                            max=1.0,
                            step=0.05
                        )
                        solara.SliderFloat(
                            label="Bot Creation Rate",
                            value=bot_rate,
                            min=0.0,
                            max=0.5,
                            step=0.05
                        )

                    with solara.Column(gap="10px", classes=["param-group"]):
                        solara.SliderInt(
                            label="Topic Shift Frequency",
                            value=topic_shift,
                            min=1,
                            max=100,
                            step=5
                        )
                        solara.SliderInt(
                            label="Random Seed",
                            value=seed,
                            min=0,
                            max=1000,
                            step=1
                        )

        # Simulation controls
        with solara.Row(gap="10px", classes=["sim-controls"]):
            solara.Button("Reset", on_click=reset_model, variant="outlined")
            solara.Button("Step", on_click=step_model, variant="outlined")
            solara.Button("Run 10 Steps", on_click=lambda: run_steps(10), variant="outlined")
            solara.Button("Run 30 Steps", on_click=lambda: run_steps(30), color="primary")

        # Main content with the simulation visualization
        with solara.Row(gap="20px", classes=["main-content"]):
            # Left panel - Network visualization and stats
            with solara.Column(gap="10px", classes=["left-panel"]):
                # This will be replaced by the network component from SolaraViz
                with solara.Card(title="Network Visualization"):
                    # Create network space component
                    network_space = make_space_component(
                        agent_portrayal=agent_portrayal,
                        space_portrayal=network_space_portrayal
                    )
                    network_space(model_instance.value)

                # Stats card
                stats_panel(model_instance.value)

            # Right panel - Charts
            with solara.Column(gap="10px", classes=["right-panel"]):
                with solara.Card(title="Active Agents"):
                    agent_count_chart(model_instance.value)

                with solara.Card(title="Human Satisfaction"):
                    satisfaction_chart(model_instance.value)

                with solara.Card(title="Deactivated Agents"):
                    deactivated_chart(model_instance.value)

        # Add CSS for layout styling
        solara.Style("""
        .controls {
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .param-group {
            min-width: 200px;
        }
        .sim-controls {
            margin: 15px 0;
        }
        .main-content {
            height: calc(100vh - 220px);
            min-height: 500px;
        }
        .left-panel {
            width: 50%;
            min-width: 450px;
            height: 100%;
        }
        .right-panel {
            width: 50%;
            min-width: 450px;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        """)

    return main
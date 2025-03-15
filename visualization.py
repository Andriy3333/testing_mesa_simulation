"""
visualization.py - Solara visualization for social media simulation
using Mesa 3.1.4 and Solara 1.44.1
"""

import solara
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import threading

from model import SmallWorldNetworkModel


# Function to visualize the network
def network_visualization(model):
    """Creates a network visualization of the social media model"""
    # Smaller figure size
    fig, ax = plt.subplots(figsize=(5, 5))

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
            node_sizes.append(40)  # Smaller node size
        else:  # bot
            node_colors.append('red')
            node_sizes.append(40)  # Smaller node size

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)

    # Add legend
    ax.plot([0], [0], 'o', color='blue', label='Human')
    ax.plot([0], [0], 'o', color='red', label='Bot')
    ax.legend(fontsize=8)

    ax.set_title(f"Social Network (Step {model.steps})", fontsize=10)
    ax.axis('off')

    # Add tight layout to make better use of space
    plt.tight_layout()
    return fig


# Function to visualize satisfaction distribution
def satisfaction_histogram(model):
    """Creates a histogram of human satisfaction levels"""
    # Smaller figure size
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Get satisfaction values from active human agents
    satisfaction_values = [
        agent.satisfaction for agent in model.agents
        if getattr(agent, "agent_type", "") == "human" and agent.active
    ]

    if satisfaction_values:
        # Create histogram
        ax.hist(satisfaction_values, bins=10, range=(0, 100), alpha=0.7, color='green')
        ax.set_title(f"Human Satisfaction (Step {model.steps})", fontsize=10)
        ax.set_xlabel("Satisfaction Level", fontsize=8)
        ax.set_ylabel("Number of Humans", fontsize=8)
        ax.set_xlim(0, 100)

        # Set a fixed y-axis limit based on the number of active humans
        # This ensures the scale doesn't change between steps
        max_humans = model.active_humans
        # Use a slightly higher value to account for potential growth
        y_max = max(20, int(max_humans * 1.2))
        ax.set_ylim(0, y_max)

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "No active human agents",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=8)
        # Set default y-limit even when no agents
        ax.set_ylim(0, 20)

    # Add tight layout to make better use of space
    plt.tight_layout()
    return fig

# Create a Solara dashboard component
@solara.component
def SocialMediaDashboard():
    # Model parameters with state
    num_initial_humans, set_num_initial_humans = solara.use_state(100)
    num_initial_bots, set_num_initial_bots = solara.use_state(20)
    human_creation_rate, set_human_creation_rate = solara.use_state(0.1)
    bot_creation_rate, set_bot_creation_rate = solara.use_state(0.05)
    connection_rewiring_prob, set_connection_rewiring_prob = solara.use_state(0.1)
    topic_shift_frequency, set_topic_shift_frequency = solara.use_state(30)
    human_human_positive_bias, set_human_human_positive_bias = solara.use_state(0.7)
    human_bot_negative_bias, set_human_bot_negative_bias = solara.use_state(0.8)
    human_satisfaction_init, set_human_satisfaction_init = solara.use_state(100)
    seed, set_seed = solara.use_state(42)

    # Flag to indicate parameters have changed
    params_changed, set_params_changed = solara.use_state(False)

    # Simulation control values
    is_running, set_is_running = solara.use_state(False)
    step_size, set_step_size = solara.use_state(1)
    model, set_model = solara.use_state(None)
    model_data_list, set_model_data_list = solara.use_state([])
    update_counter, set_update_counter = solara.use_state(0)

    # Function to update a parameter and mark as changed
    def update_param(value, setter):
        setter(value)
        set_params_changed(True)

    # Function to create a new model with current parameters
    def create_new_model():
        print(f"\n=== Creating new model with parameters ===")
        print(f"Initial Humans: {num_initial_humans}")
        print(f"Initial Bots: {num_initial_bots}")
        print(f"Human Creation Rate: {human_creation_rate}")
        print(f"Bot Creation Rate: {bot_creation_rate}")
        print(f"Connection Rewiring: {connection_rewiring_prob}")
        print(f"Topic Shift Frequency: {topic_shift_frequency}")
        print(f"Human-Human Bias: {human_human_positive_bias}")
        print(f"Human-Bot Bias: {human_bot_negative_bias}")
        print(f"Initial Satisfaction: {human_satisfaction_init}")
        print(f"Seed: {seed}")

        new_model = SmallWorldNetworkModel(
            num_initial_humans=num_initial_humans,
            num_initial_bots=num_initial_bots,
            human_creation_rate=human_creation_rate,
            bot_creation_rate=bot_creation_rate,
            connection_rewiring_prob=connection_rewiring_prob,
            topic_shift_frequency=topic_shift_frequency,
            human_human_positive_bias=human_human_positive_bias,
            human_bot_negative_bias=human_bot_negative_bias,
            human_satisfaction_init=human_satisfaction_init,
            seed=seed
        )
        return new_model

    # Function to initialize the model
    def initialize_model():
        # Reset data
        set_model_data_list([])
        # Create new model
        new_model = create_new_model()
        # Clear the params_changed flag
        set_params_changed(False)
        return new_model

    # Initialize the model if it's None
    if model is None:
        set_model(initialize_model())

    # Function to run a single step
    def step():
        if model:
            # Step the model
            model.step()

            # Get current data as a dictionary
            df_row = model.datacollector.get_model_vars_dataframe().iloc[-1:].to_dict('records')[0]
            df_row['step'] = model.steps

            # Update data list
            set_model_data_list(model_data_list + [df_row])

            # Increment update counter to trigger re-renders
            set_update_counter(update_counter + 1)

    # Function to run multiple steps
    def run_steps():
        for _ in range(step_size):
            step()

    # Reset button function
    def reset():
        set_is_running(False)
        set_model(initialize_model())
        set_update_counter(update_counter + 1)

    # Background auto-stepping using a side effect
    def auto_step_effect():
        # Only set up auto-stepping when running is true
        if not is_running:
            return None

        # Function to execute in a timer
        def timer_callback():
            # This runs in a background thread
            if is_running:
                # Use solara.patch to safely update from a background thread
                solara.patch(lambda: run_steps())

                # Schedule the next step after a delay
                threading.Timer(1.0, timer_callback).start()

        # Start the timer
        threading.Timer(1.0, timer_callback).start()

        # Return a cleanup function
        return lambda: None  # No cleanup needed

    # Set up the auto-stepping effect when is_running changes
    solara.use_effect(auto_step_effect, [is_running])

    # Convert model_data_list to DataFrame for plotting
    def get_model_dataframe():
        if model_data_list:
            return pd.DataFrame(model_data_list)
        return pd.DataFrame()

    # Create the dashboard layout
    with solara.Column():
        # First row - Initial Parameters, Graph, and Histogram
        with solara.Row():
            # Initial parameters column (left)
            with solara.Column(classes=["w-1/4"]):
                with solara.Card(title="Initial Parameters"):
                    # Add a warning if parameters have changed
                    if params_changed:
                        solara.Text("Parameters have changed. Click 'Initialize' to apply.")

                    # Initial Population and Growth Rates
                    solara.Markdown("### Initial Population")
                    solara.Text(f"Initial Humans: {num_initial_humans}")
                    solara.SliderInt(
                        label="Initial Humans",
                        min=10,
                        max=500,
                        value=num_initial_humans,
                        on_value=lambda v: update_param(v, set_num_initial_humans)
                    )

                    solara.Text(f"Initial Bots: {num_initial_bots}")
                    solara.SliderInt(
                        label="Initial Bots",
                        min=0,
                        max=200,
                        value=num_initial_bots,
                        on_value=lambda v: update_param(v, set_num_initial_bots)
                    )

                    solara.Markdown("### Growth Rates")
                    solara.Text(f"Human Creation Rate: {human_creation_rate:.2f}")
                    solara.SliderFloat(
                        label="Human Creation Rate",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=human_creation_rate,
                        on_value=lambda v: update_param(v, set_human_creation_rate)
                    )

                    solara.Text(f"Bot Creation Rate: {bot_creation_rate:.2f}")
                    solara.SliderFloat(
                        label="Bot Creation Rate",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=bot_creation_rate,
                        on_value=lambda v: update_param(v, set_bot_creation_rate)
                    )

            # Social Network Graph (middle)
            with solara.Column(classes=["w-3/8"]):
                if model:
                    solara.FigureMatplotlib(network_visualization(model))

            # Satisfaction Histogram (right)
            with solara.Column(classes=["w-3/8"]):
                if model:
                    solara.FigureMatplotlib(satisfaction_histogram(model))

        # Second row - Network Parameters and Simulation Controls
        with solara.Row():
            # Network & Interactions - much wider now (left)
            with solara.Column(classes=["w-3/4"]):
                with solara.Card(title="Network & Interactions"):
                    with solara.Row():
                        # Column 1 of parameters
                        with solara.Column(classes=["w-1/3"]):
                            solara.Text(f"Connection Rewiring: {connection_rewiring_prob:.2f}")
                            solara.SliderFloat(
                                label="Connection Rewiring",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=connection_rewiring_prob,
                                on_value=lambda v: update_param(v, set_connection_rewiring_prob)
                            )

                            solara.Text(f"Topic Shift Frequency: {topic_shift_frequency}")
                            solara.SliderInt(
                                label="Topic Shift Frequency",
                                min=1,
                                max=100,
                                value=topic_shift_frequency,
                                on_value=lambda v: update_param(v, set_topic_shift_frequency)
                            )

                        # Column 2 of parameters
                        with solara.Column(classes=["w-1/3"]):
                            solara.Text(f"Human-Human Positive Bias: {human_human_positive_bias:.2f}")
                            solara.SliderFloat(
                                label="Human-Human Positive Bias",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=human_human_positive_bias,
                                on_value=lambda v: update_param(v, set_human_human_positive_bias)
                            )

                            solara.Text(f"Human-Bot Negative Bias: {human_bot_negative_bias:.2f}")
                            solara.SliderFloat(
                                label="Human-Bot Negative Bias",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=human_bot_negative_bias,
                                on_value=lambda v: update_param(v, set_human_bot_negative_bias)
                            )

                        # Column 3 of parameters
                        with solara.Column(classes=["w-1/3"]):
                            solara.Text(f"Initial Human Satisfaction: {human_satisfaction_init}")
                            solara.SliderInt(
                                label="Initial Human Satisfaction",
                                min=0,
                                max=100,
                                value=human_satisfaction_init,
                                on_value=lambda v: update_param(v, set_human_satisfaction_init)
                            )

                            solara.Text(f"Random Seed: {seed}")
                            solara.SliderInt(
                                label="Random Seed",
                                min=0,
                                max=1000,
                                value=seed,
                                on_value=lambda v: update_param(v, set_seed)
                            )

            # Simulation Controls and Current State - moved far right
            with solara.Column(classes=["w-1/4"]):
                # Simulation controls
                with solara.Card(title="Simulation Controls"):
                    # First row of controls
                    with solara.Row():
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(
                                label="Initialize" if not params_changed else "Initialize (Apply Changes)",
                                on_click=reset
                            )
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(label="Step", on_click=step)
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(
                                label="Run" if not is_running else "Pause",
                                on_click=lambda: set_is_running(not is_running)
                            )

                    # Second row with steps slider
                    with solara.Row():
                        solara.Text(f"Steps per Click: {step_size}")
                        solara.SliderInt(
                            label="Steps per Click",
                            min=1,
                            max=30,
                            value=step_size,
                            on_value=lambda v: set_step_size(v)
                        )

                # Current state display
                if model:
                    with solara.Card(title="Current State"):
                        with solara.Row():
                            solara.Info(
                                f"Step: {model.steps} | "
                                f"Active Humans: {model.active_humans} | "
                                f"Active Bots: {model.active_bots} | "
                                f"Avg Satisfaction: {model.get_avg_human_satisfaction():.1f}"
                            )

        # Third row - Time series plots
        df = get_model_dataframe()
        with solara.Row():
            with solara.Column(classes=["w-1/2"]):
                if model and not df.empty:
                    # Line plots of key metrics
                    with solara.Card(title="Population Over Time"):
                        fig1, ax1 = plt.subplots(figsize=(6, 3))  # Smaller figure
                        ax1.plot(df['step'], df['Active Humans'], label='Humans')
                        ax1.plot(df['step'], df['Active Bots'], label='Bots')
                        ax1.set_xlabel('Step', fontsize=8)
                        ax1.set_ylabel('Count', fontsize=8)
                        ax1.legend(fontsize=8)
                        ax1.grid(True, alpha=0.3)
                        ax1.tick_params(labelsize=8)
                        plt.tight_layout()
                        solara.FigureMatplotlib(fig1)

            with solara.Column(classes=["w-1/2"]):
                if model and not df.empty:
                    # Satisfaction over time
                    with solara.Card(title="Satisfaction Over Time"):
                        fig2, ax2 = plt.subplots(figsize=(6, 3))  # Smaller figure
                        ax2.plot(df['step'], df['Average Human Satisfaction'], color='green')
                        ax2.set_xlabel('Step', fontsize=8)
                        ax2.set_ylabel('Satisfaction Level', fontsize=8)
                        ax2.set_ylim(0, 100)
                        ax2.grid(True, alpha=0.3)
                        ax2.tick_params(labelsize=8)
                        plt.tight_layout()
                        solara.FigureMatplotlib(fig2)


# Main app
@solara.component
def Page():
    SocialMediaDashboard()


# When running with `solara run visualization.py`, this will be used
if __name__ == "__main__":
    # No need to call solara.run() as the CLI will handle that
    pass
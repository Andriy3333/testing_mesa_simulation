"""
run.py - Script to run the social media simulation
using Mesa 3.1.4
"""

import matplotlib.pyplot as plt
import numpy as np
from model import SmallWorldNetworkModel


def run_simulation(steps=365, num_humans=100, num_bots=20, seed=None):
    """
    Run the simulation for the specified number of steps.

    Parameters:
    -----------
    steps : int
        Number of simulation steps (days) to run
    num_humans : int
        Initial number of human agents
    num_bots : int
        Initial number of bot agents
    seed : int, optional
        Random seed for reproducibility
    """
    print(f"Running simulation with {num_humans} humans and {num_bots} bots for {steps} days...")
    if seed is not None:
        print(f"Using random seed: {seed}")

    # Create model with the specified seed
    model = SmallWorldNetworkModel(
        num_initial_humans=num_humans,
        num_initial_bots=num_bots,
        human_creation_rate=0.1,
        bot_creation_rate=0.05,
        topic_shift_frequency=30,
        seed=seed  # Pass the seed to the model
    )

    # Run for specified number of steps
    for i in range(steps):
        if i % 30 == 0:
            print(f"  Step {i}/{steps} - Active humans: {model.active_humans}, Active bots: {model.active_bots}")
        model.step()

    # Collect final data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    print("Simulation complete!")
    print(f"Final statistics:")
    print(f"  Active humans: {model.active_humans}")
    print(f"  Active bots: {model.active_bots}")
    print(f"  Deactivated humans: {model.deactivated_humans}")
    print(f"  Deactivated bots: {model.deactivated_bots}")
    print(f"  Average human satisfaction: {model.get_avg_human_satisfaction():.2f}")

    # Plot results
    plot_results(model_data, seed)

    return model, model_data, agent_data


def plot_results(model_data, seed=None):
    """
    Plot the results of the simulation.

    Parameters:
    -----------
    model_data : DataFrame
        Data collected from the model
    seed : int, optional
        Random seed used (for plot title)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Add title with seed information if provided
    if seed is not None:
        fig.suptitle(f"Social Media Simulation Results (Seed: {seed})", fontsize=14)
    else:
        fig.suptitle("Social Media Simulation Results", fontsize=14)

    # Plot active agents over time
    axes[0].plot(model_data["Active Humans"], label="Active Humans")
    axes[0].plot(model_data["Active Bots"], label="Active Bots")
    axes[0].set_title("Active Agents Over Time")
    axes[0].set_xlabel("Steps (Days)")
    axes[0].set_ylabel("Number of Agents")
    axes[0].legend()
    axes[0].grid(True)

    # Plot average satisfaction over
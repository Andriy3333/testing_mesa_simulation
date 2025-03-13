"""
run_simulation.py - Script to run the social media simulation
"""

import matplotlib.pyplot as plt
import numpy as np
from model import SmallWorldNetworkModel


def run_simulation(steps=365, num_humans=100, num_bots=20):
    """Run the simulation for the specified number of steps."""
    print(f"Running simulation with {num_humans} humans and {num_bots} bots for {steps} days...")

    # Create model
    model = SmallWorldNetworkModel(
        num_initial_humans=num_humans,
        num_initial_bots=num_bots,
        human_creation_rate=0.1,
        bot_creation_rate=0.05,
        topic_shift_frequency=30
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
    plot_results(model_data)

    return model, model_data, agent_data


def plot_results(model_data):
    """Plot the results of the simulation."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot active agents over time
    axes[0].plot(model_data["Active Humans"], label="Active Humans")
    axes[0].plot(model_data["Active Bots"], label="Active Bots")
    axes[0].set_title("Active Agents Over Time")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Number of Agents")
    axes[0].legend()
    axes[0].grid(True)

    # Plot average satisfaction over time
    axes[1].plot(model_data["Average Human Satisfaction"], label="Avg Human Satisfaction", color="green")
    axes[1].set_title("Average Human Satisfaction Over Time")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Satisfaction (0-100)")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("simulation_results.png")
    plt.show()


if __name__ == "__main__":
    # Run simulation for 365 steps (days)
    model, model_data, agent_data = run_simulation(steps=365, num_humans=100, num_bots=20)
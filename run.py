"""
run.py - Script to run the social media simulation
using Mesa 3.1.4
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
from model import SmallWorldNetworkModel


def run_simulation(steps=365, num_humans=100, num_bots=20, seed=None, output_dir="results"):
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
    output_dir : str
        Directory to save results
    """
    print(f"Running simulation with {num_humans} humans and {num_bots} bots for {steps} days...")
    if seed is not None:
        print(f"Using random seed: {seed}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results(model_data, seed, output_dir, timestamp)

    # Save data to CSV
    model_data.to_csv(f"{output_dir}/model_data_{seed}_{timestamp}.csv")

    # Export the latest agent data (the last step)
    latest_agent_data = agent_data.xs(model.steps, level="Step")
    latest_agent_data.to_csv(f"{output_dir}/agent_data_{seed}_{timestamp}.csv")

    return model, model_data, agent_data


def plot_results(model_data, seed=None, output_dir="results", timestamp=None):
    """
    Plot the results of the simulation.

    Parameters:
    -----------
    model_data : DataFrame
        Data collected from the model
    seed : int, optional
        Random seed used (for plot title)
    output_dir : str
        Directory to save results
    timestamp : str, optional
        Timestamp to add to filenames
    """
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{output_dir}/simulation_results_{seed}_{timestamp}" if seed else f"{output_dir}/simulation_results_{timestamp}"

    # Create several plots

    # 1. Agent populations over time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    model_data[["Active Humans", "Active Bots"]].plot(ax=ax1)
    ax1.set_title(f"Active Agents Over Time (Seed: {seed})" if seed else "Active Agents Over Time")
    ax1.set_xlabel("Steps (Days)")
    ax1.set_ylabel("Number of Agents")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_base}_population.png")

    # 2. Human satisfaction over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    model_data["Average Human Satisfaction"].plot(ax=ax2, color="green")
    ax2.set_title(f"Average Human Satisfaction Over Time (Seed: {seed})" if seed else "Average Human Satisfaction Over Time")
    ax2.set_xlabel("Steps (Days)")
    ax2.set_ylabel("Satisfaction Level (0-100)")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_base}_satisfaction.png")

    # 3. Combined plot
    fig3, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Add title with seed information if provided
    fig3.suptitle(f"Social Media Simulation Results (Seed: {seed})" if seed else "Social Media Simulation Results",
                 fontsize=14)

    # Plot active agents over time
    model_data[["Active Humans", "Active Bots"]].plot(ax=axes[0])
    axes[0].set_title("Active Agents Over Time")
    axes[0].set_xlabel("Steps (Days)")
    axes[0].set_ylabel("Number of Agents")
    axes[0].grid(True)
    axes[0].legend()

    # Plot average satisfaction over time
    model_data["Average Human Satisfaction"].plot(ax=axes[1], color="green")
    axes[1].set_title("Average Human Satisfaction Over Time")
    axes[1].set_xlabel("Steps (Days)")
    axes[1].set_ylabel("Satisfaction Level")
    axes[1].set_ylim(0, 100)  # Satisfaction scale from 0 to 100
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{filename_base}_combined.png")

    # 4. Deactivated agents over time
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    model_data[["Deactivated Humans", "Deactivated Bots"]].plot(ax=ax4)
    ax4.set_title(f"Deactivated Agents Over Time (Seed: {seed})" if seed else "Deactivated Agents Over Time")
    ax4.set_xlabel("Steps (Days)")
    ax4.set_ylabel("Number of Agents")
    ax4.grid(True)
    ax4.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_base}_deactivated.png")

    # Close all figures to prevent memory leaks
    plt.close('all')


def run_parameter_sweep(
    parameter_name,
    parameter_values,
    steps=365,
    num_humans=100,
    num_bots=20,
    output_dir="parameter_sweep",
    seed_start=42,
    repetitions=1
):
    """
    Run a parameter sweep, testing different values for a specified parameter.

    Parameters:
    -----------
    parameter_name : str
        Name of the parameter to sweep (e.g., 'num_initial_bots')
    parameter_values : list
        List of values to test for the parameter
    steps : int
        Number of simulation steps to run
    num_humans : int
        Default number of initial human agents
    num_bots : int
        Default number of initial bot agents
    output_dir : str
        Directory to save results
    seed_start : int
        Starting seed for random number generation
    repetitions : int
        Number of repetitions for each parameter value
    """
    # Create output directory
    sweep_dir = f"{output_dir}/{parameter_name}_sweep"
    if not os.path.exists(sweep_dir):
        os.makedirs(sweep_dir)

    # Store results
    results_df = pd.DataFrame()

    print(f"Running parameter sweep for {parameter_name} with values: {parameter_values}")

    for rep in range(repetitions):
        for i, value in enumerate(parameter_values):
            seed = seed_start + i + rep * len(parameter_values)
            print(f"\nRunning with {parameter_name} = {value} (repetition {rep+1}/{repetitions})")

            # Create parameter dict
            params = {
                "num_initial_humans": num_humans,
                "num_initial_bots": num_bots,
                "seed": seed
            }
            # Override the swept parameter
            if parameter_name == "num_initial_humans":
                params["num_initial_humans"] = value
            elif parameter_name == "num_initial_bots":
                params["num_initial_bots"] = value

            # Run the simulation
            model = SmallWorldNetworkModel(**params)

            # Run simulation
            for step in range(steps):
                model.step()

            # Collect final metrics
            row = {
                "parameter_value": value,
                "seed": seed,
                "repetition": rep + 1,
                "final_active_humans": model.active_humans,
                "final_active_bots": model.active_bots,
                "final_deactivated_humans": model.deactivated_humans,
                "final_deactivated_bots": model.deactivated_bots,
                "final_avg_satisfaction": model.get_avg_human_satisfaction()
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    # Save results
    results_df.to_csv(f"{sweep_dir}/parameter_sweep_results.csv", index=False)

    # Plot aggregate results
    plt.figure(figsize=(10, 6))

    # Group by parameter value and compute mean and std
    grouped = results_df.groupby("parameter_value")
    means = grouped.mean().reset_index()
    stds = grouped.std().reset_index()

    # Plot with error bars
    plt.errorbar(
        means["parameter_value"],
        means["final_avg_satisfaction"],
        yerr=stds["final_avg_satisfaction"],
        fmt='-o',
        capsize=5
    )

    plt.xlabel(parameter_name)
    plt.ylabel("Average Human Satisfaction")
    plt.title(f"Effect of {parameter_name} on Final Human Satisfaction")
    plt.grid(True)
    plt.savefig(f"{sweep_dir}/parameter_sweep_plot.png")
    plt.close()

    return results_df


if __name__ == "__main__":
    # Run a single simulation with default parameters
    print("Running single simulation...")
    run_simulation(steps=365, num_humans=100, num_bots=20, seed=42)

    # Uncomment to run parameter sweep
    # print("\nRunning parameter sweep for bot ratio...")
    # run_parameter_sweep(
    #     parameter_name="num_initial_bots",
    #     parameter_values=[5, 10, 20, 30, 40, 50],
    #     steps=365,
    #     num_humans=100,
    #     seed_start=100,
    #     repetitions=3
    # )
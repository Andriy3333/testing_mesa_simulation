import solara
import mesa
import networkx as nx
import matplotlib.pyplot as plt
from agents import Human, Bot
from model import SocialMediaModel


# Helper function to define agent portrayal (this maps agents to visual representation)
def agent_portrayal(agent):
    node_color_dict = {
        'bot': "tab:red",  # Bots are red
        'human': "tab:blue",  # Humans are blue
    }

    if isinstance(agent, Bot):
        return {"color": node_color_dict['bot'], "size": 10}
    elif isinstance(agent, Human):
        return {"color": node_color_dict['human'], "size": 10}

    return {"color": "gray", "size": 5}  # Default for unknown agents


# Function to display model metrics
def get_metrics(model):
    """Displays the current model metrics in Markdown format"""
    active_humans = model.datacollector.get_agent_vars_dataframe()['state'].value_counts().get('ENGAGED', 0)
    bots = len([agent for agent in model.schedule.agents if isinstance(agent, Bot)])

    return solara.Markdown(f"""
    **Active Humans**: {active_humans}
    **Bots**: {bots}
    """)


# Define the model parameters and make the controls
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_human_count": solara.SliderInt(
        label="Initial Human Count", value=100, min=10, max=500, step=10
    ),
    "initial_bot_ratio": solara.SliderFloat(
        label="Initial Bot Ratio", value=0.1, min=0.01, max=1.0, step=0.01
    ),
    "human_growth_rate": solara.SliderFloat(
        label="Human Growth Rate", value=0.05, min=0.0, max=0.2, step=0.01
    ),
    "bot_growth_rate": solara.SliderFloat(
        label="Bot Growth Rate", value=0.1, min=0.0, max=0.2, step=0.01
    ),
    "engagement_decay_rate": solara.SliderFloat(
        label="Engagement Decay Rate", value=0.1, min=0.01, max=0.5, step=0.01
    ),
}

# Set up the space and line plot components
SpacePlot = solara.make_space_component(agent_portrayal)


def post_process_lineplot(ax):
    """Post-processing for line plots to adjust labels, etc."""
    ax.set_ylim(ymin=0)
    ax.set_ylabel("# agents")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")


StatePlot = solara.make_plot_component(
    {"Active Humans": "tab:blue", "Bots": "tab:red"},
    post_process=post_process_lineplot,
)


def social_media_simulation():
    """The main Solara app to control the model and visualization."""
    model = solara.use_state(None)
    update_trigger = solara.use_state(0)

    def update_model(reset=False):
        if reset:
            model[0] = None
        update_trigger[0] += 1

    with solara.AppBar():
        solara.AppBarTitle("Social Media Network Simulation")

    with solara.Column():
        if model[0] is None:
            # Initialize the model
            def init_model(params):
                model[0] = SocialMediaModel(
                    initial_human_count=params["initial_human_count"],
                    initial_bot_ratio=params["initial_bot_ratio"],
                    human_growth_rate=params["human_growth_rate"],
                    bot_growth_rate=params["bot_growth_rate"],
                    engagement_decay_rate=params["engagement_decay_rate"],
                )
                update_model()

            return solara.Button(
                "Initialize Model",
                on_click=lambda: init_model(model_params),
                color="primary"
            )

        # Model controls when initialized

    if model[0] is not None:
        # Visualization components
        return solara.Column([
            SpacePlot(model[0]),
            StatePlot(model[0]),
            get_metrics(model[0])
        ])
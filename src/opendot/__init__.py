"""opendot — an interactive terminal AI agent you can fully undo.

Public SDK surface (so CLI, and future clients, are thin layers over this):

    from opendot import Agent, AgentConfig

    agent = Agent(AgentConfig(model="gpt-4o"))
    async for event in agent.run("list the python files and summarize them"):
        ...
"""

from opendot.agent.config import AgentConfig
from opendot.agent.events import Event
from opendot.agent.loop import Agent

__version__ = "0.4.2"

__all__ = ["Agent", "AgentConfig", "Event", "__version__"]

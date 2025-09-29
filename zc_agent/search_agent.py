from pydantic_ai import Agent

from zc_agent.llm import read_prompt
from zc_agent.search_tools import SearchTool


def init_agent(index):
    system_prompt = read_prompt("search_agent.md")
    search_tool = SearchTool(index=index)

    agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[search_tool.search],
        model='gpt-4o-mini'
    )

    return agent
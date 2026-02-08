import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from nat.builder.builder import Builder, LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LangGraphTimeAgentConfig(AgentBaseConfig, name="langgraph_time_agent"):
    """Configuration for the LangGraph Time Agent."""
    description: str = "LangGraph Time Agent that can get current time and convert time between cities."


class LangGraphTimeAgentInput(BaseModel):
    query: str = Field(
        description="A natural language query about time, e.g. 'What is the current time in Warsaw?' "
                    "or 'When it is 15:00 in Warsaw, what time is it in Tokyo?'"
    )


def _get_timezone_for_city(city_name: str) -> str:
    """Use Open-Meteo geocoding API to get the timezone for a city."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city_name, "count": 1})
    geo_resp.raise_for_status()

    geo_data = geo_resp.json()
    if not geo_data.get("results"):
        raise ValueError(f"City not found: {city_name}")

    return geo_data["results"][0]["timezone"]


@tool
def get_current_time(city_name: str) -> str:
    """Get the current time for a given city.

    Args:
        city_name: The name of the city (e.g., 'Warsaw', 'Tokyo', 'New York').

    Returns:
        A string with the current time in the specified city.
    """
    timezone_str = _get_timezone_for_city(city_name)
    tz = ZoneInfo(timezone_str)
    now = datetime.now(tz)
    return f"The current time in {city_name} is {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (timezone: {timezone_str})"


@tool
def convert_time(source_city: str, target_city: str, time_str: str) -> str:
    """Convert a time from one city's timezone to another city's timezone.

    Args:
        source_city: The city where the given time applies (e.g., 'Warsaw').
        target_city: The city to convert the time to (e.g., 'Tokyo').
        time_str: The time in HH:MM format (e.g., '15:00').

    Returns:
        A string showing the converted time.
    """
    source_tz_str = _get_timezone_for_city(source_city)
    target_tz_str = _get_timezone_for_city(target_city)

    source_tz = ZoneInfo(source_tz_str)
    target_tz = ZoneInfo(target_tz_str)

    # Parse the time string and attach the source timezone using today's date
    time_parts = time_str.strip().split(":")
    hour, minute = int(time_parts[0]), int(time_parts[1])
    today = datetime.now(source_tz).date()
    source_dt = datetime(today.year, today.month, today.day, hour, minute, tzinfo=source_tz)

    # Convert to target timezone
    target_dt = source_dt.astimezone(target_tz)

    # Also include current time in both cities
    now_source = datetime.now(source_tz)
    now_target = datetime.now(target_tz)

    return (
        f"{source_dt.strftime('%H:%M')} in {source_city} = "
        f"{target_dt.strftime('%H:%M')} in {target_city} "
        f"(currently {now_source.strftime('%H:%M %Z')} in {source_city}, "
        f"{now_target.strftime('%H:%M %Z')} in {target_city})"
    )


@register_function(config_type=LangGraphTimeAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def langgraph_time_agent_function(config: LangGraphTimeAgentConfig, builder: Builder):
    """Register a LangGraph-based time agent."""
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = [get_current_time, convert_time]

    # Lazy init to avoid event loop conflicts with nat serve
    _graph = None

    def _get_graph():
        nonlocal _graph
        if _graph is None:
            _graph = create_react_agent(
                llm,
                tools,
                prompt="You are a time assistant. IMPORTANT: When the user asks about time conversion "
                       "between cities, you MUST also call get_current_time for EACH city mentioned "
                       "and include the current time in your final answer alongside the conversion result.",
            )
        return _graph

    async def _response_fn(query: str) -> str:
        graph = _get_graph()
        result = await graph.ainvoke({"messages": [("user", query)]})

        # Build response from tool outputs directly, bypassing LLM summarization
        tool_results = []
        cities = set()
        for msg in result["messages"]:
            if msg.type == "tool":
                tool_results.append(msg.content)
            if msg.type == "ai" and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    for key in ("city_name", "source_city", "target_city"):
                        if key in tc.get("args", {}):
                            cities.add(tc["args"][key])

        # Build deterministic answer: tool outputs + current time for all cities
        parts = list(tool_results) if tool_results else [str(result["messages"][-1].content)]
        for city in sorted(cities):
            try:
                tz_str = _get_timezone_for_city(city)
                now = datetime.now(ZoneInfo(tz_str))
                parts.append(f"Current time in {city}: {now.strftime('%H:%M %Z')}")
            except Exception:
                pass

        return ". ".join(parts)

    yield FunctionInfo.from_fn(
        _response_fn,
        input_schema=LangGraphTimeAgentInput,
        description=config.description,
    )

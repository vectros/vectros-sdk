from vectros import Agent, ask_terminal, tool


@tool
def weather(city: str) -> str:
    """Current weather for a city.

    Args:
        city: City name, e.g. "Pune".
    """
    return f"{city}: 31 C, clear sky"


@tool(approval=True)
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


agent = Agent("assistant", tools=[weather, send_email], approve=ask_terminal)
print(agent.run("What is the weather in Pune? Email it to me@example.com."))

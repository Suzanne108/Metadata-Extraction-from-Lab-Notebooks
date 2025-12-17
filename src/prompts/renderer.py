""""Prompt rendering utilities."""

from src.prompts.base import Prompt
from langchain.messages import SystemMessage

def render_prompt(prompt: Prompt, inputs: dict) -> SystemMessage:
    """Renders a prompt with the given inputs."""
    
    try:
        prompt : str = prompt.template.format(**inputs)
        return SystemMessage(content=prompt)
    except KeyError as e:
        raise ValueError(f"Missing input variable for prompt rendering: {e}") from e
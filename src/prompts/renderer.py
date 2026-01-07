""""Prompt rendering utilities."""

from src.prompts.base import Prompt

def render_prompt(prompt: Prompt, inputs: dict) -> str:
    """Renders a prompt with the given inputs."""
    
    try:
        prompt : str = prompt.template.format(**inputs)
        return prompt
    except KeyError as e:
        raise ValueError(f"Missing input variable for prompt rendering: {e}") from e
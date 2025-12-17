""""Prompt registry for managing prompts."""

from pathlib import Path
from typing import Dict, Tuple

import yaml

from src.prompts.base import Prompt

class PromptRegistry:
    """Registry for managing prompts."""
 
    _prompts: Dict[Tuple[str, str], Prompt] = {}

    def __init__(self):
        path = Path.cwd() / "src" / "prompts"
        self._load_from_directory(path)

    @classmethod
    def _load_from_directory(cls, root: Path) -> None:
        """Loads all prompts from the specified directory."""

        for path in root.rglob("*.yaml"):
            prompt = cls._load_prompt(path)
            cls._register(prompt)

    @classmethod
    def _register(cls, prompt: Prompt) -> None:
        """Registers a new prompt."""
        key = (prompt.name, prompt.version)
        cls._prompts[key] = prompt

    @classmethod
    def get_prompt(cls, name: str, version: str) -> Prompt:
        """Retrieves a prompt by name and version."""
        key = (name, version)
        if key not in cls._prompts:
            raise KeyError(f"Prompt '{name}' with version '{version}' not found.")
        return cls._prompts[key] 

    @staticmethod
    def _load_prompt(path: Path) -> Prompt:
        """Loads a prompt from a YAML file."""
   
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return Prompt(
            name=data["name"],
            version=data["version"],
            template=data["template"],
            input_variables=data["input_variables"] 
        )
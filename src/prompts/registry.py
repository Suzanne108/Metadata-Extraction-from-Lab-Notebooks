"""Prompt registry for managing prompts."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Tuple

import yaml

from src.prompts.base import Prompt


class PromptRegistry:
    """Registry for managing prompts with a shared singleton instance."""

    _instance: PromptRegistry | None = None
    _instance_lock: Lock = Lock()
    _prompts: Dict[Tuple[str, str], Prompt] = {}

    def __init__(self, force_reload: bool = False):
        # Prevent repeated directory scans when callers construct PromptRegistry() often.
        if self._prompts and not force_reload:
            return

        path = Path.cwd() / "src" / "prompts"
        self._load_from_directory(path)

    @classmethod
    def get_instance(cls, force_reload: bool = False) -> PromptRegistry:
        """Return a shared PromptRegistry instance across the process."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(force_reload=force_reload)
                    return cls._instance

        if force_reload:
            with cls._instance_lock:
                cls._prompts.clear()
                cls._instance = cls(force_reload=True)

        return cls._instance

    @classmethod
    def _register(cls, prompt: Prompt) -> None:
        """Register a new prompt."""
        key = (prompt.name, prompt.version)
        cls._prompts[key] = prompt

    @classmethod
    def get_prompt(cls, name: str, version: str) -> Prompt:
        """Retrieve a prompt by name and version."""
        key = (name, version)
        if key not in cls._prompts:
            raise KeyError(f"Prompt '{name}' with version '{version}' not found.")
        return cls._prompts[key]
    
    @classmethod
    def _load_from_directory(cls, root: Path) -> None:
        """Load all prompts from the specified directory."""
        for path in root.rglob("*.yaml"):
            try:
                prompt = cls._load_prompt(path)
                cls._register(prompt)
            except Exception as e:
                # Provide clear feedback about which file failed to load.
                print(f"[PromptRegistry] Skipping invalid YAML at {path}: {e}")

    @staticmethod
    def _load_prompt(path: Path) -> Prompt:
        """Load a prompt from a YAML file."""
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError("YAML content is empty or not a mapping")

        required_keys = {"name", "version", "template", "input_variables"}
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")

        return Prompt(
            name=data["name"],
            version=data["version"],
            template=data["template"],
            input_variables=data["input_variables"],
        )
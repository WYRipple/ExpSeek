from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: Dict = {}

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Main method to execute the tool."""
        pass


class ToolManager:
    """
    Singleton tool manager responsible for registering and calling tools.
    Tools are registered via the @register_tool decorator.
    """

    _instance = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_tool(self, name: str, tool_class: type, allow_overwrite: bool = True, **init_kwargs):
        """Register a tool class under the given name."""
        if not allow_overwrite and name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")

        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool class must inherit from BaseTool.")

        # Merge global default kwargs with per-tool kwargs
        if hasattr(self, "_default_init_kwargs"):
            merged_kwargs = {**self._default_init_kwargs, **init_kwargs}
        else:
            merged_kwargs = init_kwargs

        self._tools[name] = tool_class(**merged_kwargs)
        return tool_class

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def call_tool(self, name: str, params: Dict[str, Any]) -> str:
        """Call a registered tool with the given parameters."""
        tool = self.get_tool(name)
        if not tool:
            return f"Tool '{name}' not found."

        try:
            return tool.call(params)
        except Exception as e:
            return f"Error calling tool '{name}': {str(e)}"

    def list_tools(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())


# Global singleton tool manager instance
tool_manager = ToolManager()


def register_tool(name: str, allow_overwrite: bool = True, **init_kwargs):
    """
    Decorator for registering a tool class with the global tool manager.

    Usage:
        @register_tool("search")
        class Search(BaseTool):
            ...
    """
    def decorator(cls):
        tool_manager.register_tool(name, cls, allow_overwrite, **init_kwargs)
        return cls
    return decorator

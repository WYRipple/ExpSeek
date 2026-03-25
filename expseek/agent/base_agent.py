from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from expseek.tools.tool_manager import ToolManager


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Handles tool registration and provides a unified tool-calling interface.
    Subclasses must implement the _run() method.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        self.function_list = function_list or []
        self.description = description
        self.files = files or []
        self.kwargs = kwargs

        self.tool_manager = ToolManager()
        self._register_tools()

    def _register_tools(self):
        """
        Validate that all required tools are already registered in the tool manager.
        Tools are registered via the @register_tool decorator when their modules are imported.
        """
        for tool_name in self.function_list:
            if isinstance(tool_name, str):
                tool = self.tool_manager.get_tool(tool_name)
                if not tool:
                    raise ValueError(
                        f"Tool '{tool_name}' not found. "
                        f"Please ensure it is registered before creating the agent."
                    )

    def _call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Call a registered tool by name with the given arguments."""
        return self.tool_manager.call_tool(tool_name, tool_args)

    @abstractmethod
    def _run(self, *args, **kwargs):
        """Main agent loop. Must be implemented by subclasses."""
        pass

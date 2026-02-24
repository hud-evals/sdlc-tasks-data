from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from mcp.types import ContentBlock


class ResponseTool(BaseTool):
    """
    Protocol for handling responses within environments.

    This abstract tool defines the interface for response handling in environments.
    Subclasses should implement the __call__ method to handle responses according
    to their specific needs.

    Example:
        class MyEnvironmentResponseTool(ResponseTool):
            async def __call__(
                self,
                response: str | None = None,
                messages: list[ContentBlock] | None = None
            ) -> list[ContentBlock]:
                # Custom implementation for handling responses
                from mcp.types import TextContent
                blocks = []
                if response:
                    # Process response according to environment needs
                    blocks.append(TextContent(text=f"[ENV] {response}", type="text"))
                if messages:
                    # Process messages according to environment needs
                    blocks.extend(messages)
                return blocks
    """

    name: str = "response"
    title: str = "Response Tool"
    description: str = "Send a text response or list of messages to the environment"

    def __init__(
        self, name: str | None = None, title: str | None = None, description: str | None = None
    ) -> None:
        super().__init__(
            name=name or self.name,
            title=title or self.title,
            description=description or self.description,
        )

    @abstractmethod
    async def __call__(
        self, response: str | None = None, messages: list[ContentBlock] | None = None
    ) -> list[ContentBlock]:
        """Handle response or messages and return as ContentBlocks.

        Args:
            response: A single text response to handle
            messages: A list of ContentBlock messages to handle

        Returns:
            List of ContentBlock containing the processed response(s)
        """
        raise NotImplementedError("Subclasses must implement __call__")

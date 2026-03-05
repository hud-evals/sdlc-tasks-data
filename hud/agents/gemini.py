"""Gemini MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import mcp.types as types
from google import genai
from google.genai import types as genai_types

from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, AgentType, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.types import with_signature

from .base import MCPAgent
from .types import GeminiConfig, GeminiCreateParams

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpec

logger = logging.getLogger(__name__)

PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]

GEMINI_CUA_INSTRUCTIONS = """
You are an autonomous computer-using agent. Follow these guidelines:

1. NEVER ask for confirmation. Complete all tasks autonomously.
2. Do NOT send messages like "I need to confirm before..." or "Do you want me to
   continue?" - just proceed.
3. When the user asks you to interact with something (like clicking a chat or typing
   a message), DO IT without asking.
4. Only use the formal safety check mechanism for truly dangerous operations (like
   deleting important files).
5. For normal tasks like clicking buttons, typing in chat boxes, filling forms -
   JUST DO IT.
6. The user has already given you permission by running this agent. No further
   confirmation is needed.
7. Be decisive and action-oriented. Complete the requested task fully.

Remember: You are expected to complete tasks autonomously. The user trusts you to do
what they asked.
""".strip()


class GeminiAgent(MCPAgent):
    """
    Gemini agent that uses MCP servers for tool execution.

    This agent uses Gemini's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.GEMINI_COMPUTER_WIDTH,
        "display_height": computer_settings.GEMINI_COMPUTER_HEIGHT,
    }
    config_cls: ClassVar[type[BaseAgentConfig]] = GeminiConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for Gemini."""
        return AgentType.GEMINI

    # Legacy tool name patterns for backwards compatibility
    _LEGACY_COMPUTER_NAMES = ("gemini_computer", "computer_gemini", "computer")

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect Gemini native tools by name for backwards compatibility.

        Supports old environments that expose tools like 'gemini_computer'
        without native_tools metadata.

        Each tuple is ordered by preference — first name that exists wins.
        Only returns a spec if this tool IS that preferred match.
        """
        from hud.tools.native_types import NativeToolSpec

        available = {t.name for t in (self._available_tools or [])} | {tool.name}
        preferred = lambda names: next((n for n in names if n in available), None) == tool.name

        if preferred(self._LEGACY_COMPUTER_NAMES):
            logger.debug("Legacy fallback: detected %s as computer tool", tool.name)
            return NativeToolSpec(
                api_type="computer_use",
                api_name="gemini_computer",
                role="computer",
            )

        return None

    @with_signature(GeminiCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GeminiAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GeminiCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: GeminiConfig

        model_client = self.config.model_client
        if model_client is None:
            if settings.api_key:
                from hud.agents.gateway import build_gateway_client

                model_client = build_gateway_client("gemini")
            elif settings.gemini_api_key:
                model_client = genai.Client(api_key=settings.gemini_api_key)
            else:
                raise ValueError(
                    "No API key found. Set HUD_API_KEY for HUD gateway, "
                    "or GEMINI_API_KEY for direct Gemini access."
                )

        if self.config.validate_api_key:
            try:
                list(
                    model_client.models.list(
                        config=genai_types.ListModelsConfig(page_size=1)
                    )
                )
            except Exception as e:
                raise ValueError(f"Gemini API key is invalid: {e}") from e

        self.gemini_client: genai.Client = model_client
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.max_output_tokens = self.config.max_output_tokens
        self.hud_console = HUDConsole(logger=logger)

        self._computer_tool_name = "gemini_computer"
        self.excluded_predefined_functions = list(self.config.excluded_predefined_functions)

        self.max_recent_turn_with_screenshots = (
            computer_settings.GEMINI_MAX_RECENT_TURN_WITH_SCREENSHOTS
        )

        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{GEMINI_CUA_INSTRUCTIONS}"
        else:
            self.system_prompt = GEMINI_CUA_INSTRUCTIONS

        # Track mapping from Gemini tool names to MCP tool names
        self._gemini_to_mcp_tool_map: dict[str, str] = {}
        self.gemini_tools: genai_types.ToolListUnion = []

    def _on_tools_ready(self) -> None:
        """Build Gemini-specific tool mappings after tools are discovered."""
        self._convert_tools_for_gemini()

    async def get_system_messages(self) -> list[genai_types.Content]:
        """No system messages for Gemini because applied in get_response"""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[genai_types.Content]:
        """Format messages for Gemini."""
        gemini_parts: list[genai_types.Part] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                gemini_parts.append(genai_types.Part(text=block.text))
            elif isinstance(block, types.ImageContent):
                import base64

                image_bytes = base64.b64decode(block.data)
                gemini_parts.append(
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=block.mimeType)
                )
            else:
                self.hud_console.log(f"Unknown content block type: {type(block)}", level="warning")

        return [genai_types.Content(role="user", parts=gemini_parts)]

    async def get_response(self, messages: list[genai_types.Content]) -> AgentResponse:
        """Get response from Gemini including any tool calls."""
        self._remove_old_screenshots(messages)

        generate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            tools=self.gemini_tools,
            system_instruction=self.system_prompt,
        )

        response = await self.gemini_client.aio.models.generate_content(
            model=self.config.model,
            contents=cast("Any", messages),
            config=generate_config,
        )

        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            messages.append(response.candidates[0].content)

        result = AgentResponse(content="", tool_calls=[], done=True)
        collected_tool_calls: list[MCPToolCall] = []

        if not response.candidates:
            self.hud_console.warning("Response has no candidates")
            return result

        candidate = response.candidates[0]

        text_content = ""
        thinking_content = ""

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.function_call:
                    func_name = part.function_call.name or ""
                    raw_args = dict(part.function_call.args) if part.function_call.args else {}

                    if func_name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
                        normalized_args: dict[str, Any] = {"action": func_name}

                        coord = raw_args.get("coordinate") or raw_args.get("coordinates")
                        if isinstance(coord, list | tuple) and len(coord) >= 2:
                            try:
                                normalized_args["x"] = int(coord[0])
                                normalized_args["y"] = int(coord[1])
                            except (TypeError, ValueError):
                                pass

                        dest = (
                            raw_args.get("destination")
                            or raw_args.get("destination_coordinate")
                            or raw_args.get("destinationCoordinate")
                        )
                        if isinstance(dest, list | tuple) and len(dest) >= 2:
                            try:
                                normalized_args["destination_x"] = int(dest[0])
                                normalized_args["destination_y"] = int(dest[1])
                            except (TypeError, ValueError):
                                pass

                        for key in (
                            "text", "press_enter", "clear_before_typing",
                            "safety_decision", "direction", "magnitude",
                            "url", "keys", "x", "y",
                            "destination_x", "destination_y",
                        ):
                            if key in raw_args:
                                normalized_args[key] = raw_args[key]

                        tool_call = MCPToolCall(
                            name=self._computer_tool_name,
                            arguments=normalized_args,
                            gemini_name=func_name,  # type: ignore[arg-type]
                        )
                        collected_tool_calls.append(tool_call)
                    else:
                        mcp_tool_name = self._gemini_to_mcp_tool_map.get(func_name, func_name)
                        tool_call = MCPToolCall(
                            name=mcp_tool_name,
                            arguments=raw_args,
                        )
                        collected_tool_calls.append(tool_call)
                elif part.thought is True and part.text:
                    if thinking_content:
                        thinking_content += "\n"
                    thinking_content += part.text
                elif part.text:
                    text_content += part.text

        if collected_tool_calls:
            result.tool_calls = collected_tool_calls
            result.done = False

        result.content = text_content
        if thinking_content:
            result.reasoning = thinking_content

        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[genai_types.Content]:
        """Format tool results into Gemini messages."""
        function_responses = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            gemini_name = getattr(tool_call, "gemini_name", tool_call.name)
            is_computer_call = tool_call.name == self._computer_tool_name

            response_dict: dict[str, Any] = {}
            url = None

            if result.isError:
                error_msg = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        if content.text.startswith("__URL__:"):
                            url = content.text.replace("__URL__:", "")
                        else:
                            error_msg = content.text
                            break
                response_dict["error"] = error_msg
                response_dict["url"] = url if url else "about:blank"
            else:
                response_dict["success"] = True

            screenshot_parts = []
            if is_computer_call:
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        if content.text.startswith("__URL__:"):
                            url = content.text.replace("__URL__:", "")
                    elif isinstance(content, types.ImageContent):
                        import base64

                        image_bytes = base64.b64decode(content.data)
                        screenshot_parts.append(
                            genai_types.FunctionResponsePart(
                                inline_data=genai_types.FunctionResponseBlob(
                                    mime_type=content.mimeType or "image/png",
                                    data=image_bytes,
                                )
                            )
                        )

                response_dict["url"] = url if url else "about:blank"

                requires_ack = False
                if tool_call.arguments:
                    requires_ack = bool(tool_call.arguments.get("safety_decision"))
                if requires_ack:
                    response_dict["safety_acknowledgement"] = True
            else:
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        response_dict["output"] = content.text
                        break

            function_response = genai_types.FunctionResponse(
                name=gemini_name,
                response=response_dict,
                parts=screenshot_parts if screenshot_parts else None,
            )
            function_responses.append(function_response)

        return [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(function_response=fr) for fr in function_responses],
            )
        ]

    async def create_user_message(self, text: str) -> genai_types.Content:
        """Create a user message in Gemini's format."""
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def _convert_tools_for_gemini(self) -> None:
        """Convert MCP tools to Gemini tool format using native specs.

        Uses shared categorize_tools() for role-based exclusion.
        """
        self._gemini_to_mcp_tool_map = {}
        self.gemini_tools = []

        categorized = self._categorized_tools

        for tool, spec in categorized.hosted:
            gemini_tool = self._build_hosted_tool(spec)
            if gemini_tool:
                self.gemini_tools.append(gemini_tool)
                logger.debug("Added hosted tool %s (%s) for Gemini", tool.name, spec.api_type)

        for tool, spec in categorized.native:
            gemini_tool = self._build_native_tool(tool, spec)
            if gemini_tool:
                self._gemini_to_mcp_tool_map[tool.name] = tool.name
                self.gemini_tools.append(gemini_tool)

        for tool in categorized.generic:
            if tool.description is None or tool.inputSchema is None:
                raise ValueError(
                    f"MCP tool {tool.name} requires both a description and inputSchema."
                )
            function_decl = genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.inputSchema,
            )
            gemini_tool = genai_types.Tool(function_declarations=[function_decl])
            self._gemini_to_mcp_tool_map[tool.name] = tool.name
            self.gemini_tools.append(gemini_tool)

        tool_names = sorted(self._gemini_to_mcp_tool_map.keys())
        self.console.info(
            f"Agent initialized with {len(tool_names)} tools: {', '.join(tool_names)}"
        )

    def _build_hosted_tool(self, spec: NativeToolSpec) -> genai_types.Tool | None:
        """Build a Gemini hosted tool from a NativeToolSpec."""
        match spec.api_type:
            case "google_search":
                return genai_types.Tool(google_search=genai_types.GoogleSearch(**spec.extra))
            case "code_execution":
                return genai_types.Tool(code_execution=genai_types.ToolCodeExecution())
            case "url_context":
                return genai_types.Tool(url_context=genai_types.UrlContext())
            case _:
                logger.warning("Unknown hosted tool type: %s", spec.api_type)
                return None

    def _build_native_tool(self, tool: types.Tool, spec: NativeToolSpec) -> genai_types.Tool | None:
        """Build a Gemini native tool from a NativeToolSpec."""
        match spec.api_type:
            case "computer_use":
                logger.debug("Using native ComputerUse for tool %s", tool.name)
                self._computer_tool_name = tool.name
                return genai_types.Tool(
                    computer_use=genai_types.ComputerUse(
                        environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=self.excluded_predefined_functions,
                    )
                )
            case _:
                logger.debug(
                    "Native tool type %s for %s, using function declaration",
                    spec.api_type,
                    tool.name,
                )
                if tool.description is None or tool.inputSchema is None:
                    raise ValueError(
                        f"MCP tool {tool.name} requires both a description and inputSchema."
                    )
                function_decl = genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.inputSchema,
                )
                return genai_types.Tool(function_declarations=[function_decl])

    def _remove_old_screenshots(self, messages: list[genai_types.Content]) -> None:
        """Remove screenshots from old turns to manage context length."""
        turn_with_screenshots_found = 0

        for content in reversed(messages):
            if content.role == "user" and content.parts:
                has_screenshot = False
                for part in content.parts:
                    if (
                        part.function_response
                        and part.function_response.parts
                        and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                    ):
                        has_screenshot = True
                        break

                if has_screenshot:
                    turn_with_screenshots_found += 1
                    if turn_with_screenshots_found > self.max_recent_turn_with_screenshots:
                        for part in content.parts:
                            if (
                                part.function_response
                                and part.function_response.parts
                                and part.function_response.name
                                in PREDEFINED_COMPUTER_USE_FUNCTIONS
                            ):
                                part.function_response.parts = None

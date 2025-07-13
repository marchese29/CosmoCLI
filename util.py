import json
from typing import Awaitable, Callable, Union

from bedrock.model import InputSchema, ToolSpec, ToolSpecPayload

JSONPrimitive = Union[int, str, float, None]
JSONObject = dict[str, Union[JSONPrimitive, 'JSONValue']]
JSONList = list['JSONValue']
JSONValue = Union[JSONPrimitive, JSONObject, JSONList]

PromptTool = Callable[[str], Awaitable[str]]


class SonicTools:
    """Registry for tools"""

    def __init__(self):
        self._agents: dict[str, tuple[str | None, PromptTool]] = {}

    def agent_tool(
        self, name: str, description: str | None = None
    ) -> Callable[[PromptTool], PromptTool]:
        def wrapper(f: PromptTool) -> PromptTool:
            self._agents[name] = (description, f)
            return f

        return wrapper

    async def invoke_agent(self, name: str, prompt: str) -> str:
        if name not in self._agents:
            raise KeyError(f'No agent tool named "{name}" was registered')
        return await self._agents[name][1](prompt)

    def specs(self) -> list[ToolSpec]:
        result: list[ToolSpec] = []
        for name, (description, _) in self._agents.items():
            result.append(
                ToolSpec(
                    tool_spec=ToolSpecPayload(
                        name=name,
                        description=description or name,
                        input_schema=InputSchema(
                            json=json.dumps(
                                {
                                    'type': 'object',
                                    'properties': {
                                        'prompt': {
                                            'type': 'string',
                                            'description': (
                                                'The instruction being sent to the home '
                                                'controller agent'
                                            ),
                                        }
                                    },
                                    'required': ['prompt'],
                                }
                            )
                        ),
                    )
                )
            )
        return result

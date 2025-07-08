from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

####################
# BEDROCK REQUESTS #
####################


class BedrockEventModel(BaseModel):
    """Base model for all Bedrock events with camelCase serialization"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# Configuration Models
class InferenceConfiguration(BedrockEventModel):
    max_tokens: int = 1024
    top_p: float = 0.9
    temperature: float = 0.7


class AudioInputConfiguration(BedrockEventModel):
    media_type: Literal['audio/lpcm'] = 'audio/lpcm'
    sample_rate_hertz: int = 16000
    sample_size_bits: int = 16
    channel_count: int = 1
    audio_type: Literal['SPEECH'] = 'SPEECH'
    encoding: Literal['base64'] = 'base64'


class TextInputConfiguration(BedrockEventModel):
    media_type: Literal['text/plain'] = 'text/plain'


class ToolResultInputConfiguration(BedrockEventModel):
    tool_use_id: str
    type: Literal['TEXT'] = 'TEXT'
    text_input_configuration: TextInputConfiguration


class AudioOutputConfiguration(BedrockEventModel):
    media_type: Literal['audio/lpcm'] = 'audio/lpcm'
    sample_rate_hertz: int = 24000
    sample_size_bits: int = 16
    channel_count: int = 1
    voice_id: str = 'matthew'  # Available: matthew, tiffany, amy
    encoding: Literal['base64'] = 'base64'
    audio_type: Literal['SPEECH'] = 'SPEECH'


class TextOutputConfiguration(BedrockEventModel):
    media_type: Literal['text/plain'] = 'text/plain'


class ToolUseOutputConfiguration(BedrockEventModel):
    media_type: Literal['application/json'] = 'application/json'


# Tool Configuration Models
class InputSchema(BedrockEventModel):
    json_schema: dict[str, Any] = Field(alias='json')


class ToolSpec(BedrockEventModel):
    name: str
    description: str
    input_schema: InputSchema


class ToolConfiguration(BedrockEventModel):
    tools: list[ToolSpec]


# Event Models
class SessionStartPayload(BedrockEventModel):
    inference_configuration: InferenceConfiguration


class SessionStart(BedrockEventModel):
    session_start: SessionStartPayload


class SessionEndPayload(BedrockEventModel):
    pass


class SessionEnd(BedrockEventModel):
    session_end: SessionEndPayload


class ContentStartPayload(BedrockEventModel):
    prompt_name: str
    content_name: str
    type: Literal['AUDIO', 'TEXT', 'TOOL']
    interactive: bool
    role: Literal['USER', 'ASSISTANT', 'SYSTEM', 'TOOL']
    audio_input_configuration: AudioInputConfiguration | None = None
    text_input_configuration: TextInputConfiguration | None = None
    tool_result_input_configuration: ToolResultInputConfiguration | None = None


class ContentStart(BedrockEventModel):
    content_start: ContentStartPayload


class ContentEndPayload(BedrockEventModel):
    prompt_name: str
    content_name: str


class ContentEnd(BedrockEventModel):
    content_end: ContentEndPayload


class AudioInputPayload(BedrockEventModel):
    prompt_name: str
    content_name: str
    content: str  # base64 encoded audio


class AudioInput(BedrockEventModel):
    audio_input: AudioInputPayload


class TextInputPayload(BedrockEventModel):
    prompt_name: str
    content_name: str
    content: str


class TextInput(BedrockEventModel):
    text_input: TextInputPayload


class PromptStartPayload(BedrockEventModel):
    prompt_name: str
    text_output_configuration: TextOutputConfiguration
    audio_output_configuration: AudioOutputConfiguration
    tool_use_output_configuration: ToolUseOutputConfiguration
    tool_configuration: ToolConfiguration


class PromptStart(BedrockEventModel):
    prompt_start: PromptStartPayload


class PromptEndPayload(BedrockEventModel):
    prompt_name: str


class PromptEnd(BedrockEventModel):
    prompt_end: PromptEndPayload


class ToolResult(BedrockEventModel):
    prompt_name: str
    content_name: str
    content: str  # JSON string of the tool result


###############################
# BIDIRECTIONAL STREAM EVENTS #
###############################


class StreamEventModel(BaseModel):
    """Base model for all streaming events."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class CompletionEvent(StreamEventModel):
    session_id: str
    prompt_name: str
    completion_id: str


class CompletionStartEvent(CompletionEvent):
    """Completion start event."""


class ContentStartEvent(CompletionEvent):
    """Content start event."""

    additional_model_fields: (
        Literal['{"generationStage":"FINAL"}', '{"generationStage":"SPECULATIVE"}'] | None
    ) = None
    content_id: str
    type: Literal['TEXT', 'TOOL', 'AUDIO']
    role: Literal['USER', 'ASSISTANT', 'TOOL']
    text_output_configuration: TextOutputConfiguration | None = None
    tool_use_output_configuration: ToolUseOutputConfiguration | None = None
    audio_output_configuration: AudioOutputConfiguration | None = None


class UsageEvent(CompletionEvent):
    """Usage event"""

    details: dict[str, Any] | None = None
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int


class TextOutputEvent(CompletionEvent):
    """Text output event."""

    content_id: str
    content: str


class ToolUseEvent(CompletionEvent):
    """Tool use event."""

    content_id: str
    content: dict[str, Any]
    tool_name: str
    tool_use_id: str


class AudioOutputEvent(CompletionEvent):
    """Audio output event."""

    content_id: str
    content: str


class ContentEndEvent(CompletionEvent):
    """Content end event."""

    content_id: str
    stop_reason: Literal['PARTIAL_TURN', 'END_TURN', 'INTERRUPTED', 'TOOL_USE']
    type: Literal['TEXT', 'TOOL', 'AUDIO']


class CompletionEndEvent(CompletionEvent):
    """Completion end event."""

    stop_reason: Literal['END_TURN']


class StreamEventHandler:
    """Visitor for stream events"""

    async def on_completion_start(self, event: CompletionStartEvent):
        """Invoked for completion start"""
        pass

    async def on_usage(self, event: UsageEvent):
        """Invoked for usage"""
        pass

    async def on_content_start(self, event: ContentStartEvent):
        """Invoked for content start"""
        pass

    async def on_text_output(self, event: TextOutputEvent):
        """Invoked for text output"""
        pass

    async def on_tool_use(self, event: ToolUseEvent):
        """Invoked for tool use"""
        pass

    async def on_audio_output(self, event: AudioOutputEvent):
        """Invoked for audio output"""
        pass

    async def on_content_end(self, event: ContentEndEvent):
        """Invoked for content end"""
        pass

    async def on_completion_end(self, event: CompletionEndEvent):
        """Invoked for completion end"""
        pass


class BidirectionalStreamEvent(StreamEventModel):
    """An event from the Bedrock bidirectional stream."""

    completion_start: CompletionStartEvent | None = None
    usage_event: UsageEvent | None = None
    content_start: ContentStartEvent | None = None
    text_output: TextOutputEvent | None = None
    tool_use: ToolUseEvent | None = None
    audio_output: AudioOutputEvent | None = None
    content_end: ContentEndEvent | None = None
    completion_end: CompletionEndEvent | None = None

    async def visit(self, handler: StreamEventHandler):
        """Visitor pattern that invokes the correct handler for the type of event we
        encountered."""
        if self.completion_start is not None:
            await handler.on_completion_start(self.completion_start)
        if self.usage_event is not None:
            await handler.on_usage(self.usage_event)
        elif self.content_start is not None:
            await handler.on_content_start(self.content_start)
        elif self.text_output is not None:
            await handler.on_text_output(self.text_output)
        elif self.tool_use is not None:
            await handler.on_tool_use(self.tool_use)
        elif self.audio_output is not None:
            await handler.on_audio_output(self.audio_output)
        elif self.content_end is not None:
            await handler.on_content_end(self.content_end)
        elif self.completion_end is not None:
            await handler.on_completion_end(self.completion_end)
        else:
            print('Unknown event received')

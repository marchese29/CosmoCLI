import asyncio
import base64
import json
import logging
import os
import traceback
import uuid
from typing import Optional, Self, override

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    DuplexEventStream,
    InvokeModelWithBidirectionalStreamInput,
    InvokeModelWithBidirectionalStreamOperationInput,
    InvokeModelWithBidirectionalStreamOperationOutput,
    InvokeModelWithBidirectionalStreamOutput,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOutputChunk,
)
from pydantic import BaseModel
from smithy_aws_core.credentials_resolvers import EnvironmentCredentialsResolver

from audio import AudioInterface
from bedrock.model import (
    AudioInput,
    AudioInputConfiguration,
    AudioInputPayload,
    AudioOutputConfiguration,
    AudioOutputEvent,
    BidirectionalStreamEvent,
    ContentEnd,
    ContentEndPayload,
    ContentStart,
    ContentStartPayload,
    InferenceConfiguration,
    PromptEnd,
    PromptEndPayload,
    PromptStart,
    PromptStartPayload,
    SessionEnd,
    SessionEndPayload,
    SessionStart,
    SessionStartPayload,
    StreamEventHandler,
    TextInput,
    TextInputConfiguration,
    TextInputPayload,
    TextOutputConfiguration,
    TextOutputEvent,
    ToolConfiguration,
    ToolUseEvent,
    ToolUseOutputConfiguration,
)

logger = logging.getLogger(__name__)


class BedrockStreamManager(StreamEventHandler):
    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt
        # Debug logging for voice configuration
        raw_voice_env = os.getenv('NOVA_SONIC_VOICE')
        self._voice_name = os.getenv('NOVA_SONIC_VOICE', 'matthew')
        logger.info(
            f'🎵 Voice Configuration - Raw env var: "{raw_voice_env}", '
            f'Final voice_name: "{self._voice_name}"'
        )
        self._br_client = BedrockRuntimeClient(
            config=Config(
                region=os.getenv('AWS_REGION'),
                aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
                http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
                http_auth_schemes={'aws.auth#sigv4': SigV4AuthScheme()},
            )
        )
        self._br_stream: Optional[
            DuplexEventStream[
                InvokeModelWithBidirectionalStreamInput,
                InvokeModelWithBidirectionalStreamOutput,
                InvokeModelWithBidirectionalStreamOperationOutput,
            ]
        ] = None
        self._model_id = os.getenv('BEDROCK_MODEL_ID')

        # Initialized Tasks
        self._br_incoming_task: Optional[asyncio.Task[None]] = None
        self._user_incoming_task: Optional[asyncio.Task[None]] = None
        self._audio: Optional[AudioInterface] = None

        # Audio Interfaces
        self._audio_in_q: asyncio.Queue[bytes] = asyncio.Queue()

        # Session Info
        self._prompt_name = str(uuid.uuid4())
        self._content_name = str(uuid.uuid4())
        self._audio_content_name = str(uuid.uuid4())

    async def _send_event(self, event_model: BaseModel):
        logger.debug('Sending event of type %s', event_model.__class__.__name__)
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(
                bytes_=json.dumps(
                    {'event': event_model.model_dump(by_alias=True)}
                ).encode('utf-8')
            )
        )
        try:
            assert self._br_stream is not None
            await self._br_stream.input_stream.send(event)
        except Exception as e:
            traceback.print_exc()
            raise e

    async def _process_br_incoming(self):
        assert self._br_stream is not None
        try:
            while True:
                output = await self._br_stream.await_output()
                result = await output[1].receive()
                if result is None:
                    continue

                # Switch/case based on the type of result
                match result:
                    case None:
                        # No result, continue to the next stream item
                        continue
                    case InvokeModelWithBidirectionalStreamOutputChunk(part):
                        # We have a result, let's process it
                        if part.bytes_ is None:
                            continue
                        json_data = json.loads(part.bytes_.decode('utf-8'))
                        if 'event' not in json_data:
                            continue
                        event = BidirectionalStreamEvent(**json_data['event'])
                        await event.visit(self)
                    case _:
                        # Handle all exception and unknown output types
                        pass
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception:
            # TODO: Handle other exceptions appropriately
            traceback.print_exc()

    async def _handle_audio_in(self, data: bytes):
        # Only process audio input when we're not currently playing output
        await self._audio_in_q.put(data)

    async def _process_user_incoming(self):
        # Initialize audio content
        logger.info(
            f'🎤 Starting audio content stream - Content ID: {self._audio_content_name}'
        )
        await self._send_event(
            ContentStart(
                content_start=ContentStartPayload(
                    prompt_name=self._prompt_name,
                    content_name=self._audio_content_name,
                    type='AUDIO',
                    interactive=True,
                    role='USER',
                    audio_input_configuration=AudioInputConfiguration(),
                )
            )
        )
        try:
            audio_chunk_count = 0
            while True:
                data = await self._audio_in_q.get()
                blob = base64.b64encode(data)
                audio_chunk_count += 1
                logger.debug(
                    f'📤 Sending audio chunk #{audio_chunk_count} '
                    f'({len(data)} bytes) to Bedrock'
                )
                await self._send_event(
                    AudioInput(
                        audio_input=AudioInputPayload(
                            prompt_name=self._prompt_name,
                            content_name=self._audio_content_name,
                            content=blob.decode('utf-8'),
                        )
                    )
                )
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            logger.info('🛑 Audio input processing cancelled')
            pass
        except Exception:
            # TODO: Handle other exceptions appropriately
            logger.error('❌ Error in audio input processing')
            traceback.print_exc()

    @override
    async def on_completion_start(self, event):
        logger.info(
            f'🚀 COMPLETION START - Session: {event.session_id}, '
            f'Prompt: {event.prompt_name}, Completion: {event.completion_id}'
        )

    @override
    async def on_content_start(self, event):
        speculative_info = ''
        if hasattr(event, 'additional_model_fields') and event.additional_model_fields:
            speculative_info = f' [{event.additional_model_fields}]'
        logger.info(
            f'📝 CONTENT START - Type: {event.type}, Role: {event.role}, '
            f'Content ID: {event.content_id}{speculative_info}'
        )

    @override
    async def on_text_output(self, event: TextOutputEvent):
        logger.info(f"💬 TEXT OUTPUT: '{event.content}'")

    @override
    async def on_audio_output(self, event: AudioOutputEvent):
        audio_bytes = base64.b64decode(event.content)
        logger.info(
            f'🔊 AUDIO OUTPUT: {len(audio_bytes)} bytes, Content ID: {event.content_id}'
        )
        assert self._audio is not None
        await self._audio.speak(audio_bytes)

    @override
    async def on_content_end(self, event):
        logger.info(
            f'🏁 CONTENT END - Type: {event.type}, Stop Reason: {event.stop_reason}, '
            f'Content ID: {event.content_id}'
        )

    @override
    async def on_completion_end(self, event):
        logger.info(
            f'✅ COMPLETION END - Stop Reason: {event.stop_reason}, '
            f'Completion ID: {event.completion_id}'
        )

    @override
    async def on_usage(self, event):
        logger.info(
            f'📊 USAGE - Input: {event.total_input_tokens}, '
            f'Output: {event.total_output_tokens}, Total: {event.total_tokens}'
        )

    @override
    async def on_tool_use(self, event: ToolUseEvent):
        logger.info(f'🔧 TOOL USE - Tool: {event.tool_name}, ID: {event.tool_use_id}')
        # TODO: Invoke Tools
        pass

    async def initialize(self) -> Self:
        """Initialize the stream manager and audio interface"""
        # Initiate the bidirectional stream
        self._br_stream = await self._br_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model_id)
        )

        # Send the initial system prompt
        await self._send_event(
            SessionStart(
                session_start=SessionStartPayload(
                    inference_configuration=InferenceConfiguration()
                )
            )
        )
        # Debug logging for AudioOutputConfiguration
        logger.info(
            f'🎙️ Creating AudioOutputConfiguration with voice_id: "{self._voice_name}"'
        )
        await self._send_event(
            PromptStart(
                prompt_start=PromptStartPayload(
                    prompt_name=self._prompt_name,
                    text_output_configuration=TextOutputConfiguration(),
                    audio_output_configuration=AudioOutputConfiguration(
                        voice_id=self._voice_name
                    ),
                    tool_use_output_configuration=ToolUseOutputConfiguration(),
                    tool_configuration=ToolConfiguration(tools=[]),
                )
            )
        )
        await self._send_event(
            ContentStart(
                content_start=ContentStartPayload(
                    prompt_name=self._prompt_name,
                    content_name=self._content_name,
                    type='TEXT',
                    role='SYSTEM',
                    interactive=True,
                    text_input_configuration=TextInputConfiguration(),
                )
            )
        )
        await self._send_event(
            TextInput(
                text_input=TextInputPayload(
                    prompt_name=self._prompt_name,
                    content_name=self._content_name,
                    content=self._system_prompt,
                )
            )
        )
        await self._send_event(
            ContentEnd(
                content_end=ContentEndPayload(
                    prompt_name=self._prompt_name, content_name=self._content_name
                )
            )
        )

        # Start processing incoming messages from Bedrock
        self._br_incoming_task = asyncio.create_task(self._process_br_incoming())
        self._user_incoming_task = asyncio.create_task(self._process_user_incoming())

        # Hook up the audio interface
        self._audio = AudioInterface(self._handle_audio_in).start()

        return self

    async def close(self):
        """Close the stream manager and clean up all resources"""
        tasks_to_cancel = []

        # Cancel running tasks
        if self._br_incoming_task is not None:
            self._br_incoming_task.cancel()
            tasks_to_cancel.append(self._br_incoming_task)

        if self._user_incoming_task is not None:
            self._user_incoming_task.cancel()
            tasks_to_cancel.append(self._user_incoming_task)

        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Stop the audio interface
        if self._audio is not None:
            try:
                await self._audio.stop()
            except Exception:
                # Continue cleanup even if audio stop fails
                traceback.print_exc()

        # Close the Bedrock stream
        if self._br_stream is not None:
            await self._send_event(
                ContentEnd(
                    content_end=ContentEndPayload(
                        prompt_name=self._prompt_name,
                        content_name=self._audio_content_name,
                    )
                )
            )
            await self._send_event(
                PromptEnd(prompt_end=PromptEndPayload(prompt_name=self._prompt_name))
            )
            await self._send_event(SessionEnd(session_end=SessionEndPayload()))
            try:
                await self._br_stream.close()
            except Exception:
                # Continue cleanup even if stream close fails
                traceback.print_exc()

        # Reset all references
        self._br_incoming_task = None
        self._user_incoming_task = None
        self._audio = None
        self._br_stream = None

    async def __aenter__(self) -> Self:
        """Async context manager entry - returns self without initialization"""
        return await self.initialize()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - automatically calls close()"""
        await self.close()

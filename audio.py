import asyncio
import logging
from asyncio import Task
from typing import Awaitable, Callable, Self

import pyaudio
from pyaudio import PyAudio

logger = logging.getLogger(__name__)

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024  # Number of frames per buffer


class AudioInterface:
    """Manages the audio interfaces for the application."""

    def __init__(
        self,
        input_handler: Callable[[bytes], Awaitable[None]],
    ):
        self._loop = asyncio.get_event_loop()
        self._pa = PyAudio()
        self._input_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._input_callback,
        )
        self._output_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        self._input_handler: Callable[[bytes], Awaitable[None]] = input_handler
        self._output_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._is_playing_output: bool = False
        self._output_task: Task[None] | None = None
        self._current_timer_task: Task[None] | None = None

    def start(self) -> Self:
        """Start the audio streams and begin processing."""
        logger.debug('Starting audio streams')
        self._input_stream.start_stream()
        self._output_task = asyncio.create_task(self._play_output_audio())
        return self

    async def stop(self):
        """Stop the audio streams and clean up resources."""
        # Cancel the timer task if it exists
        if self._current_timer_task is not None and not self._current_timer_task.done():
            self._current_timer_task.cancel()
            await self._current_timer_task

        # End the output task
        if self._output_task is not None:
            self._output_task.cancel()
            await self._output_task

        # Close the input stream
        if self._input_stream.is_active():
            self._input_stream.stop_stream()
        self._input_stream.close()

        # Close the output stream
        if self._output_stream.is_active():
            self._output_stream.stop_stream()
        self._output_stream.close()

    async def speak(self, data: bytes):
        await self._output_q.put(data)

    def _input_callback(self, audio_data: bytes | None, frame_count, time_info, status):
        if audio_data is not None:

            async def _handle_input(data: bytes):
                # Check the boolean inside the event loop for thread safety
                if not self._is_playing_output:
                    await self._input_handler(data)

            asyncio.run_coroutine_threadsafe(_handle_input(audio_data), self._loop)

        # PyAudio stream callback must return (data, continue_flag)
        # For input streams, we return (None, pyaudio.paContinue)
        return (None, pyaudio.paContinue)

    async def _play_output_audio(self):
        logger.debug('Starting audio playing task')
        while True:
            try:
                audio_data = await self._output_q.get()
                if audio_data is not None:
                    # Cancel any existing timer task
                    if self._current_timer_task and not self._current_timer_task.done():
                        self._current_timer_task.cancel()

                    # Set flag when starting to play audio
                    self._is_playing_output = True

                    # Calculate exact duration of this audio chunk
                    # Duration = bytes / (sample_rate * channels * bytes_per_sample)
                    duration_seconds = len(audio_data) / (
                        OUTPUT_SAMPLE_RATE * CHANNELS * 2
                    )

                    # Write audio data to PyAudio buffer
                    for i in range(0, len(audio_data), CHUNK_SIZE):
                        end = min(i + CHUNK_SIZE, len(audio_data))
                        chunk = audio_data[i:end]

                        await asyncio.get_running_loop().run_in_executor(
                            None, self._output_stream.write, chunk
                        )

                        # Give other tasks a chance
                        await asyncio.sleep(0.001)

                    # Schedule timer to clear flag when audio finishes playing
                    async def _clear_flag_after_playback(duration: float):
                        await asyncio.sleep(duration)
                        # Double-check no new audio was queued while we were waiting
                        if self._output_q.empty():
                            self._is_playing_output = False

                    self._current_timer_task = asyncio.create_task(
                        _clear_flag_after_playback(duration_seconds)
                    )

            except asyncio.CancelledError:
                break

    async def __aenter__(self) -> Self:
        return self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

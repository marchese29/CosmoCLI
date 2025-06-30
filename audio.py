import asyncio
from asyncio import Task
from typing import Awaitable, Callable, Self

import pyaudio
from pyaudio import PyAudio

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
        output_q: asyncio.Queue[bytes],
        interruption_event: asyncio.Event | None = None,
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
        self._output_q: asyncio.Queue[bytes] = output_q
        self._output_task: Task[None] | None = None
        self._interruption_event: asyncio.Event | None = interruption_event

    def start(self):
        """Start the audio streams and begin processing."""
        self._input_stream.start_stream()
        self._output_task = asyncio.create_task(self._play_output_audio())

    async def stop(self):
        """Stop the audio streams and clean up resources."""
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

    def _input_callback(self, audio_data: bytes | None, frame_count, time_info, status):
        if audio_data is not None:

            async def _handle_input(data: bytes):
                await self._input_handler(data)

            asyncio.run_coroutine_threadsafe(_handle_input(audio_data), self._loop)

        # PyAudio stream callback must return (data, continue_flag)
        # For input streams, we return (None, pyaudio.paContinue)
        return (None, pyaudio.paContinue)

    async def _play_output_audio(self):
        while True:
            try:
                # Check for interruption before processing
                if self._interruption_event and self._interruption_event.is_set():
                    # Clear the output queue
                    while not self._output_q.empty():
                        try:
                            self._output_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    # Clear the interruption flag
                    self._interruption_event.clear()

                    # Small delay after clearing
                    await asyncio.sleep(0.05)
                    continue

                audio_data = await self._output_q.get()
                if audio_data is not None:
                    for i in range(0, len(audio_data), CHUNK_SIZE):
                        end = min(i + CHUNK_SIZE, len(audio_data))
                        chunk = audio_data[i:end]

                        # Capture by value
                        def _write_data(data: bytes):
                            self._output_stream.write(data)

                        await asyncio.get_running_loop().run_in_executor(
                            None, _write_data, chunk
                        )

                        # Give other tasks a chance
                        await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

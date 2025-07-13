import logging
import os

import httpx

from util import SonicTools

logger = logging.getLogger(__name__)

_SERVER_ADDR = os.getenv('COSMO_SERVER_ADDRESS')
_SERVER_PORT = os.getenv('COSMO_SERVER_PORT')
_BASE_URL = f'http://{_SERVER_ADDR}:{_SERVER_PORT}'
SIMPLE_URL = f'{_BASE_URL}/simple'
COMPLEX_URL = f'{_BASE_URL}/complex'

_SIMPLE_DESCRIPTION = """
Send a simple request (such as controlling a small number of devices) to the home
controller agent."""
_COMPLEX_DESCRIPTION = """
Send a complex request (such as scene control or multi-step actions) to the home
controller agent."""

nova_tools = SonicTools()


@nova_tools.agent_tool('simple_request', description=_SIMPLE_DESCRIPTION)
async def simple_request(prompt: str) -> str:
    logging.info(f'🐰 Sending simple prompt to cosmo server: "{prompt}"')
    async with httpx.AsyncClient() as client:
        result = await client.post(SIMPLE_URL, json={'message': prompt})
        logging.info(f'TOOL RESULT: {result.text}')
        return result.text


@nova_tools.agent_tool('complex_request', description=_COMPLEX_DESCRIPTION)
async def complex_request(prompt: str) -> str:
    logging.info(f'🐰 Sending complex prompt to cosmo server: "{prompt}"')
    async with httpx.AsyncClient() as client:
        result = await client.post(COMPLEX_URL, json={'message': prompt})
        logging.info(f'TOOL RESULT: {result.text}')
        return result.text


async def server_is_connected() -> bool:
    async with httpx.AsyncClient() as client:
        result = await client.get(f'{_BASE_URL}/hello')
        if result.status_code != 200:
            logger.warning(f'🐰 /hello endpoint returned a {result.status_code}')
            return False
        return True

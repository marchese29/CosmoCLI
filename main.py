import asyncio
import logging
import os
import sys

from cosmovoice import BedrockStreamManager
from dotenv import load_dotenv

from tools import server_is_connected

load_dotenv()
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - (%(name)s) - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


async def main(system_prompt: str):
    """Main entry point for the program"""

    if not await server_is_connected():
        sys.exit('Cosmo Server not running')

    async with BedrockStreamManager(system_prompt):
        # Wait for the user to hit enter
        await asyncio.get_event_loop().run_in_executor(None, input)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'cosmo_prompt.txt')) as f:
        system_prompt = f.read()
    asyncio.run(main(system_prompt))

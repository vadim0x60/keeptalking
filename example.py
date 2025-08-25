import requests
from keeptalking import vibe
import asyncio
import logging
import openai
import logging

logging.basicConfig(level=logging.DEBUG)

@vibe()
async def is_general_purpose(descr) -> bool:
    """Filter out coding models and other non-general language models"""
    return f"Is this a general purpose language model?\n\n{descr}"

async def is_alive(slug):
    @vibe(model=slug)
    async def greet():
        """Greeting test to check if the LLM is alive"""
        return "Hi!"

    try:
        logging.info(await greet())
        return True
    except (openai.NotFoundError, openai.InternalServerError):
        return False

async def openrouter_models():
    models = requests.get("https://openrouter.ai/api/frontend/models/find?order=top-weekly").json()['data']['models']
    models = ((model, is_general_purpose(model['description'])) for model in models)
    models = (model for model, general_purpose in models if await general_purpose)
    models = ((model, is_alive(model['slug'])) async for model in models)
    models = (model async for model, alive in models if await alive)

    async for model in models:
        print(model['slug'])

if __name__ == '__main__':  
    asyncio.run(openrouter_models())
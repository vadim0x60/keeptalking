import os
from functools import wraps
from openai import AsyncOpenAI, OpenAI
from pydantic import create_model, BaseModel
import asyncio
import itertools

DEFAULT_TOKENS = 2048
DEFAULT_MODEL = 'google/gemini-2.5-flash'
DEFAULT_ROLES = itertools.chain(['system'], itertools.cycle(['user']))
MAX_ASYNC = 10

class SyncAsyncError(RuntimeError):
    pass

params = {
    'max_retries': int(os.getenv("MAX_RETRIES", 50))
}
base_url = os.getenv("BASE_URL")
if base_url:
    params['base_url'] = base_url
    params['api_key'] = os.getenv("API_KEY")
else:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        params['base_url'] = 'https://openrouter.ai/api/v1'
        params['api_key'] = api_key
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            params['api_key'] = api_key
        else:
            raise ValueError("No backend config found. Set BASE_URL and API_KEY or OPENROUTER_API_KEY or OPENAI_API_KEY environment variables")

client_sync = OpenAI(**params)
client_async = AsyncOpenAI(**params)
sem = asyncio.Semaphore(MAX_ASYNC)

def _chat(client, model, roles, messages, structure, tokens):
    messages = [{'role': role, 'content': content} for role, content in zip(roles, messages)]

    with sem:
        if structure == str:
            return (client.chat.completions.create(model=model, messages=messages, max_completion_tokens=tokens), 
                    lambda response: response.choices[0].message.content)
        else:
            if issubclass(structure, BaseModel):
                return (client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=structure,
                    max_completion_tokens=tokens
                ), lambda response: response.choices[0].message.parsed)
            else:
                # OpenAI interface supports pydantic models only
                # But we can just wrap anything into a pydantic model and unwrap it back
                # This will be our little secret
                return (client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=create_model('ModelResponse', response=structure),
                    max_completion_tokens=tokens
                ), lambda response: response.choices[0].message.parsed.response)  

async def write(model=DEFAULT_MODEL, roles=DEFAULT_ROLES, messages=[], structure=str, tokens=DEFAULT_TOKENS):
    response, postproc = _chat(client_async, model, roles=roles, messages=messages, structure=structure, tokens=tokens)
    return postproc(await response)

def talk(model=DEFAULT_MODEL, roles=DEFAULT_ROLES, messages=[], structure=str, tokens=DEFAULT_TOKENS):
    response, postproc = _chat(client_sync, model, roles=roles, messages=messages, structure=structure, tokens=tokens)
    return postproc(response)

def vibe(model=DEFAULT_MODEL, tokens=DEFAULT_TOKENS):
    def _vibe(f):
        rt = f.__annotations__.get('return', str)
        if asyncio.iscoroutinefunction(f):
            @wraps(f)
            async def __vibe(*args, **kwargs):
                return await write(model, messages=[f.__doc__, await f(*args, **kwargs)], 
                                   structure=rt, tokens=tokens)
        else:
            @wraps(f)
            def __vibe(*args, **kwargs):
                return talk(model, messages=[f.__doc__, f(*args, **kwargs)], structure=rt, tokens=tokens)
        return __vibe
    return _vibe
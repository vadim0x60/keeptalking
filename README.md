# keeptalking

A simple, pythonic interface to any OpenAI-compatible LLM server.
You will never type `response.choices[0].message.content` ever again.

## Installation

```bash
pip install keeptalking
```

## Usage

The entire library is 3 functions:

```python
from keeptalking import talk, write, vibe
```

### Conversation

```python
talk(model='google/gemini-2.5-flash', 
     roles=['system', 'user'], 
     messages=['Solve a math problem', 'Sum up all possible bases in which 97 is divisible by 17'],
     structure=int,
     tokens=10)
```

will use grammar constrained decoding and return a single integer with the answer.
The return value of `talk` will always be of type `structure`, which defaults to `str` is omitted.
If `roles` are omitted, the first message is considered a system message, the rest are user messages.
If `model` is omitted, `gemini-2.5-flash` is used (default model can be overriden by setting the `MODEL` environment variable).
If `tokens` is omitted, generation is limited to 2048 new tokens (default token limit can be overridden by setting the `TOKENS` environment variable).

The only parameter that should not be omitted is `messages`:

```python
talk(['Solve a math problem. Provide your reasoning', 'Sum up all possible bases in which 97 is divisible by 17'])
```

`write` is an asynchronous version of `talk` that lets you beautifully parallelize batch requests:

```python
asyncio.gather(
    write(model='google/gemini-2.5-flash', 
          roles=['system', 'user'], 
          messages=[sys, berry],
          structure=int,
          tokens=10)
    for berry in ['Strawberry', 'Blackberry', 'Raspberry', 'Blueberry', 'Canterbury']
)
```

`write` automatically self-throttles as necessary so it's safe to call thousands of `write()`s in parallel with no external rate limiting.

### Vibe functions

Vibe functions are functions defined in natural language.

```python
@vibe(model='google/gemini-2.5-flash', tokens=10)
def do_job(job_details):
    """System message"""
    return f"User message with {job_details}"
```

[ELL](https://github.com/madcowd/ell) users will notice that this format is ~~shamelessly stolen~~ inspired by ELL.
However, `keeptalking` is much simpler than ELL.
Despite being much simpler, `keeptalking` supports additional features like async vibe functions

```python
@vibe()
async def homework_assistant(topic, pages=5):
    """Help the student with their homework"""
    return f"Write a {pages}-page essay on {topic}"
```

fully parallelizable like so

```python
asyncio.gather(
    homework_assistant('Math'),
    homework_assistant('History'),
    homework_assistant('English')
)
```

and enabling structured outputs with a single type hint:

```python
@vibe()
def count_rs(request) -> int:
    """Count how many Rs are in the request"""
    return request
```

unlike in the rest of the Python ecosystem, in vibe functions type hints actually ensure that the return value is always of the type in question

## Backend configuration

The model server to be used is defined via environment variables.
It can be defined directly by setting `BASE_URL` and `API_KEY`.
If those are not set, `keeptalking` will default to [OpenRouter](https://openrouter.ai) if `OPENROUTER_API_KEY` is set, then [OpenAI](https://openai.com) if `OPENAI_API_KEY` is set.
~~Perv~~ advanced users can monkey patch `keeptalking.client_sync` and `keeptalking.client_async` instead.

## Example

You will find a detailed example in [example.py](example.py). It takes top 10 models from openrouter's model catalog and reads text description of each model to filter out specialized models like coding or edit models, then runs a small test on each to check if they are working.

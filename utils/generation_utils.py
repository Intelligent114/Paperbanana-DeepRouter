# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# Initialize clients lazily or with robust defaults
api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    gemini_client = genai.Client(api_key=api_key)
    print("Initialized Gemini Client with API Key")
else:
    print("Warning: Could not initialize Gemini Client. Missing credentials.")
    gemini_client = None


anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
anthropic_base_url = get_config_val("base_urls", "anthropic_base_url", "ANTHROPIC_BASE_URL", "")
if anthropic_api_key:
    anthropic_client_kwargs = {"api_key": anthropic_api_key}
    if anthropic_base_url:
        anthropic_client_kwargs["base_url"] = anthropic_base_url
    anthropic_client = AsyncAnthropic(**anthropic_client_kwargs)
    print(f"Initialized Anthropic Client with API Key{' and custom base_url' if anthropic_base_url else ''}")
else:
    print("Warning: Could not initialize Anthropic Client. Missing credentials.")
    anthropic_client = None

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
openai_base_url = get_config_val("base_urls", "openai_base_url", "OPENAI_BASE_URL", "")
# "chat" -> /v1/chat/completions  |  "responses" -> /v1/responses
openai_api_style = get_config_val("", "openai_api_style", "OPENAI_API_STYLE", "chat")
if not openai_api_style:
    openai_api_style = model_config.get("openai_api_style", "chat") or "chat"

# API style for image models; falls back to openai_api_style if not explicitly set
image_api_style = model_config.get("image_api_style") or os.getenv("IMAGE_API_STYLE") or openai_api_style

if openai_api_key:
    openai_client_kwargs = {"api_key": openai_api_key}
    if openai_base_url:
        openai_client_kwargs["base_url"] = openai_base_url
    openai_client = AsyncOpenAI(**openai_client_kwargs)
    print(f"Initialized OpenAI Client with API Key{' and custom base_url' if openai_base_url else ''}")
else:
    print("Warning: Could not initialize OpenAI Client. Missing credentials.")
    openai_client = None

# --- Image model client (may use a different API key / base URL) ---
# Falls back to openai_api_key / openai_base_url when image-specific values are not set.
image_openai_api_key = get_config_val("api_keys", "image_openai_api_key", "IMAGE_OPENAI_API_KEY", "") or openai_api_key
image_openai_base_url = get_config_val("base_urls", "image_openai_base_url", "IMAGE_OPENAI_BASE_URL", "") or openai_base_url
image_custom_endpoint = get_config_val("base_urls", "image_custom_endpoint", "IMAGE_CUSTOM_ENDPOINT", "")

if image_openai_api_key:
    _img_same_as_text = (image_openai_api_key == openai_api_key and image_openai_base_url == openai_base_url)
    if _img_same_as_text and openai_client is not None:
        # Reuse the same client instance to avoid unnecessary connections
        image_openai_client = openai_client
        print("Image OpenAI Client: reusing text model client (same credentials)")
    else:
        image_client_kwargs = {"api_key": image_openai_api_key}
        if image_openai_base_url:
            image_client_kwargs["base_url"] = image_openai_base_url
        image_openai_client = AsyncOpenAI(**image_client_kwargs)
        print(f"Initialized Image OpenAI Client with separate API Key{' and custom base_url' if image_openai_base_url else ''}")
else:
    print("Warning: Could not initialize Image OpenAI Client. Missing credentials.")
    image_openai_client = None


def get_model_backend(model_name: str) -> str:
    """
    Determine which backend to use for a given model name.

    Rules:
    - If openai_base_url is configured, ALL models are routed through the OpenAI-compatible
      client (third-party provider). The model name is passed as-is to the API.
    - Otherwise fall back to native SDK routing:
        * "gemini" in name  -> "gemini"
        * "claude" / "anthropic" in name -> "claude"
        * default -> "openai"
    """
    if openai_base_url:
        return "openai"
    if "gemini" in model_name.lower():
        return "gemini"
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        return "claude"
    return "openai"



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.

    When `openai_base_url` is configured (third-party provider), this function
    transparently forwards the request to `call_openai_with_retry_async` so that
    all existing callers work without modification.
    """
    # --- Third-party provider redirect ---
    if openai_base_url:
        # Extract generation parameters from the Gemini config object gracefully
        system_prompt = ""
        temperature = 1.0
        candidate_num = 1
        max_output_tokens = 50000
        if hasattr(config, "system_instruction"):
            system_prompt = config.system_instruction or ""
        if hasattr(config, "temperature") and config.temperature is not None:
            temperature = config.temperature
        if hasattr(config, "candidate_count") and config.candidate_count is not None:
            candidate_num = config.candidate_count
        if hasattr(config, "max_output_tokens") and config.max_output_tokens is not None:
            max_output_tokens = config.max_output_tokens

        openai_config = {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "candidate_num": candidate_num,
            "max_completion_tokens": max_output_tokens,
        }
        if openai_api_style == "responses":
            return await call_openai_responses_with_retry_async(
                model_name=model_name,
                contents=contents,
                config=openai_config,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )
        return await call_openai_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=openai_config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key. "
            "Please set GOOGLE_API_KEY in environment, or configure api_keys.google_api_key in configs/model_config.yaml."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                # OpenAI expects data URL format
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_responses_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Responses API (/v1/responses) with retry logic.
    Used when openai_api_style = "responses" (e.g. DeepRouter /v1/responses endpoint).

    The Responses API uses:
      client.responses.create(model, input, instructions, ...)
    and returns response.output_text directly.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_completion_tokens"]
    response_text_list = []

    # Build the input in Responses API format.
    # Text-only -> pass as plain string for simplicity.
    # Multimodal -> build a user message with image_url parts.
    def _build_responses_input(c):
        parts = _convert_to_openai_format(c)
        # If all parts are text, collapse to a single string
        if all(p.get("type") == "text" for p in parts):
            return " ".join(p["text"] for p in parts)
        # Otherwise wrap in a user message object
        return [{"role": "user", "content": parts}]

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            input_contents = _build_responses_input(contents)
            first_response = await openai_client.responses.create(
                model=model_name,
                input=input_contents,
                instructions=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            response_text_list.append(first_response.output_text if first_response.output_text is not None else "")
            is_input_valid = True
            break
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Responses API attempt {attempt + 1} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(f"Error: All {max_attempts} attempts failed{context_msg}. Returning errors.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        input_contents = _build_responses_input(contents)
        tasks = [
            openai_client.responses.create(
                model=model_name,
                input=input_contents,
                instructions=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.output_text if res.output_text is not None else "")

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=10, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            # Guard against None content (some providers return null on empty output)
            content = first_response.choices[0].message.content
            response_text_list.append(content if content is not None else "")
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                content = res.choices[0].message.content
                response_text_list.append(content if content is not None else "")

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_image_model_with_retry_async(
    model_name,
    prompt,
    system_prompt="",
    input_image_b64=None,
    input_image_media_type="image/jpeg",
    aspect_ratio="1:1",
    image_size="1k",
    temperature=1.0,
    max_output_tokens=8192,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """Unified entrypoint for image generation/editing used by both visualizer and demo refine."""
    backend = get_model_backend(model_name)

    if backend == "gemini":
        contents = [{"type": "text", "text": prompt}]
        if input_image_b64:
            contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": input_image_media_type,
                    "data": input_image_b64,
                },
            })

        size_map = {
            "1K": "1k",
            "2K": "2k",
            "4K": "4k",
            "1k": "1k",
            "2k": "2k",
            "4k": "4k",
        }
        gemini_image_size = size_map.get(image_size, str(image_size).lower())

        return await call_gemini_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt or None,
                temperature=temperature,
                candidate_count=1,
                max_output_tokens=max_output_tokens,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=gemini_image_size,
                ),
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    return await call_openai_image_chat_with_retry_async(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        input_image_b64=input_image_b64,
        input_image_media_type=input_image_media_type,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_openai_image_chat_with_retry_async(
    model_name,
    prompt,
    system_prompt="",
    input_image_b64=None,
    input_image_media_type="image/jpeg",
    max_attempts=10,
    retry_delay=30,
    error_context="",
):
    """
    ASYNC: Generate an image via an OpenAI-compatible endpoint.
    Supports both chat completions (/v1/chat/completions) and
    Responses API (/v1/responses) depending on image_api_style.
    Uses image_openai_client (which may have a separate API key / base URL).

    The response may be:
      - A data: URL with inline base64
      - A raw base64 string
      - A Markdown image link: ![...](https://...)
      - A plain https:// URL pointing to an image

    In the URL cases the image is downloaded automatically and returned as base64.

    Returns a list with one base64-encoded PNG/JPEG string, or ["Error"] on failure.
    """
    import re as _re

    def _extract_image_or_url(content: str):
        """Return (b64_str, url) from a text response. Exactly one will be non-None."""
        # 1. Inline data URL
        data_url_match = _re.search(
            r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content
        )
        if data_url_match:
            return data_url_match.group(1), None

        # 2. Markdown image link: ![alt](url)
        md_match = _re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", content)
        if md_match:
            return None, md_match.group(1)

        # 3. Plain URL that looks like an image (with or without extension)
        url_match = _re.search(
            r"https?://\S+?\.(?:jpg|jpeg|png|gif|webp)(?:[?#]\S*)?",
            content,
            _re.IGNORECASE,
        )
        if url_match:
            return None, url_match.group(0)

        # 4. Bare https URL on its own line (CDN links without extension)
        bare_url_match = _re.search(r"^(https?://\S+)$", content.strip(), _re.MULTILINE)
        if bare_url_match:
            return None, bare_url_match.group(1)

        # 5. Raw base64 string
        stripped = content.strip()
        if stripped and _re.fullmatch(r"[A-Za-z0-9+/=\n]+", stripped) and len(stripped) > 200:
            return stripped.replace("\n", ""), None

        return None, None

    async def _download_url_to_b64(url: str) -> str | None:
        """Download an image from a URL and return as base64, or None on failure."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return base64.b64encode(resp.content).decode("utf-8")
        except Exception as e:
            print(f"[Warning] Failed to download image from {url}: {e}")
            return None

    def _normalise_content(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get("text"):
                        text_parts.append(item["text"])
                    elif item.get("type") == "output_text" and item.get("text"):
                        text_parts.append(item["text"])
            return "\n".join(text_parts)
        return str(content)

    async def _call_custom_image_endpoint(messages):
        import httpx

        if not image_custom_endpoint:
            return None

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        if image_api_style == "responses":
            payload = {
                "model": model_name,
                "input": messages,
                "stream": False,
            }
            if system_prompt:
                payload["instructions"] = system_prompt

        headers = {
            "Authorization": f"Bearer {image_openai_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(image_custom_endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Try a few common response shapes.
        if isinstance(data, dict):
            if data.get("output_text"):
                return _normalise_content(data.get("output_text"))
            if data.get("choices"):
                choice = data["choices"][0]
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                return _normalise_content(message.get("content"))
            if data.get("data") and isinstance(data["data"], list):
                first_item = data["data"][0]
                if isinstance(first_item, dict):
                    if first_item.get("b64_json"):
                        return first_item["b64_json"]
                    if first_item.get("url"):
                        return first_item["url"]
        return _normalise_content(data)

    for attempt in range(max_attempts):
        try:
            if image_api_style == "responses":
                if input_image_b64:
                    response_input = [{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:{input_image_media_type};base64,{input_image_b64}",
                            },
                        ],
                    }]
                else:
                    response_input = prompt

                messages = response_input
                if image_custom_endpoint:
                    content = await _call_custom_image_endpoint(messages)
                else:
                    response = await image_openai_client.responses.create(
                        model=model_name,
                        input=response_input,
                        instructions=system_prompt or None,
                    )
                    content = response.output_text or ""
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                user_content = []
                if prompt:
                    user_content.append({"type": "text", "text": prompt})
                if input_image_b64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{input_image_media_type};base64,{input_image_b64}"
                        }
                    })

                messages.append({
                    "role": "user",
                    "content": user_content if user_content else prompt,
                })
                if image_custom_endpoint:
                    content = await _call_custom_image_endpoint(messages)
                else:
                    response = await image_openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                    )
                    content = _normalise_content(response.choices[0].message.content)

            b64, url = _extract_image_or_url(content)

            if b64:
                return [b64]

            if url:
                print(f"[Info] call_openai_image_chat: response contains image URL, downloading... {url[:100]}")
                b64 = await _download_url_to_b64(url)
                if b64:
                    return [b64]
                # Download failed – retry the whole API call
                print(f"[Warning] Image download failed, retrying API call (attempt {attempt + 1})...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

            print(
                f"[Warning] call_openai_image_chat: response does not appear to contain "
                f"a base64 image or URL. Raw content preview: {content[:200]}"
            )
            # Unrecognisable content – no point retrying the same call; return as-is
            return [content]

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for image-chat model {model_name} failed"
                f"{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]



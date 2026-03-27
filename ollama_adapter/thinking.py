import re
import json

from ollama_adapter import state


def remove_thinking_tags(content, model_id, remove_enabled):
    """
    Remove <think> or <thinking> tags from the beginning of content if enabled.
    Logs removed thinking content at DEBUG level.
    """
    if not remove_enabled or not content:
        return content

    pattern = r'^\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*'

    match = re.match(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        thinking_content = match.group(1)
        cleaned_content = content[match.end():]

        thinking_preview = thinking_content[:100] + "..." if len(thinking_content) > 100 else thinking_content
        state.logger.debug(f"Removed thinking tags from model '{model_id}'. Thinking content ({len(thinking_content)} chars): {thinking_preview}")

        return cleaned_content

    return content


def _process_stream(response_stream, remove_tags, model_id, display_name, make_chunk, usage):
    """
    Process OpenAI streaming response, optionally removing thinking tags.
    Shared logic for both /api/chat and /api/generate endpoints.

    Args:
        response_stream: OpenAI streaming response iterator
        remove_tags: Boolean — whether to strip <think>/<thinking> tags
        model_id: Model identifier for logging
        display_name: Display name for response chunks
        make_chunk: Callable(display_name, content) -> dict for endpoint-specific chunk format
        usage: Mutable dict to store {'prompt_tokens': int, 'completion_tokens': int}

    Yields:
        JSON-encoded response lines (str ending with '\\n')
    """
    if remove_tags:
        buffer = ""
        thinking_buffer = ""
        close_tag_buffer = ""

        # STATE: INITIAL — capture first chunk
        state_name = "INITIAL"

        for chunk in response_stream:
            if chunk.usage:
                usage['prompt_tokens'] = chunk.usage.prompt_tokens
                usage['completion_tokens'] = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if not content:
                continue

            if state_name == "INITIAL":
                buffer = content
                state_name = "DETECTING_OPEN_TAG"
                continue

            elif state_name == "DETECTING_OPEN_TAG":
                buffer += content
                buffer_lower = buffer.lstrip().lower()

                found_tag = False
                for tag in ["<think>", "<thinking>"]:
                    if buffer_lower.startswith(tag):
                        whitespace_len = len(buffer) - len(buffer.lstrip())
                        tag_end_pos = whitespace_len + len(tag)
                        buffer = buffer[tag_end_pos:]
                        thinking_buffer = ""
                        state_name = "BUFFERING_THINKING"
                        found_tag = True
                        state.logger.debug(f"Detected opening {tag} tag for model '{model_id}'")
                        break

                if found_tag:
                    continue

                if len(buffer_lower) > 20 or (len(buffer.lstrip()) > 0 and buffer.lstrip()[0] != '<'):
                    state_name = "STREAMING_NORMAL"
                    state.logger.debug(f"No thinking tag detected for model '{model_id}'")
                    if buffer:
                        yield json.dumps(make_chunk(display_name, buffer)) + '\n'
                        buffer = ""
                continue

            elif state_name == "BUFFERING_THINKING":
                buffer += content

                if "</" in buffer:
                    close_idx = buffer.index("</")
                    thinking_buffer += buffer[:close_idx]
                    close_tag_buffer = buffer[close_idx:]
                    buffer = ""
                    state_name = "DETECTING_CLOSE_TAG"
                else:
                    if len(buffer) > 1000:
                        thinking_buffer += buffer
                        buffer = ""
                continue

            elif state_name == "DETECTING_CLOSE_TAG":
                close_tag_buffer += content
                close_lower = close_tag_buffer.lower()

                found_close = False
                for tag in ["</think>", "</thinking>"]:
                    if close_lower.startswith(tag):
                        remainder = close_tag_buffer[len(tag):].lstrip()

                        preview = thinking_buffer[:100] + "..." if len(thinking_buffer) > 100 else thinking_buffer
                        state.logger.debug(f"Removed thinking tags from model '{model_id}'. Thinking content ({len(thinking_buffer)} chars): {preview}")

                        state_name = "STREAMING_NORMAL"
                        thinking_buffer = ""
                        close_tag_buffer = ""
                        found_close = True

                        if remainder:
                            yield json.dumps(make_chunk(display_name, remainder)) + '\n'
                        break

                if found_close:
                    continue

                if len(close_tag_buffer) > 15 or ('>' in close_tag_buffer and not any(close_lower.startswith(t[:len(close_lower)]) for t in ["</think>", "</thinking>"])):
                    thinking_buffer += close_tag_buffer
                    close_tag_buffer = ""
                    state_name = "BUFFERING_THINKING"
                continue

            elif state_name == "STREAMING_NORMAL":
                yield json.dumps(make_chunk(display_name, content)) + '\n'

        # End of stream — flush remaining buffered content
        if state_name == "DETECTING_OPEN_TAG" and buffer:
            yield json.dumps(make_chunk(display_name, buffer)) + '\n'
        elif state_name in ["BUFFERING_THINKING", "DETECTING_CLOSE_TAG"]:
            fallback = thinking_buffer + close_tag_buffer
            state.logger.warning(f"Stream ended while buffering thinking content for model '{model_id}'. No closing tag found. Outputting {len(fallback)} chars as fallback.")
            if fallback.strip():
                yield json.dumps(make_chunk(display_name, fallback)) + '\n'

    else:
        # Feature disabled — simple pass-through
        for chunk in response_stream:
            if chunk.usage:
                usage['prompt_tokens'] = chunk.usage.prompt_tokens
                usage['completion_tokens'] = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield json.dumps(make_chunk(display_name, content)) + '\n'

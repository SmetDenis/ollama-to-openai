"""Tests for system_prompt_mode, developer-role placement, list-part caching and related debug output."""

import pytest
import yaml

from ollama_adapter import state
from ollama_adapter.config import load_config
from ollama_adapter.debug_prompt import build_debug_content, last_user_text
from ollama_adapter.models import (
    apply_ip_routing,
    apply_prompt_caching,
    apply_system_prompt,
    get_model_config,
    merge_system_content,
    place_system_message,
)


class TestMergeSystemContent:
    @pytest.mark.parametrize(
        ("client", "mode", "expected"),
        [
            ("client", "replace", "CFG"),
            ("client", "prepend", "CFG\n\nclient"),
            ("client", "append", "client\n\nCFG"),
            (None, "prepend", "CFG"),
            ("   ", "append", "CFG"),
            ([], "prepend", "CFG"),
        ],
    )
    def test_matrix(self, client, mode, expected):
        assert merge_system_content(client, "CFG", mode) == expected

    def test_list_parts(self):
        parts = [{"type": "text", "text": "client"}]
        assert merge_system_content(parts, "CFG", "prepend") == [{"type": "text", "text": "CFG"}, *parts]
        assert merge_system_content(parts, "CFG", "append") == [*parts, {"type": "text", "text": "CFG"}]

    def test_unsupported_type_falls_back_to_replace(self):
        assert merge_system_content(42, "CFG", "append") == "CFG"


class TestPlaceSystemMessageModes:
    def test_replace_is_exact(self):
        msgs = [{"role": "system", "content": "old", "name": "x"}, {"role": "user", "content": "hi"}]
        assert place_system_message(msgs, "NEW")[0] == {"role": "system", "content": "NEW"}

    def test_merge_keeps_other_keys(self):
        msgs = [{"role": "system", "content": "old", "name": "x"}]
        assert place_system_message(msgs, "NEW", mode="prepend")[0] == {
            "role": "system",
            "content": "NEW\n\nold",
            "name": "x",
        }

    def test_developer_role_is_target_and_kept(self):
        msgs = [{"role": "developer", "content": "tools"}, {"role": "user", "content": "hi"}]
        assert place_system_message(msgs, "CFG", mode="append")[0] == {
            "role": "developer",
            "content": "tools\n\nCFG",
        }
        assert place_system_message(msgs, "CFG")[0] == {"role": "developer", "content": "CFG"}

    @pytest.mark.parametrize("mode", ["replace", "prepend", "append"])
    def test_insert_when_absent(self, mode):
        result = place_system_message([{"role": "user", "content": "hi"}], "CFG", mode=mode)
        assert result[0] == {"role": "system", "content": "CFG"}

    def test_apply_system_prompt_reads_mode(self):
        msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "hi"}]
        result = apply_system_prompt(msgs, {"system_prompt_inline": "CFG", "system_prompt_mode": "prepend"}, "m")
        assert result[0]["content"] == "CFG\n\nagent"


class TestModeRouting:
    def test_ip_routing_overrides_mode(self):
        state.CONFIG = {**state.CONFIG, "clients": {}}
        entry = {
            "name": "m",
            "system_prompt_mode": "replace",
            "ip_routing": [{"ip": "1.1.1.1", "system_prompt_mode": "append"}],
        }
        assert apply_ip_routing(entry, "1.1.1.1")["system_prompt_mode"] == "append"

    def test_get_model_config_exposes_mode(self):
        state.CONFIG = {**state.CONFIG, "models": [{"name": "m", "system_prompt_mode": "prepend"}]}
        _, adapter, _ = get_model_config("m")
        assert adapter["system_prompt_mode"] == "prepend"


class TestModeConfigValidation:
    def _write(self, tmp_path, minimal_config, model):
        path = tmp_path / "c.yml"
        path.write_text(yaml.dump({**minimal_config, "models": [model]}), encoding="utf-8")
        return str(path)

    def test_valid(self, tmp_path, minimal_config):
        load_config(self._write(tmp_path, minimal_config, {"name": "m", "system_prompt_mode": "append"}))

    def test_invalid_root(self, tmp_path, minimal_config):
        with pytest.raises(ValueError, match="system_prompt_mode"):
            load_config(self._write(tmp_path, minimal_config, {"name": "m", "system_prompt_mode": "merge"}))

    def test_invalid_ip_routing(self, tmp_path, minimal_config):
        model = {"name": "m", "ip_routing": [{"ip": "1.1.1.1", "system_prompt_mode": 1}]}
        with pytest.raises(ValueError, match=r"ip_routing\[0\]"):
            load_config(self._write(tmp_path, minimal_config, model))


class TestListPartCaching:
    def test_marks_last_text_part_without_mutation(self):
        parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]
        msgs = [{"role": "system", "content": parts}]
        result = apply_prompt_caching(msgs, {"prompt_caching": True}, "m")
        assert result[0]["content"][2]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[0]["content"][0]
        assert "cache_control" not in parts[2]

    def test_existing_marker_respected(self):
        parts = [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "b"}]
        result = apply_prompt_caching([{"role": "system", "content": parts}], {"prompt_caching": True}, "m")
        assert result[0]["content"] == parts

    def test_developer_string_cached(self):
        result = apply_prompt_caching([{"role": "developer", "content": "x"}], {"prompt_caching": True}, "m")
        assert result[0]["role"] == "developer"
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


class TestDebugIntegration:
    def test_last_user_text_parts(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "de"}, {"type": "image_url"}, {"type": "text", "text": "bug"}],
            }
        ]
        assert last_user_text(msgs, include_parts=True) == "de\nbug"
        assert last_user_text(msgs) is None
        assert last_user_text([{"role": "user", "content": [{"type": "image_url"}]}], include_parts=True) is None

    def test_mode_note_and_merge(self, prompts_dir):
        msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "debug"}]
        out = build_debug_content(msgs, {"system_prompt_inline": "CFG", "system_prompt_mode": "prepend"}, "m")
        assert "system_prompt_mode=prepend" in out
        assert out.index("CFG") < out.index("agent")

    def test_no_mode_note_on_replace(self, prompts_dir):
        msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "debug"}]
        out = build_debug_content(msgs, {"system_prompt_inline": "CFG"}, "m")
        assert "system_prompt_mode" not in out
        assert "agent" not in out

    def test_data_urls_shortened(self, prompts_dir):
        url = "data:image/png;base64," + "A" * 5000
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
        out = build_debug_content(msgs, {}, "m")
        assert "data:image/png;base64,<5000 chars omitted>" in out
        assert "A" * 200 not in out

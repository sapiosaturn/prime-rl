from datasets import Dataset

from prime_rl.configs.sft import LossMaskConfig
from prime_rl.trainer.sft.data import SFTDataset

ROLE_TOKENS = {
    "system": 10,
    "user": 20,
    "assistant": 30,
    "tool": 40,
}


class DummyChatTokenizer:
    def __init__(self, assistant_mask_behavior: str = "valid", is_fast: bool = True):
        self.assistant_mask_behavior = assistant_mask_behavior
        self.is_fast = is_fast
        self.eos_token_id = 999
        self.calls: list[dict] = []

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in ids)

    def _render(self, conversation: list[dict], add_generation_prompt: bool) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        for message in conversation:
            role = message["role"]
            content = message.get("content") or ""
            input_ids.extend([ROLE_TOKENS[role], 100 + len(content)])
            assistant_masks.extend([0, 1 if role == "assistant" else 0])
        if add_generation_prompt:
            input_ids.append(ROLE_TOKENS["assistant"])
            assistant_masks.append(0)
        return input_ids, assistant_masks

    def apply_chat_template(
        self,
        conversation: list[dict],
        tools=None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **kwargs,
    ):
        self.calls.append(
            {
                "conversation": conversation,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
                "return_assistant_tokens_mask": return_assistant_tokens_mask,
                "kwargs": kwargs,
            }
        )

        input_ids, assistant_masks = self._render(conversation, add_generation_prompt)
        if return_assistant_tokens_mask:
            if self.assistant_mask_behavior == "valid":
                return {"input_ids": input_ids, "assistant_masks": assistant_masks}
            if self.assistant_mask_behavior == "missing":
                return {"input_ids": input_ids}
            if self.assistant_mask_behavior == "mismatch":
                return {"input_ids": input_ids, "assistant_masks": assistant_masks[:-1]}
            raise ValueError(f"Unsupported assistant_mask_behavior={self.assistant_mask_behavior}")

        if return_dict:
            return {"input_ids": input_ids}
        return input_ids


def build_dataset_example() -> dict:
    return {
        "prompt": [{"role": "user", "content": "  hi  "}],
        "completion": [{"role": "assistant", "content": "  ok  "}],
        "chat_template_kwargs": {
            "tokenize": False,
            "return_dict": False,
            "return_assistant_tokens_mask": False,
            "return_tensors": "pt",
            "chat_template": "custom-template",
        },
    }


def test_assistant_mask_auto_uses_one_pass_tokenization():
    tokenizer = DummyChatTokenizer(assistant_mask_behavior="valid", is_fast=True)
    dataset = SFTDataset(
        Dataset.from_list([build_dataset_example()]),
        tokenizer=tokenizer,
        shuffle=False,
        max_examples=1,
        loss_mask_config=LossMaskConfig(),
        loss_mask_mode="assistant_mask_auto",
    )

    sample = next(iter(dataset))

    assert sample["input_ids"] == [20, 106, 30, 106]
    assert sample["target_ids"] == [106, 30, 106, 999]
    assert sample["loss_mask"] == [False, False, True, True]
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["return_assistant_tokens_mask"] is True
    assert tokenizer.calls[0]["kwargs"] == {"chat_template": "custom-template"}


def test_incremental_mode_keeps_existing_loop_behavior():
    tokenizer = DummyChatTokenizer(assistant_mask_behavior="valid", is_fast=True)
    dataset = SFTDataset(
        Dataset.from_list([build_dataset_example()]),
        tokenizer=tokenizer,
        shuffle=False,
        max_examples=1,
        loss_mask_config=LossMaskConfig(),
        loss_mask_mode="incremental",
    )

    sample = next(iter(dataset))

    assert sample["input_ids"] == [20, 102, 30, 102]
    assert sample["target_ids"] == [102, 30, 102, 999]
    assert sample["loss_mask"] == [False, False, True, True]
    assert len(tokenizer.calls) == 3
    assert all(call["return_assistant_tokens_mask"] is False for call in tokenizer.calls)


def test_assistant_mask_auto_falls_back_to_incremental_when_mask_is_missing():
    tokenizer = DummyChatTokenizer(assistant_mask_behavior="missing", is_fast=True)
    dataset = SFTDataset(
        Dataset.from_list([build_dataset_example()]),
        tokenizer=tokenizer,
        shuffle=False,
        max_examples=1,
        loss_mask_config=LossMaskConfig(),
        loss_mask_mode="assistant_mask_auto",
    )

    sample = next(iter(dataset))

    assert sample["input_ids"] == [20, 102, 30, 102]
    assert sample["target_ids"] == [102, 30, 102, 999]
    assert sample["loss_mask"] == [False, False, True, True]
    assert len(tokenizer.calls) == 4
    assert tokenizer.calls[0]["return_assistant_tokens_mask"] is True
    assert all(call["return_assistant_tokens_mask"] is False for call in tokenizer.calls[1:])

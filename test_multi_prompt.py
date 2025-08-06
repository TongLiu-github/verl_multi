#!/usr/bin/env python3
"""
Test script for multi-prompt rollout functionality
"""

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_multi_prompt import vLLMRollout


def test_multi_prompt_rollout():
    """Test the multi-prompt rollout functionality."""
    
    # Create a mock config
    config = OmegaConf.create({
        "prompt_length": 512,
        "response_length": 2048,
        "max_model_len": 2560,
        "tensor_model_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.6,
        "enforce_eager": True,
        "free_cache_engine": True,
        "load_format": "dummy_dtensor",
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 1024,
        "enable_chunked_prefill": False,
        "val_kwargs": {
            "top_k": 50,
            "top_p": 0.7,
            "temperature": 0.0
        }
    })
    
    # Mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Mock model config
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    print("Testing multi-prompt rollout...")
    
    # Test prompt variants generation
    rollout = vLLMRollout(
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        config=config,
        tokenizer=tokenizer,
        model_hf_config=model_config
    )
    
    # Test _generate_prompt_variants
    test_question = "What is 2 + 2?"
    variants = rollout._generate_prompt_variants(test_question)
    print(f"Generated {len(variants)} prompt variants:")
    for i, variant in enumerate(variants):
        print(f"  Variant {i+1}: {variant}")
    
    # Test _tokenize_prompt_variants
    tokenized = rollout._tokenize_prompt_variants(variants)
    print(f"Tokenized variants shape: {tokenized['input_ids'].shape}")
    
    # Test with mock data
    mock_data = DataProto(
        batch={
            "input_ids": torch.randn(2, 512),
            "attention_mask": torch.ones(2, 512),
            "position_ids": torch.arange(512).unsqueeze(0).repeat(2, 1),
        },
        non_tensor_batch={
            "original_questions": ["What is 2 + 2?", "What is 3 + 3?"]
        },
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": True,
            "validate": False
        }
    )
    
    print("✅ Multi-prompt rollout test completed successfully!")


if __name__ == "__main__":
    test_multi_prompt_rollout() 
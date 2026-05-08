#!/usr/bin/env python3
import sys
import argparse
import asyncio

# 1) 先打补丁（必须在 import vllm 之前）
import turboquant.adapter.vllm_adapter as tq

tq.enable_no_alloc(
    key_bits=4,
    value_bits=3,
    buffer_size=128,
    initial_layers_count=32
)

print("[TurboQuant] patch loaded", flush=True)

# 2) 引入 vLLM 的参数解析与启动逻辑
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.openai.api_server import run_server


def main():
    parser = argparse.ArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # 强制 eager 模式，避免 CUDA Graph 与 TQ no_alloc hooks 冲突
    args.enforce_eager = True

    validate_parsed_serve_args(args)
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen_tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-29
# * Version     : 1.0.092922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from utils.llm.reasoning_from_scratch.qwen3 import download_qwen3_small
from utils.llm.reasoning_from_scratch.qwen3 import Qwen3Tokenizer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# tokenizer model path
tokenizer_model_dir = "./downloaded_models/qwen3_model"
tokenizer_model_path = Path(tokenizer_model_dir).joinpath("tokenizer-base.json")

# tokenizer model download
if not tokenizer_model_path.exists():
    download_qwen3_small(
        kind="base", 
        tokenizer_only=True, 
        out_dir=tokenizer_model_dir,
    )

# tokenizer
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_model_path)




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    prompt = "Explain large language models."

    input_token_ids_list = tokenizer.encode(prompt)
    logger.info(f"input_token_ids_list: {input_token_ids_list}") 

    text = tokenizer.decode(input_token_ids_list)
    logger.info(f"text: {text}")

    for i in input_token_ids_list:
        logger.info(f"{i} -> {tokenizer.decode([i])}")

if __name__ == "__main__":
    main()

import json
import os
# training_env = json.loads(os.environ["SM_TRAINING_ENV"])
# os.environ["MASTER_ADDR"] = training_env["master_addr"]
# os.environ["MASTER_PORT"] = training_env["master_port"]

import os
import sys

import pickle
import random
import time
import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import torch
import os
import wandb
import torch.distributed as dist

import numpy as np
from sklearn.metrics import accuracy_score
import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import datasets
import deepspeed

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class DataTrainingArguments:
    save_hf_model_name : str = field(
        default="WhipParty/KRX_1130",
        metadata={"help": "The name of the model to save"}
    )
    wandb_project: str = field(
        default="krx-gemma2-fft-v1",
        metadata={"help": "The name of the project"}
    )
    wandb_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token for wandb"}
    )
    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token for huggingface"}
    )
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset"}
    )
    is_debug: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run in debug mode"}
    )
    deepspeed_rank: Optional[int] = field(
        default=0,
        metadata={"help": "The rank of the deepspeed"}
    )
    chat_template: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use chat template"}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The model id"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "Set this flag if you are using an uncased model."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use a fast tokenizer (backed by the tokenizers library) or not."}
    )
    
    
def initialize_seed(seed=42):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.is_debug:
        training_args.eval_steps = 1
        training_args.logging_steps = 1
    
    wandb.login(key=data_args.wandb_token)
    os.system(f"huggingface-cli login --token={data_args.hf_token}")
    initialize_seed(training_args.seed)
    
    def is_main_process():
        return not dist.is_initialized() or dist.get_rank() == 0
    
    if is_main_process():
        wandb.init(
            project=data_args.wandb_project, 
            dir=training_args.output_dir, 
        )
        
        output = f"{time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())}"
        training_args.report_to = "wandb"
        training_args.output_dir = os.path.abspath(os.path.join(training_args.output_dir, output))
        training_args.run_name = output
        wandb.run.name = output
    else:
        training_args.report_to = "none"
    
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    dataset = datasets.load_dataset(data_args.dataset_name)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    if data_args.is_debug:
        train_dataset = train_dataset.select(range(100))
        test_dataset = test_dataset.select(range(100))
        
    print(train_dataset)
    print(test_dataset)
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir if model_args.cache_dir is not None else None,
        padding_side="right"
    )
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )
    
    
    if data_args.chat_template:
        def preprocess_function(examples):
            q = examples['prompt'].replace("### 질문:", "").replace("\n### 정답:", "").strip()
            a = examples['response'].strip()
            messages = [
                {"role": "user", "content": q},
                {"role": "model", "content": a  + tokenizer.eos_token},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token
            return {'text': text}
    else:
        def preprocess_function(examples):
            text = examples['prompt'] + "\n" + examples['response'] + tokenizer.eos_token
            return {'text': text}
        
        
    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples), batched=False)
    test_dataset = test_dataset.map(lambda examples: preprocess_function(examples), batched=False)
    train_dataset.remove_columns(remove_columns)
    test_dataset.remove_columns(remove_columns)

    if data_args.chat_template:
        response_template = "<start_of_turn>model\n"
    else:
        response_template = "### 정답:"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    
    example_text = train_dataset[0]['text']
    tokenized_example = tokenizer(example_text, return_tensors="pt")
    tokenized_batch = [{"input_ids": tokenized_example["input_ids"].squeeze(0)}]
    processed_batch = data_collator(tokenized_batch)
    print("Processed Input IDs:", processed_batch["input_ids"])
    print("Processed Labels:", processed_batch["labels"])
    print("Decoded Input Text:", tokenizer.decode(processed_batch["input_ids"][0]))
    print("Decoded Label Text:", tokenizer.decode([x for x in processed_batch["labels"][0] if x != -100]))
    

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
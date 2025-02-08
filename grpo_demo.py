# train_grpo.py
import re
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trainer import GRPOTrainer_new

import wandb
from accelerate import Accelerator

accelerator = Accelerator()

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    if split == "train":
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
                #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
                #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                #    answer="7"
                #)},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
    
    return data # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/Llama-1B-GRPO")
    parser.add_argument("--data", type=str, default="openai/gsm8k")
    parser.add_argument("--wandb_entity", type=str, default="zeng0176")
    parser.add_argument("--wandb_project", type=str, default="R1-Experiment")
    parser.add_argument("--trainer", type=str, default="grpo")
    
    # trainer parameters
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6)
    parser.add_argument("--num_generations", type=int, default=12)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=786)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    # parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--report_to", type=str, default="wandb")
    
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    
    if accelerator.is_main_process:
        print("*" * 40)
        print(f"Starting training with the arguments")
        for k, v in vars(args).items():
            print(f"{k:30} {v}")
        print("*" * 40)
    
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    run_name = args.output_dir.split("/")[-1] + str(args.lr) + "-bs-" + str(args.per_device_train_batch_size)
        
    if args.report_to == "wandb" and accelerator.is_main_process:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
    
    dataset = get_gsm8k_questions()
    # test_dataset = get_gsm8k_questions(split="test")
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        learning_rate=args.lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=1,
        eval_strategy="no",
        # eval_steps=args.eval_steps,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
    )
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=64,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    #     task_type="CAUSAL_LM",
    #     lora_dropout=0.05,
    # )
    
    # if "Llama" in args.model_name:
    #     model=args.model_name
    # elif "Qwen" in args.model_name:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # use peft at your own risk; not working for me with multi-GPU training
    if args.trainer.lower() == "grpo":
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func],
            args=training_args,
            train_dataset=dataset,
            # eval_dataset=test_dataset,
            # peft_config=peft_config
        )
    elif args.trainer.lower() == "grpo_new":
        # for the new trainer, always keep the correctness reward the first
        trainer = GRPOTrainer_new(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                correctness_reward_func,
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func],
            args=training_args,
            train_dataset=dataset,
            # eval_dataset=test_dataset,
            # peft_config=peft_config
        )
    
    trainer.train()
    
    # TODO: add a evaluation 
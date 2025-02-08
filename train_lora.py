import os
import gc
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, logging)
from utils.get_data import get_data
from utils.args import parse_args
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")

    compute_dtype = getattr(torch, "float16")

    train_data, eval_data, _, _ = get_data(args.filename)

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias=args.bias,
        target_modules=args.target_modules,
        task_type=args.task_type,
    )

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        optim=args.optim,
        save_steps=0,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=True,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy=args.evaluation_strategy
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
    ).to(device)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    del [model, tokenizer, peft_config, trainer, train_data, eval_data, bnb_config, training_arguments]
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.finetuned_model,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merged_model_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.merged_model_dir)


if __name__ == "__main__":
    main()

import os
from utils.args import parse_args
import torch
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          logging)
from utils.predict import predict
from utils.evaluate import evaluate
from utils.get_data import get_data
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

    _, _, test_data, y_gt = get_data(args.filename)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        temperature=args.temperature
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, tokenizer = setup_chat_format(model, tokenizer)

    y_pred = predict(test_data, model, tokenizer)
    evaluate(y_gt, y_pred)


if __name__ == "__main__":
    main()

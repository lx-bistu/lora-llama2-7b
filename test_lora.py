import os
from utils.evaluate import evaluate
from utils.predict import predict
from utils.get_data import get_data
from utils.args import parse_args
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          logging)
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    _, _, test_data, y_gt = get_data()
    compute_dtype = getattr(torch, "float16")

    merged_model = AutoModelForCausalLM.from_pretrained(
        args.merged_model_dir,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.merged_model_dir)


    y_pred = predict(test_data, merged_model, tokenizer)
    evaluate(y_gt, y_pred)

if __name__ == '__main__':
    main()
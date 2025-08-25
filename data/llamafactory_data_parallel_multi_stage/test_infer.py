import json
from pathlib import Path

import torch
import torch_npu
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NPU_DEVICE = 'npu:0'

device = torch.device(NPU_DEVICE)

def read_json(json_file: str) -> dict | list:
    print(f'=== Loading data from {json_file} ===')
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    print('=== Loading finished ===')
    return data

def load_model_and_tokenizer(model_path: str):
    print(f'=== Loading model from {model_path} ===')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=NPU_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=NPU_DEVICE)
    print('=== Successfully loaded model ===')
    return model, tokenizer

def run_model(model, tokenizer, input_text: str, max_new_tokens: int = 2048, temperature: float = .0) -> str:
    inputs = tokenizer(input_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text[len(input_text):].strip()

    return output_text

def test(model_path: str, eval_path: str, output_dir: str):
    output_path = output_dir + '/' + Path(model_path).name + '.jsonl'

    with open(output_path, 'w', encoding='utf-8') as _:
        pass

    eval_data = read_json(eval_path)
    model, tokenizer = load_model_and_tokenizer(model_path)
    em_count = 0
    with open(output_path, 'a', encoding='utf-8') as f:
        for data in tqdm(eval_data, total=len(eval_data)):
            input_text = data['instruction']
            output_text = run_model(model, tokenizer, input_text)

            em = data['output'].strip() == output_text.strip()
            if em:
                em_count += 1

            obj = {
                'instruction': data['instruction'],
                'input': data['input'],
                'ground_truth': data['output'],
                'model_output': output_text,
                'em': em
            }

            f.write(json.dumps(obj) + '\n')

    print(f'EM: {em_count}/{len(eval_data)} ({em_count / len(eval_data) * 100:.2f}%)')

def main():
    output_dir = '/data1/gsn/CustomTrain/data/gsn_data/data_formatting/test_results'

    # full zeta
    model_path = '/data1/gsn/CustomTrain/output/data_formatting/7b_full_zeta'
    eval_path = '/data1/gsn/CustomTrain/data/gsn_data/data_formatting/zeta/eval.json'
    test(model_path, eval_path, output_dir)

    # full line diff
    model_path = '/data1/gsn/CustomTrain/output/data_formatting/7b_full_line_diff'
    eval_path = '/data1/gsn/CustomTrain/data/gsn_data/data_formatting/line_diff/eval.json'
    test(model_path, eval_path, output_dir)

    # 20% zeta & 80% line diff
    model_path = '/data1/gsn/CustomTrain/output/data_formatting/7b_20zeta_80line_diff'
    eval_path = '/data1/gsn/CustomTrain/data/gsn_data/data_formatting/line_diff/eval.json'
    test(model_path, eval_path, output_dir)

if __name__ == '__main__':
    main()

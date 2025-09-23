import os
import argparse
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

# SYSTEM_PROMPT = {
#     'qwen': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
#     'phi4': 'You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions.'
# }

def run_eval():
    parser = argparse.ArgumentParser(description='Run an evaluation task using Opencompass Beckend')
    parser.add_argument('model-name', '-m', type=str, default='qwen', help='The model name in vllm')
    parser.add_argument('dataset-name', '-d', type=str, default='mbpp', help='The dataset name')
    parser.add_argument('max-new-tokens', type=int, help='The maximum number of new tokens to generate')
    parser.add_argument('port', type=str, help='The port number for the service')
    parser.add_argument('output-path', type=str, help='The path to save output files')
    args = parser.parse_args()

    api_meta_template = dict(
       round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True)
       ]
    )

    task_cfg = task_cfg_dict = dict(
                                eval_backend='OpenCompass',
                                eval_config={
                                    'datasets': [args.dataset_name],
                                    'models': [
                                        {'path': args.model_name, 
                                        'openai_api_base': f"http://localhost:{args.port}/v1/chat/completions", 
                                        'is_chat': True,
                                        'meta_template': api_meta_template,
                                        'batch_size': 32,
                                        'max_out_len': args.max_new_tokens,
                                        'run_cfg': {"num_gpus": 1}
                                        },
                                    ],
                                    'work_dir': args.output_path
                                    },
                                )
    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()

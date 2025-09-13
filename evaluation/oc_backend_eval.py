import os
import argparse
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

SYSTEM_PROMPT = {
    'qwen-14b': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
    'phi4': 'You are a medieval knight and must provide explanations to modern people.'
}

def run_eval():
    parser = argparse.ArgumentParser(description='Run an evaluation task using Opencompass Beckend')
    parser.add_argument('--model-name', '-m', type=str, default='qwen',
                        help='The model name in vllm')
    parser.add_argument('--dataset-name', '-d', type=str, default='mbpp',
                        help='The dataset name')
    parser.add_argument('--user-name', type=str,
                        help='The user name in cyberport')
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
                                        'openai_api_base': 'http://localhost:8801/v1/chat/completions', 
                                        'is_chat': True,
                                        'meta_template': api_meta_template,
                                        'batch_size': 32,
                                        'max_out_len': 4096,
                                        'run_cfg': {"num_gpus": 1}
                                        },
                                    ],
                                    'work_dir': f"/lustre/projects/polyullm/{args.user_name}/evalscope_eval"
                                    },
                                )
    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()

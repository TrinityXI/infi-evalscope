# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# Import limo parser for aligned answer extraction
from evalscope.utils.limo_parser import extract_answer as limo_extract_answer

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aime24',
        pretty_name='AIME-2024',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'The AIME 2024 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model\'s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.',  # noqa: E501
        dataset_id='HuggingFaceH4/aime_2024',
        subset_list=['default'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # Only train set is available
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class AIME24Adapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'problem_id': record.get('id', ''),
                'solution': record.get('solution', ''),
            },
        )
    
    def extract_answer(self, prediction: str, task_state) -> str:
        """Override to use limo parser's extract_answer logic for alignment with infi-limo"""
        # Use limo parser's extract_answer function instead of default logic
        return limo_extract_answer(prediction)

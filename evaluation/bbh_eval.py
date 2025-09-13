import json
import re
import os
from pathlib import Path
import logging
from typing import Dict, Any
import argparse
from datetime import datetime

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging with a timestamped log file in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"bbh_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

def extract_answer_multistep_arithmetic_two(original_prediction: str, references: str) -> str:
    """For multistep_arithmetic_two, return the original prediction as is."""
    return original_prediction.strip()

def extract_answer_word_sorting(original_prediction: str) -> str:
    """Extract the last line after ':' and remove only commas and periods for word_sorting."""
    try:
        last_line = original_prediction.split("\n")[-1].strip()
        if ":" in last_line:
            answer = last_line.split(":", 1)[-1].strip()
            # Remove only commas and periods
            answer = re.sub(r'[,.]', '', answer)
            return answer.strip()
        return last_line
    except:
        return original_prediction.strip()

def extract_answer_formal_fallacies(original_prediction: str) -> str:
    """Extract the last line and check for 'valid' or 'invalid' for formal_fallacies."""
    try:
        last_line = original_prediction.split("\n")[-1].strip().lower()
        if "invalid" in last_line:
            return "invalid"
        elif "valid" in last_line:
            return "valid"
        return "not sure"
    except:
        return "not sure"

def extract_answer_causal_judgement(original_prediction: str) -> str:
    """Extract the last line and check for 'Yes' or 'No' for causal_judgement."""
    try:
        last_line = original_prediction.split("\n")[-1].strip().lower()
        if "yes" in last_line:
            return "Yes"
        elif "no" in last_line:
            return "No"
        return last_line
    except:
        return original_prediction.strip()

def extract_answer_boolean_expressions(original_prediction: str) -> str:
    """Extract the last line and find the last 'True' or 'False' for boolean_expressions."""
    try:
        last_line = original_prediction.split("\n")[-1].strip()
        # Search from the end for True or False
        words = last_line.split()
        for word in reversed(words):
            if word in ["True", "False"]:
                return word
        return last_line
    except:
        return original_prediction.strip()

def extract_answer_hyperbaton(original_prediction: str) -> str:
    """Extract the last line and match '(A)', '(B)', etc. for hyperbaton."""
    try:
        last_line = original_prediction.split("\n")[-1].strip()
        match = re.search(r'(\([A-D]\))', last_line)
        if match:
            return match.group(1).upper()
        return last_line
    except:
        return original_prediction.strip()

def extract_answer_sports_understanding(original_prediction: str, predictions: str) -> str:
    """For sports_understanding, standardize to 'Yes' or 'No' based on predictions or original_prediction."""
    try:
        # Check if predictions contains 'yes' or 'no'
        predictions_lower = predictions.lower()
        if "yes" in predictions_lower:
            return "Yes"
        elif "no" in predictions_lower:
            return "No"
        
        # If neither, check the last line of original_prediction
        last_line = original_prediction.split("\n")[-1].strip().lower()
        if "is not plausible" in last_line:
            return "No"
        elif "is plausible" in last_line:
            return "Yes"
        return predictions.strip()
    except:
        return predictions.strip()

def direct_match(prediction: str, reference: str) -> bool:
    """Perform a direct string match, normalizing for minor differences."""
    normalized_prediction = re.sub(r'[.!?]+$', '', prediction).strip().lower()
    normalized_reference = re.sub(r'[.!?]+$', '', reference).strip().lower()
    return normalized_prediction == normalized_reference

def re_evaluate_entry(entry: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Re-evaluate an entry based on task-specific rules."""
    original_prediction = entry['choices'][0]['message']['content']
    predictions = entry['choices'][0]['message']['content']
    references = entry['raw_input']['target']
    
    # Extract new prediction based on task
    if task == "bbh_multistep_arithmetic_two":
        new_prediction = extract_answer_multistep_arithmetic_two(original_prediction, references)
        is_correct = references.strip() in new_prediction
    elif task == "bbh_word_sorting":
        new_prediction = extract_answer_word_sorting(original_prediction)
        is_correct = direct_match(new_prediction, references)
    elif task == "bbh_formal_fallacies":
        new_prediction = extract_answer_formal_fallacies(original_prediction)
        is_correct = direct_match(new_prediction, references)
    elif task == "bbh_causal_judgement":
        new_prediction = extract_answer_causal_judgement(original_prediction)
        is_correct = direct_match(new_prediction, references)
    elif task == "bbh_boolean_expressions":
        new_prediction = extract_answer_boolean_expressions(original_prediction)
        is_correct = direct_match(new_prediction, references)
    elif task == "bbh_hyperbaton":
        new_prediction = extract_answer_hyperbaton(original_prediction)
        is_correct = direct_match(new_prediction, references)
    elif task == "bbh_sports_understanding":
        new_prediction = extract_answer_sports_understanding(original_prediction, predictions)
        is_correct = direct_match(new_prediction, references)
    else:
        new_prediction = predictions
        is_correct = entry.get("correct", False)
    
    return {
        "original_prediction": original_prediction,
        "predictions": new_prediction,
        "references": references,
        "correct": is_correct
    }

def process_json_file(json_path: str, task: str, score_json_path: str, re_evaluate_tasks: list, logger: logging.Logger) -> tuple:
    """Process a single JSON file and return total, correct counts, updated data, and original score."""
    with open(score_json_path, 'r', encoding='utf-8') as f:
        bbh_scores = json.load(f)
    bbh_scores = bbh_scores['metrics'][0]['categories'][0]['subsets']
    for i in range(len(bbh_scores)):
        if task.replace('bbh_', '') == bbh_scores[i]['name']:
            task_score = bbh_scores[i]
    
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return 0, 0, {}, 0.0
    
    data = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        logger.info(f"Successfully loaded JSON: {json_path}")
    except Exception as e:
        logger.error(f"Failed to load JSON {json_path}: {str(e)}")
        return 0, 0, {}, 0.0
    
    total_count = 0
    correct_count = 0
    updated_details = {}
    original_score = task_score.get("score", 0.0) * 100
    
    # details = data.get("details", {})
    for key, entry in enumerate(data):
        total_count += 1
        updated_entry = entry.copy()
        
        # Re-evaluate if task is in re_evaluate_tasks and correct is False
        if task in re_evaluate_tasks:
            updated_entry = re_evaluate_entry(entry, task)
        else:
            # Use existing correct value
            updated_entry["correct"] = entry.get("correct", False)
        
        if updated_entry["correct"]:
            correct_count += 1
        
        updated_details[key] = updated_entry
    
    # Calculate new score
    new_score = (correct_count / total_count * 100) if total_count > 0 and task in re_evaluate_tasks else original_score
    correct_count = original_score * total_count / 100 if task not in re_evaluate_tasks else correct_count

    return total_count, correct_count, {"score": new_score, "details": updated_details}, original_score

def evaluate_bbh(json_dir: str, model_name: str, output_dir: str, re_evaluate_tasks: list, logger: logging.Logger) -> None:
    """Evaluate all BBH JSON files, compute accuracy, and compare performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_questions = 0
    total_correct = 0
    performance_changes = []
    
    # Process all JSON files in the directory
    json_dir_pred = os.path.join(json_dir, f"predictions/{model_name}")
    for json_file in os.listdir(json_dir_pred):
        if not json_file.endswith(".jsonl") or not json_file.startswith("bbh_"):
            continue
        
        json_path = os.path.join(json_dir_pred, json_file)
        task = json_file.replace(".jsonl", "")

        score_json_path = os.path.join(json_dir, f"reports/{model_name}/bbh.json")
        
        logger.info(f"Processing task: {task}")
        task_total, task_correct, updated_data, original_score = process_json_file(json_path, task, score_json_path, re_evaluate_tasks, logger)
        
        if task_total > 0:
            # Save updated JSON
            output_path = os.path.join(output_dir, json_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved updated JSON to: {output_path}")
            
            total_questions += task_total
            total_correct += task_correct
            task_accuracy = task_correct / task_total if task_total > 0 else 0
            new_score = updated_data["score"]
            score_change = new_score - original_score
            
            performance_changes.append({
                "task": task,
                "original_score": original_score,
                "new_score": new_score,
                "change": score_change
            })
            
            logger.info(f"Task {task}: {task_correct}/{task_total} correct, Original Score: {original_score:.2f}%, New Score: {new_score:.2f}%, Change: {score_change:+.2f}%")
    
    # Compute overall accuracy
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    logger.info(f"Overall: {total_correct}/{total_questions} correct, Accuracy: {overall_accuracy:.2%}")
    
    # Log performance comparison, sorted by change descending
    logger.info("\nPerformance Comparison:")
    logger.info(f"{'Task':<30} | {'Original Score':<15} | {'New Score':<12} | {'Change':<10}")
    logger.info("-" * 70)
    sorted_changes = sorted(performance_changes, key=lambda x: x['change'], reverse=True)
    for change in sorted_changes:
        logger.info(f"{change['task']:<30} | {change['original_score']:>12.2f}% | {change['new_score']:>9.2f}% | {change['change']:>+7.2f}%")
    logger.info("-" * 70)

def find_latest_timestamp_dir(base_dir):
    """
    在指定目录中查找时间戳最大的文件夹，并返回完整路径
    
    Args:
        base_dir (str): 要搜索的目录路径
        
    Returns:
        str: 包含最新时间戳文件夹的完整路径，如果没有找到则返回None
    """
    # 确保目录存在
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        raise ValueError(f"目录不存在或不是有效目录: {base_dir}")
    
    # 获取所有子目录
    all_items = os.listdir(base_dir)
    directories = [item for item in all_items if os.path.isdir(os.path.join(base_dir, item))]

    # 提取时间戳并找到最大的
    timestamp_dirs = []
    for dir_name in directories:
        # 使用正则表达式提取数字时间戳
        match = re.search(r'(\d+)', dir_name)
        if match:
            timestamp = int(match.group(1))
            timestamp_dirs.append((timestamp, dir_name))
    
    # 如果没有找到时间戳文件夹，返回None
    if not timestamp_dirs:
        return None
    
    # 找到时间戳最大的文件夹
    latest_timestamp, latest_dir_name = max(timestamp_dirs, key=lambda x: x[1])
    print(latest_dir_name)
    # 返回完整路径
    return os.path.join(base_dir, latest_dir_name)

def main():
    parser = argparse.ArgumentParser(description="Evaluate BBH JSON files with task-specific re-evaluation.")
    parser.add_argument(
        "--json-dir",
        type=str,
        default="bbh_data/",
        help="Directory containing BBH JSON files"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen",
    )
    
    args = parser.parse_args()
    args.json_dir = find_latest_timestamp_dir(args.json_dir)
    # List of tasks with potential evaluation issues (modify as needed)
    re_evaluate_tasks = [
        "bbh_word_sorting",
        "bbh_causal_judgement",
        "bbh_multistep_arithmetic_two",
        "bbh_formal_fallacies",
        "bbh_boolean_expressions",
        "bbh_hyperbaton",
        "bbh_sports_understanding"
    ]
    cur_path = os.getcwd()
    json_dir_base = os.path.basename(args.json_dir)
    if json_dir_base == "": # "bbh_json/"
        if args.json_dir[-1] == "/":
            output_dir_prefix = args.json_dir[:-1]
    else:
        output_dir_prefix = json_dir_base

    output_dir = os.path.join(cur_path, output_dir_prefix+"_updated") 
    logger = setup_logging(output_dir)
    logger.info(f"Tasks to re-evaluate: {re_evaluate_tasks}")
    
    evaluate_bbh(
        json_dir=args.json_dir,
        model_name=args.model_name,
        output_dir=output_dir,
        re_evaluate_tasks=re_evaluate_tasks,
        logger=logger
    )

if __name__ == "__main__":
    main()

#%%
import numpy as np
from PIL import Image
import logging
from collections import defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import utils
import base64
from evaluation.metric import calculate_iou, calculate_MAE


logger = logging.getLogger(__name__)

def get_iou_score_report(ground_truth_path: str, submitted_answer_path: str, ground_truth_root: str, submitted_answer_root: str) -> dict:
    raw_metric_val = calculate_iou(ground_truth_path, submitted_answer_path, ground_truth_root, submitted_answer_root)

    # temp evaluation score formulation logic:
    # Max: 100
    # Min: 0
    # Score: raw_metric_val * 100

    score_report_dict = {
        "raw_metric_val": raw_metric_val,
        "score_max": 100,
        "score_min": 0,
        "score_val": raw_metric_val * 100 if raw_metric_val * 100 >= 0 else 0
    }

    return score_report_dict

def get_MAE_score_report(ground_truth, submitted_answer) -> dict:
    raw_metric_val = calculate_MAE(ground_truth, submitted_answer)

    # temp evaluation score formulation logic:
    # Max: 100
    # Min: 0
    # Score: Requires a baseline to compare with. Now reporting the raw value.

    score_report_dict = {
        "raw_metric_val": raw_metric_val,
        "score_max": 100,
        "score_min": 0,
        "score_val": (1 - raw_metric_val) * 100 if (1 - raw_metric_val) * 100 >= 0 else 0
    }

    return score_report_dict
# %%

metric_function_map = {
    'IOU': get_iou_score_report,
    'MAE': get_MAE_score_report
}

def get_metric_function(solution_dict: dict):
    metric_func = metric_function_map.get(solution_dict['metric'])
    if metric_func is None:
        logger.error(f"[get_metric_function] {solution_dict['metric']} is not a supported metric type.")
        return None
    else:
        logger.info(f"{solution_dict['metric']} is used as metric")
        return metric_func
    

def is_valid_path(path_string: str):
    # Check if the path exists
    if os.path.exists(path):
        return True
    else:
        return False

def get_submitted_result(id: str, result_df: pd.DataFrame) -> pd.DataFrame:
    # submitted_result_path = './output/result.csv'   # consider making this path a config
    if id in result_df['id'].unique():
        result = result_df[result_df['id'] == id]['result'].tolist()[0]
    else:
        result = "Fail"
    return result

def compile_task_scores(student_submission: pd.DataFrame, ta_solution: pd.DataFrame, ground_truth_root: str, submitted_answer_root: str) -> pd.DataFrame:
    updated_ta_solution_list = list()

    for index, row in ta_solution.iterrows():
        solution_dict = row.to_dict()
        task_id = solution_dict['id']
        metric_func = get_metric_function(solution_dict)
        ground_truth = solution_dict['ground_truth']
        submitted_answer = get_submitted_result(task_id, student_submission)

        if metric_func is not None:
            try:
                if solution_dict['metric'] =='IOU':
                    score_report_dict = metric_func(ground_truth, submitted_answer, ground_truth_root, submitted_answer_root)
                else:
                    score_report_dict = metric_func(ground_truth, submitted_answer)
                solution_dict['raw_metric_val'] = score_report_dict['raw_metric_val']
                solution_dict['score'] = score_report_dict['score_val']
            except Exception as e:
                solution_dict['raw_metric_val'] = 0
                solution_dict['score'] = 0
                print('Can not get image path or other issuses happen.')
                
            updated_ta_solution_list.append(solution_dict)
        else:
            logger.warning(f"task id {task_id} does not match existing metric function")
            continue

    updated_ta_solution_df = pd.DataFrame(updated_ta_solution_list)
    updated_ta_solution_df['score'] = pd.to_numeric(updated_ta_solution_df['score'], errors='coerce')
    return updated_ta_solution_df

def create_radar_chart(student_submission: pd.DataFrame, ta_solution: pd.DataFrame, output_path="./output", title="Radar Chart", ground_truth_root='.', submitted_answer_root='.', lower_limit=None, upper_limit=None):
    """Creates the radar chart and save to the local directory. Returns the file name and the dataframe containing the students' score
    alongside the other relevant task information provided by the ta_solution dataframe.

    Args:
        student_submission (pd.DataFrame): _description_
        ta_solution (pd.DataFrame): _description_
        title (str, optional): _description_. Defaults to "Radar Chart".
        lower_limit (_type_, optional): _description_. Defaults to None.
        upper_limit (_type_, optional): _description_. Defaults to None.

    Returns:
        str: file name of the radar chart (in png format)
        pd.DataFrame: the dataframe containing the scores and other relevant task information
    """
    updated_ta_solution_df = compile_task_scores(student_submission, ta_solution, ground_truth_root, submitted_answer_root)

    radar_chart_path, updated_ta_solution_df = draw_radar_chart_from_updated_ta_solution(updated_ta_solution_df, output_path=output_path, title=title, lower_limit=lower_limit, upper_limit=upper_limit)
    total_score = round(updated_ta_solution_df['score'].mean(),2)
    print(f'Total Score: {total_score}')
    return radar_chart_path, updated_ta_solution_df, total_score

def draw_radar_chart_from_updated_ta_solution(updated_ta_solution_df:pd.DataFrame, output_path="./output", title="Radar Chart", lower_limit=None, upper_limit=None):
    task_count_dict = updated_ta_solution_df['task'].value_counts().to_dict()
    task_total_score_dict = updated_ta_solution_df.groupby("task")["score"].sum().to_dict()
    categories = [cat for cat in task_count_dict.keys()]
    num_vars = len(categories)

    # find the average score
    average_scores = dict()
    for cat in categories:
        average_scores[cat] = task_total_score_dict[cat] / task_count_dict[cat]

    # find the angles for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # average_scores to plot
    average_scores_to_plot = [avg for avg in average_scores.values()]
    average_scores_to_plot += average_scores_to_plot[:1]

    # initialize radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # draw one axis per variable
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # set y-axis limits
    if lower_limit is None:
        lower_limit = min(average_scores_to_plot) - 1   # Set a buffer below the minimum score
    if upper_limit is None:
        upper_limit = max(average_scores_to_plot) + 1   # Set a buffer above the maximum score

    ax.set_yticks(np.linspace(lower_limit, upper_limit, 5))
    ax.set_ylim(lower_limit, upper_limit)

    # plot data
    ax.plot(angles, average_scores_to_plot, linewidth=1, linestyle='solid')

    # fill area
    ax.fill(angles, average_scores_to_plot, 'b', alpha=0.1)

    # add title
    ax.set_title(title, size=20, color='blue', y=1.1)

    fig.tight_layout()

    # save radar chart
    ax.figure.savefig(f"{output_path}/{title}.png")
    updated_ta_solution_df.to_csv(f"{output_path}/{title}.csv")

    return f"{title}.png", updated_ta_solution_df
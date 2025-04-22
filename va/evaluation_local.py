import pandas as pd

from evaluation.evaluation import create_radar_chart

student_submission = pd.read_csv('../output/result.csv')
# student_submission = pd.read_csv('../output/fake_result.csv')

solution = pd.read_csv('../data/release_private_set.csv')
# solution = pd.read_csv('../data/private_set(TA).csv')
# solution = pd.read_csv('../data/private_service_set(TA).csv')

create_radar_chart(student_submission, solution, output_path='../output', title="radar_chart", lower_limit=0, upper_limit=100, ground_truth_root='../', submitted_answer_root='./')
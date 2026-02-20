from factors_config import get_design_points_robot, is_valid_point
import torch

task_name = 'pickblueblock'


all_points = get_design_points_robot()

valid_points_list = [p for p in all_points if is_valid_point(p, task_name=task_name)]
if not valid_points_list:
    print(f"Error: No valid points found in the design space.")
    print("Check your 'is_valid_point' function and task-specific constraints.")
    exit()
points = torch.stack(valid_points_list)

print(points[0:points.shape[0]//3, :3])
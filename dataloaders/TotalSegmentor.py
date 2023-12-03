from totalsegmentator.python_api import totalsegmentator
import os

if __name__ == "__main__":
    root_path = '/data/private/autoPET/Task1/pelvis/'
    output_root_path = '/data/private/autoPET/Task1/ct_label'
    people_name = os.listdir(root_path)
    for item in people_name:
        if item != 'overview':
            input_path = os.path.join(root_path, item, 'ct.nii.gz')
            output_path = os.path.join(output_root_path, item)
            os.makedirs(output_path, exist_ok=True)
            totalsegmentator(input_path, output_path)


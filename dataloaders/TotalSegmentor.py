from totalsegmentator.python_api import totalsegmentator

if __name__ == "__main__":
    input_path = '/data/private/autoPET/Task1/pelvis/1PA001/ct.nii.gz'
    output_path = '/data/private/autoPET/Task1/ct_label'
    totalsegmentator(input_path, output_path)
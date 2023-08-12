from raop.utils.find_root_path import find_project_root

import os
import subprocess
import re


def get_project_metrics(project_path, radon_exe_path, exclude_dirs=None):
    try:
        # Get a list of all Python files in the project directory
        python_files = []
        for root, dirs, files in os.walk(project_path):
            if exclude_dirs:
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
            python_files.extend([os.path.join(root, file) for file in files if file.endswith(".py")])

        # Initialize variables to store aggregated metrics
        total_lines_of_code = 0
        total_logical_lines_of_code = 0

        # Loop through each Python file and calculate metrics
        for file in python_files:
            filepath = os.path.join(project_path, file)

            # Get raw metrics using "radon raw"
            raw_output = subprocess.run([radon_exe_path, "raw", filepath], capture_output=True, text=True)
            raw_metrics = raw_output.stdout.strip().split()

            # Extract lines of code and complexity from raw metrics
            lines_of_code = int(raw_metrics[2])
            logical_lines_of_code = int(raw_metrics[4])

            # Add to the total metrics
            total_lines_of_code += lines_of_code
            total_logical_lines_of_code += logical_lines_of_code

        # Get the total number of Python files in the project
        total_python_files = len(python_files)

        # Calculate average complexity
        cmd = [radon_exe_path, "cc", "--total", project_path + "/raop"]
        cc_output = subprocess.run(cmd, capture_output=True, text=True)
        cc_lines = cc_output.stdout.strip().split("\n")
        avg_cyclomatic_complexity = float(re.findall(pattern=r"\((.*?)\)", string=cc_lines[-2])[0])

        # Return the aggregated metrics
        return {
            "Total Python Files": total_python_files,
            "Total Lines of Code": total_lines_of_code,
            "Total Logical Lines of Code": total_logical_lines_of_code,
            "Lib Average Cyclomatic Complexity": avg_cyclomatic_complexity,
        }

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    proj_path = find_project_root()
    radon_exe = proj_path+ "/venv/Scripts/radon.exe"
    exclude_directories = ["venv", "test", "data", "docs"]  # Add directories to exclude, if any
    project_metrics = get_project_metrics(proj_path, radon_exe, exclude_dirs=exclude_directories)
    for key, val in project_metrics.items():
        print(f"{key}: {val}")

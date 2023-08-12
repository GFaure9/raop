import os


def find_project_root(marker_file="README.md"):
    current_dir = os.path.abspath(os.getcwd())

    while True:
        # Check if the current directory contains the marker file
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir

        # Move up to the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        # Check if we have reached the root directory (no parent directory)
        if parent_dir == current_dir:
            break

        current_dir = parent_dir

    raise FileNotFoundError(f"Could not find the project root directory."
                            f" Make sure the marker file ({marker_file}) exists in your project.")

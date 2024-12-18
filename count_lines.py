import os

def count_lines_of_code(directory, extensions=['.py']):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
    return total_lines

if __name__ == "__main__":
    project_directory = 'e:/2 - 3_Technical_material/Grasp_learning/Grasp Failure Recovery'
    lines_of_code = count_lines_of_code(project_directory)
    print(f'Total lines of code: {lines_of_code}')
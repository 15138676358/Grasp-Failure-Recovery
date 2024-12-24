import os
import re

def count_lines_and_imports(directory, extensions=['.py']):
    total_lines = 0
    imports = set()
    import_pattern = re.compile(r'^\s*(import|from)\s+(\S+)')

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    for line in lines:
                        match = import_pattern.match(line)
                        if match:
                            imports.add(match.group(2).split('.')[0])

    return total_lines, imports

if __name__ == "__main__":
    project_directory = 'e:/2 - 3_Technical_material/Grasp_learning/Grasp Failure Recovery'
    lines_of_code, imports = count_lines_and_imports(project_directory)
    print(f'Total lines of code: {lines_of_code}')
    print(f'Imports: {", ".join(sorted(imports))}')
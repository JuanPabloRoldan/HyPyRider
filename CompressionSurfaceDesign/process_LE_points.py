import pandas as pd

def extract_points_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read all lines of the file
        lines = file.readlines()

    # Ignore the first 6 lines
    lines = lines[6:]

    # The next line contains the number of points
    num_points = int(lines[0].strip())

    # Extract the points, which are the next 'num_points' lines
    point_lines = lines[1:num_points + 1]
    
    # Parse the points into a pandas DataFrame
    data = [list(map(float, line.split()[0:3])) for line in point_lines]
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])

    return df

# Example usage
file_path = 'inputs/LeadingEdgeData_LeftSide.nmb'  # Replace with the actual file path
df = extract_points_from_file(file_path)
print(df)
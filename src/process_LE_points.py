import pandas as pd

def extract_points_from_file(file_path):
    '''
    Extracts 3D points from a formatted text file and returns them as a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the input file containing the points.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the 3D points with columns ['X', 'Y', 'Z'].
    '''
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
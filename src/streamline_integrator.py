import os
import random
import numpy as np
import pandas as pd
from conical_flow_analyzer import ConicalFlowAnalyzer
import process_LE_points

class StreamlineIntegrator:
    def __init__(self, gamma, M1, theta_s):
        """
        Initializes the streamline integrator.
        This includes creating an instance of the conical flow solver,
        as well as saving a data frame of useful solutions from the conical shock
        solutions from theta = [shock angle, cone angle].

        Parameters:
            gamma (float): Specific heat ratio for the fluid.
            M1 (float): Freestream Mach number upstream of the shock.
            theta_s (float): Shock wave angle in radians.
        """
        self.gamma = gamma
        self.M1 = M1
        self.theta_s = theta_s
        self.streamline_data = [] # store streamline points with ID

        # Create an instance of ConicalFlowAnalyzer
        self.conical_analyzer = ConicalFlowAnalyzer(M1, gamma)
        self.theta_c, _, _ = self.conical_analyzer.solve_taylor_maccoll(self.theta_s)

        # Tabulate post-shock flow properties from shock angle to the cone angle
        self.TM_tabulation = self.conical_analyzer.tabulate_tm_shock_to_cone(theta_s)

    def calculate_vector_angle(self, vector):
        """
        Calculates the angle between the given vector and the reference vector [1, 0, 0].

        Parameters:
            vector (list or np.array): A 3D vector [vx, vy, vz].

        Returns:
            float: Angle in degrees between the input vector and [1, 0, 0].
        """
        reference_vector = np.array([1, 0, 0])
        vector = np.array(vector)

        # Compute dot product and magnitudes
        dot_product = np.dot(vector, reference_vector)
        vector_magnitude = np.linalg.norm(vector)
        ref_magnitude = np.linalg.norm(reference_vector)

        # Avoid division by zero
        if vector_magnitude == 0:
            return None

        # Compute the angle in radians
        angle_rad = np.arccos(np.clip(dot_product / (vector_magnitude * ref_magnitude), -1.0, 1.0))

        # Convert to degrees
        return np.degrees(angle_rad)

    def trace_streamline(self, x, y, z, streamline_id):
        """
        Traces a streamline starting from the given leading-edge (LE) normalized coordinates.

        Parameters:
            x (float): Normalized x-coordinate of the LE point.
            y (float): Normalized y-coordinate of the LE point.
            z (float): Normalized z-coordinate of the LE point.
            streamline_id (int): Unique ID for this streamline.

        Returns:
            None (stores streamline points in self.streamline_data)
        """
        theta = self.theta_s
        streamline_points = []
        order = 0  # Tracks the number of points in a streamline

        # **Store the first point explicitly (the leading edge point)**
        first_point = [x * self.ref_length, y * self.ref_length, z * self.ref_length]
        streamline_points.append(first_point + [streamline_id, order])
        order += 1  # Increment order before stepping forward

        alpha = np.arctan(abs(z / y))
        prev_point = np.array(first_point)  # Store the first point as previous point

        while x < 1:
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            dt = 0.02

            # Interpolate V_r and V_theta from tabulated data
            print(f"Interpolating at Theta = {np.degrees(theta):.4f}")
            V_r = np.interp(theta, self.TM_tabulation['Theta (radians)'], self.TM_tabulation['V_r'])
            V_theta = np.interp(theta, self.TM_tabulation['Theta (radians)'], self.TM_tabulation['V_theta'])
            print(f"V_r = {V_r}, V_theta = {V_theta}")

            # Update theta and r
            d_theta = V_theta * dt / r
            theta += d_theta
            r += (V_r * dt)
            w = r * np.sin(theta)

            # Update coordinates
            x = r * np.cos(theta)
            y = -w * np.cos(alpha)
            z = w * np.sin(alpha)

            # Store new point
            new_point = np.array([x * self.ref_length, y * self.ref_length, z * self.ref_length])
            streamline_points.append(list(new_point) + [streamline_id, order])
            order += 1

            # Compute vector between the previous and new point
            vector = new_point - prev_point
            print(f'Vector between the two points\n{vector}')
            angle = self.calculate_vector_angle(vector)

            # Print or store the angle for later analysis
            print(f"Streamline {streamline_id}, Step {order}: Angle = {angle:.2f} degrees")

            # Update previous point
            prev_point = new_point

            # if np.isclose(theta, self.theta_c, rtol=0.01):
            #     break

        # Set last point x-coordinate to ref_length
        streamline_points[-1][0] = self.ref_length

        # Append streamline points to global data
        self.streamline_data.extend(streamline_points)

    def create_lower_surface(self):
        """
        Generates the lower surface by tracing streamlines from the leading-edge (LE) points.

        The function normalizes the LE points, traces each streamline, and prints the 
        trajectory to the console.

        Parameters:
            None

        Returns:
            None
        """

        # Find the maximum x-coordinate to normalize the lengths
        self.ref_length = self.LE_points['X'].max()

        # Iterate over all LE points
        for index, row in self.LE_points.iterrows():

            # Grab points and normalize lengths
            x, y, z = row['X'], row['Y'], row['Z']
            x /= self.ref_length
            y /= self.ref_length
            z /= self.ref_length

            # Trace the streamline for the current LE point
            self.trace_streamline(x, y, z, streamline_id=index)

    def export_streamlines_dat(self, filename):
        """
        Exports the streamlines to a .dat file in a column format (x y z),
        with tab delimiters and the number of points at the top of each segment.

        Parameters:
            filename (str): Name of the output .dat file.

        Returns:
            None
        """
        output_dir = "src/outputs"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            for streamline_id in set(point[3] for point in self.streamline_data):
                # Filter out points that exceed ref_length
                streamline = [p for p in self.streamline_data if p[3] == streamline_id]
                
                if not streamline or len(streamline) == 1:
                    continue
                
                # Write the number of points in the streamline
                f.write(f"{len(streamline)}\n")
                
                # Write the coordinates with streamline_id and order
                for x, y, z, s_id, order in streamline:
                    f.write(f"{x}\t{y}\t{z}\n")

        print(f"Streamline segment data saved to {filename}")

    def close_streamline_segments(self, filename):
        """
        Closes the segments by adding additional segments connecting the first points 
        and last points of adjacent streamlines.

        Parameters:
            filename (str): Name of the output .dat file.

        Returns:
            None
        """
        output_dir = "src/outputs"
        filepath = os.path.join(output_dir, filename)

        # Get sorted unique streamline IDs
        streamline_ids = sorted(set(point[3] for point in self.streamline_data))

        # Extract first and last points of each streamline
        first_points = []
        last_points = []
        for streamline_id in streamline_ids:
            streamline = [p for p in self.streamline_data if p[3] == streamline_id]
            if streamline:
                first_points.append(streamline[0])  # First point of the streamline
                last_points.append(streamline[-1])  # Last point of the streamline

        # Append new segments to the file
        with open(filepath, "a") as f:

            # Connect first points of adjacent streamlines
            for i in range(len(first_points) - 1):
                f.write(f"2\n")
                f.write(f"{first_points[i][0]}\t{first_points[i][1]}\t{first_points[i][2]}\n")
                f.write(f"{first_points[i + 1][0]}\t{first_points[i + 1][1]}\t{first_points[i + 1][2]}\n")

            # Connect last points of adjacent streamlines
            for i in range(len(last_points) - 1):
                f.write(f"2\n")
                f.write(f"{last_points[i][0]}\t{last_points[i][1]}\t{last_points[i][2]}\n")
                f.write(f"{last_points[i + 1][0]}\t{last_points[i + 1][1]}\t{last_points[i + 1][2]}\n")

        print(f"Closing segments appended to {filename}") 

# Example Usage
if __name__ == "__main__":
    # Initialize the streamline integrator with specific parameters
    integrator = StreamlineIntegrator(gamma=1.2, M1=10.0, theta_s=np.radians(20))

    # Extract leading-edge points from a file
    file_path = 'src/inputs/LeadingEdgeData_LeftSide.nmb'
    integrator.LE_points = process_LE_points.extract_points_from_file(file_path)

    # Create the lower surface by tracing streamlines
    integrator.create_lower_surface()

    print(np.degrees(integrator.TM_tabulation['Theta (radians)']))
    # # Export streamline data
    # dat_filename = "streamlines.dat"
    # integrator.export_streamlines_dat(dat_filename)

    # # Close the streamline segments
    # integrator.close_streamline_segments(dat_filename)
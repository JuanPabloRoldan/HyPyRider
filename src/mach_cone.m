clc
clear

file = fopen('LeadingEdgeData_LeftSide.nmb', 'r'); %Open the File
if file == -1
    error('File could not be opened.');
end

dim_chordinates = nmb_reader(file);  %Call reader function and print output
disp(dim_chordinates)

mach_vertex_chords = mach_vertex(dim_chordinates);
disp(mach_vertex_chords)

function mach_vertex_chords = mach_vertex(chords)
    x_v = chords(1,1) ;     %Id dimensionlized chords for vertex
    y_v = chords(1,2);
    z_v = chords(1,3);
    
    x_wt = chords(46,1);    %and for wingtip
    y_wt = chords(46,2);
    z_wt = chords(46,3);
    
    mu = asind(0.1);        %asin(0.1) if needed in radians
    
    c = y_v + x_v * tan(mu)-y_wt;                %Obtain c
    
    x_e_num = c^2 + z_wt^2 - tan(mu)^2;          %Find Xe
    x_e_dem = 2*tan(mu)*(c-tan(mu));
    x_e = x_e_num/x_e_dem;
    
    y_e = ((1-x_e)^2 * tan(mu)^2 - z_wt^2)^0.5 + y_wt; %Find Ye
    
    z_e = 0;

    mach_vertex_chords = [x_e,y_e,z_e];
end

function non_dim_chords = nmb_reader(fileID)
    %{
    This function reads and returns the non dimesnionlized chordinates x y z 
    of the left side leading edge.
    
    Parameters
    ----------
        fileID : .nmb file containing left side leading edge data
    
    Returns
    ----------
        non_dim_chords : 46x3 array of non dimensionlized x y z values
    %}
    chords = zeros(46,3);      %Initlize the array for storage
    lineNumber = 0;            %Initialize line number counter
    
    tline = fgetl(fileID);
    while ischar(tline)
        lineNumber = lineNumber + 1;
        
        if lineNumber >= 8 && lineNumber <= 53    %Exclude unwanted data
            strParts = strsplit(tline);           %Split the values into 3 and store in array
            chords(lineNumber-7,1) = str2double(strParts{1});
            chords(lineNumber-7,2) = str2double(strParts{2});
            chords(lineNumber-7,3) = str2double(strParts{3});
        end
        
        tline = fgetl(fileID);
    end
    fclose(fileID);  %Close the file
    
    ref_length = max(chords(:,1));      %obtain reference length
    non_dim_chords = chords/ref_length; %adjust values
end

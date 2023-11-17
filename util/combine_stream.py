import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
import arff
import sys


def get_arff_data_labels(filename):
    # Return ndarray corresponding to data and labels from arff data
    arff_content = arff.load(f.replace(',\n','\n') for f in open(filename, 'r'))
    data = arff_content['data']
    X = np.array([i[:1] for i in data])
    y = np.array([i[-1] for i in data])
    return X, y.astype(float)


def find_anomaly_intervals(y):
    # Return the intervals where there is an anomaly
    # @param y: ndarray of shape (N,) corresponding to anomaly labels
    # @Return list of lists denoting anomaly intervals in the form [start, end)
    change_indices = np.where(np.diff(y) != 0)[0]
    if len(change_indices) == 0:
        return []
    anom_intervals = []

    if y[change_indices[0]] == 0:
        i = 0
    else:
        i = 1
        anom_intervals.append([0,change_indices[0]+1])

    while (i + 1 < len(change_indices)):
        anom_intervals.append([change_indices[i]+1,change_indices[i+1]+1])
        i += 2

    if y[-1] == 1:
        anom_intervals.append([change_indices[-1]+1,len(y)])

    return anom_intervals


def split_arff(filepath, indices, trial_name, output_dir):
    #  Write new .arff files which splits original file from index
    #  File 1 contains points from index range [0, index)
    #  File 2 contains points from index range [index, file_length)
    #  @param filepath: String representing filepath of .arff file to split
    #  @param indices: list of int representing index of original data to split file
    #  @param trial_name: String representing identifying name of split
    #  @param output_dir: String representing directory to write arff file
    #  @Return list of filepath
    file = filepath.split('/')[-1]
    if output_dir == None:
        output_dir = filepath[:-len(file)]
    filename = file.split('.arff')[0]

    content = []
    output_files = []

    with open(filepath, 'r') as input_file:
        for line in input_file:
            content.append(f"{line.strip()}\n")

    header_lines = content[:7]
    for i in range(len(indices)):
        if i > 0:
            data = content[indices[i-1]+7:indices[i]+7]
        else:
            data = content[7:indices[i]+7]
        output_file_name = f"{output_dir}{filename}_{trial_name}_{i}.arff"
        with open(output_file_name, 'w') as output_file:
            output_file.writelines(header_lines + data)
        output_files.append(output_file_name)
        print(f"Generated file: {output_file_name}")
    data = content[indices[-1]+7:]
    output_file_name = f"{output_dir}{filename}_{trial_name}_{len(indices)}.arff"
    with open(output_file_name, 'w') as output_file:
        output_file.writelines(header_lines + data)
    print(f"Generated file: {output_file_name}")
    output_files.append(output_file_name)
    return output_files


def generate_moa_abrupt(
        moa_file_path, 
        stream_1_path, 
        stream_2_path,
        position,
        length,
        output_path
    ):
    # Generate command to run with MOA CLI to create abrupt stream
    command_p1 = f'!cd {moa_file_path} && java -cp moa.jar -javaagent:sizeofag-1.0.4.jar' 
    stream_1 = f'(ArffFileStream -f {stream_1_path} -c 0)'
    stream_2 = f'(ArffFileStream -f {stream_2_path} -c 0)'
    drift_stream = f'(ConceptDriftStream -s {stream_1} -d {stream_2} -p {position} -w 1)'
    command_p2 = f'moa.DoTask "WriteStreamToARFFFile  -s {drift_stream} -f {output_path} -m {length}"'
    return f'{command_p1} {command_p2}'


def generate_moa_gradual(
        moa_file_path, 
        stream_1_path, 
        stream_2_path,
        position,
        width,
        length,
        output_path
    ):
    # Generate command to run with MOA CLI to create gradual stream
    command_p1 = f'!cd {moa_file_path} && java -cp moa.jar -javaagent:sizeofag-1.0.4.jar' 
    stream_1 = f'(ArffFileStream -f {stream_1_path} -c 0)'
    stream_2 = f'(ArffFileStream -f {stream_2_path} -c 0)'
    drift_stream = f'(ConceptDriftStream -s {stream_1} -d {stream_2} -p {position} -w {width})'
    command_p2 = f'moa.DoTask "WriteStreamToARFFFile  -s {drift_stream} -f {output_path} -m {length}"'
    return f'{command_p1} {command_p2}'


def plot_anomaly(X, y, start=0, end=sys.maxsize, title="", marker="-"):
    # Plot the data with highlighted anomaly
    plt.figure(figsize=(12,2))
    plt.plot(np.arange(start,min(X.shape[0],end)), X[start:end], f"{marker}b")
    for (anom_start, anom_end) in find_anomaly_intervals(y):
        if start <= anom_end and anom_start <= anom_end:
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], f"{marker}r")
    if len(title) > 0:
        plt.title(title)

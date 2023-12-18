import arff
import numpy as np
import matplotlib.pyplot as plt
import random
import sys


## Find ndarray corresponding to data and labels from arff data
#  @param filename: string, filename of arff data source
#  @Return X, y: ndarrays corresponding to data (N, 1) and labels (N,) from arff data
def get_arff_data_labels(filename):
    arff_content = arff.load(f.replace(',\n','\n') for f in open(filename, 'r'))
    data = arff_content['data']
    X = np.array([i[:1] for i in data])
    y = np.array([i[-1] for i in data])
    return X, y.astype(float)


## Find the intervals where there is an anomaly
#  @param y: ndarray of shape (N,) corresponding to anomaly labels
#  @Return list of lists denoting anomaly intervals in the form [start, end)
def find_anomaly_intervals(y):
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


## Function to return indices to cut source arff files to combine
#  @param target_p_drift: float, target percentage of drift
#  @param target_n_drift: int, target number of drift sequences
#  @param length: int, total length of new stream
#  @param p_drift_after: float, target percent of drift coming after anomaly
#  @param max_stream: int, maximum stream index (ex. for 6 streams, stream index 5 is max) 
#  @param anom_ints: list of list of int, anomaly intervals [start, end] of input streams
#  @Returns streams, positions, w_drift, stream_cuts, seq_drift_after
def get_split_index(target_p_drift, target_n_drift, length, p_drift_after, max_stream, anom_ints):
    print('Getting divisions...')
    # multiply n_drift by a factor to account for reduced number of drifts later
    divisions = get_divisions(target_p_drift*1.5, int(target_n_drift*1.5), length)
    w_drift = [divisions[i] for i in range(1,2*target_n_drift,2)]
    w_stream = [divisions[i] for i in range(0,2*target_n_drift+1,2)]
    while min(w_stream) < min(w_drift) * 5 or w_stream[0] < w_drift[0]: ## check this
        divisions = get_divisions(target_p_drift, target_n_drift, length)
        w_drift = [divisions[i] for i in range(1,2*target_n_drift,2)]
        w_stream = [divisions[i] for i in range(0,2*target_n_drift+1,2)]
    
    print('Getting order of drifts coming after anomaly...')
    seq_drift_after = get_seq_drift_after(p_drift_after, target_n_drift)

    print('Getting drift center positions...') 
    curr_stream = random.randint(0,max_stream)
    streams = [curr_stream]
    positions = []
    init_pos = divisions[0]
    for n in range(target_n_drift):
        curr_stream = get_next_stream(curr_stream, max_stream)
        streams.append(curr_stream)
        # Note: may not result in exact lengths specified in divisions
        # due to positions of actual anomalies in data
        drift_pos = find_next_drift_pos(
            init_pos,
            w_drift[n], 
            seq_drift_after[n], 
            anom_ints[streams[n]], 
            anom_ints[streams[n+1]]
        )
        if drift_pos == -1:
            break
        init_pos = drift_pos + w_stream[n+1]
        positions.append(drift_pos)
    
    print('Getting stream file cuts...')
    n_drift = len(positions)
    seq_drift_after = seq_drift_after[:n_drift]
    w_drift = w_drift[:n_drift]
    streams = streams[:n_drift+1]
    
    stream_cuts = [[] for i in range(max_stream + 1)]
    for (i,drift_after) in enumerate(seq_drift_after):
        s_prev, s_next  = streams[i], streams[i+1]
        if drift_after:
            stream_cuts[s_prev].append(positions[i] + w_drift[i])
            for j in range(max_stream + 1):
                if j != s_prev: 
                    stream_cuts[j].append(positions[i] )
        else:
            stream_cuts[s_next].append(positions[i] - w_drift[i])
            for j in range(max_stream + 1):
                if j != s_next: 
                    stream_cuts[j].append(positions[i] )

    ## @Returns
    #    streams: list of int, denoting order of streams in combination
    #    positions: list of int, center position of drift
    #    w_drift: list of int, width of each drift
    #    stream_cuts: list of list of int, where to cut each source arff file
    #    seq_drift_after: list of boolean indicating whether drift comes after or before anomaly 

    return streams, positions, w_drift, stream_cuts, seq_drift_after


## Randomly generate distribution of source and drift stream based on percentage of drift
#  @param p_drift: float, percentage of target drift
#  @param n_drift: int, number of drift sequences
#  @param length: int, total length of final data stream
#  @Return divisions: list of int, widths corresponding to each alternating 
#     non-drift/drift sequence
def get_divisions(p_drift, n_drift, length):
    p_non_drift = 1 - p_drift
    drift_div = [random.uniform(1,100) for _ in range(n_drift)]
    non_drift_div = [random.uniform(1,100) for _ in range(n_drift + 1)]
    total_drift = sum(drift_div)
    total_non_drift = sum(non_drift_div)
    drift_div_norm = [val/total_drift for val in drift_div]
    non_drift_div_norm = [val/total_non_drift for val in non_drift_div]
    divisions = []
    for i in range(len(drift_div_norm)):
        divisions.append(non_drift_div_norm[i] * p_non_drift)
        divisions.append(drift_div_norm[i] * p_drift)
    divisions.append(1 - sum(divisions))
    divisions = [int(length * p) for p in divisions]
    return divisions


## Randomly generates next different stream in sequence
#  @param curr_stream: int, index of current stream (ex. stream 0, stream 1, etc)
#  @param total_streams: int, total number of streams
#  @Return next_stream: int, index of next stream
def get_next_stream(curr_stream, max_stream=5):
    next_stream = random.randint(0,max_stream)
    while next_stream == curr_stream:
        next_stream = random.randint(0,max_stream)
    return next_stream


## Randomly generates sequence of relative drift position (ie. after or before anomaly)
#    where True indicates that the drift occurs after the anomaly
#  @param p_drift_after: float, percentage of drift transitions where anomaly comes before
#  @param n_drift: int, number of drift sequences
#  @Return seq_drift_after: list of boolean of relative drift position
def get_seq_drift_after(p_drift_after, n_drift):
    n_before = int(p_drift_after * n_drift)
    n_after = n_drift - n_before
    seq_drift_after = [True] * n_before + [False] * n_after
    random.shuffle(seq_drift_after)
    return seq_drift_after


## Identify position of drift based on an initial position and existing anomalies in stream
#  @param init_pos: int, initial position to guide selection of new drift position
#  @param w_drift: int, width of drift
#  @param drift_after: boolean, relative position of drift surrounding anomaly
#  @param anom_int_curr: list of list of int, anomaly intervals of current stream
#  @param anom_int_next: list of list of int, anomaly intervals of next stream
#  @Return drift_pos: int, center position of drift
def find_next_drift_pos(init_pos, w_drift, drift_after, anom_int_curr, anom_int_next):
    end = init_pos + w_drift
    if drift_after:
        try:
            drift_pos = min(anom_int[1] for anom_int in anom_int_curr if anom_int[1] >= end)
        except:
            drift_pos = -1
        else:
            drift_pos += w_drift // 2
    else:
        try:
            drift_pos = min(anom_int[0] for anom_int in anom_int_next if anom_int[0] >= end)
        except:
            drift_pos = -1
        else:
            drift_pos -= w_drift // 2
    return drift_pos


## Write new .arff files which splits original file from index
#    File 0 contains points from index range [0, index_1)
#    File n contains points from index range [index_{n-1}, index_n) for 0 < n < N
#    File N contains points from index range [index_N - 1, file_length)
#  @param filepath: String representing filepath of .arff file to split
#  @param indices: list of int representing N-1 indices of original data to split file
#  @param trial_name: String representing identifying name of split
#  @param output_dir: String representing directory to write arff file
#  @Return list of filepath
def split_arff(filepath, indices, trial_name, output_dir):
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
        # print(f"Generated file: {output_file_name}")
    data = content[indices[-1]+7:]
    output_file_name = f"{output_dir}{filename}_{trial_name}_{len(indices)}.arff"
    with open(output_file_name, 'w') as output_file:
        output_file.writelines(header_lines + data)
    # print(f"Generated file: {output_file_name}")
    output_files.append(output_file_name)
    return output_files


def generate_moa_command(moa_file_path, stream, output_path, length):
    # Generate command to run with MOA CLI to create gradual stream
    # @param moa_file_path: String, path to execute moa
    # @param stream: String representing stream to be generated
    command_p1 = f'!cd {moa_file_path} && java -cp moa.jar -javaagent:sizeofag-1.0.4.jar' 
    command_p2 = f'moa.DoTask "WriteStreamToARFFFile  -s ({stream}) -f {output_path} -m {length}"'
    return f'{command_p1} {command_p2}'


def generate_grad_stream_from_stream(
        stream_1,
        stream_2,
        position,
        width
    ):
    drift_stream = f'ConceptDriftStream -s ({stream_1}) -d ({stream_2}) -p {position} -w {width}'
    return drift_stream


def get_moa_stream_from_arff(file_path):
    return f'ArffFileStream -f {file_path} -c 0'


def generate_moa_abrupt_stream(
        stream_1, 
        stream_2,
        position
    ):
    # Generate command to run with MOA CLI to create abrupt stream
    drift_stream = f'ConceptDriftStream -s ({stream_1}) -d ({stream_2}) -p {position} -w 1'
    return drift_stream


def plot_anomaly(X, y, ax, start=0, end=sys.maxsize, title="", marker="-"):
    # Plot the data with highlighted anomaly
    ax.plot(np.arange(start,min(X.shape[0],end)), X[start:end], f"{marker}b")
    for (anom_start, anom_end) in find_anomaly_intervals(y):
        if start <= anom_end and anom_start <= anom_end:
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            ax.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], f"{marker}r")
    if len(title) > 0:
        ax.set_title(title)

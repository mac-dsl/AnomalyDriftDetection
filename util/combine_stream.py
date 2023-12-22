import arff
import numpy as np
import random
import sys


COLOURS = {
    0: 'tab:blue',
    1: 'tab:green',
    2: 'tab:red',
    3: 'tab:cyan',
    4: 'tab:pink',
    5: 'tab:purple',
    'drift': 'gold'
}


#  @param filename: string, filename of arff data source
#  @Return X, y: ndarrays corresponding to data (N, 1) and labels (N,)
#              from arff data
def get_arff_data_labels(filename):
    """
    Find ndarray corresponding to data and labels from arff data
    """
    arff_content = arff.load(f.replace(',\n', '\n') for f in open(filename, 'r'))
    data = arff_content['data']
    X = np.array([i[:1] for i in data])
    y = np.array([i[-1] for i in data])
    return X.astype(float), y.astype(float)


#  @param y: ndarray of shape (N,) corresponding to anomaly labels
#  @Return list of lists denoting anomaly intervals in the form [start, end)
def find_anomaly_intervals(y):
    """
    Method to find the intervals where there is an anomaly
    """
    change_indices = np.where(np.diff(y) != 0)[0]
    if len(change_indices) == 0:
        return []
    anom_intervals = []

    if y[change_indices[0]] == 0:
        i = 0
    else:
        i = 1
        anom_intervals.append([0, change_indices[0]+1])

    while i + 1 < len(change_indices):
        anom_intervals.append([change_indices[i]+1, change_indices[i+1]+1])
        i += 2

    if y[-1] == 1:
        anom_intervals.append([change_indices[-1]+1, len(y)])

    return anom_intervals


#  @param p_drift: float, target percentage of drift
#  @param n_drift: int, target number of drift sequences
#  @param length: int, total length of new stream
#  @param p_drift_after: float, target percent of drift coming after anomaly
#  @param max_stream: int, maximum stream index
#                 (ex. for 6 streams, stream index 5 is max)
#  @param anom_ints: list of list of int, anomaly intervals [start, end] of
#                 input streams
#  @Returns streams, positions, w_drift, stream_cuts, seq_drift_after
def get_split_index(
        p_drift,
        n_drift,
        length,
        p_drift_after,
        max_stream,
        anom_ints
        ):
    """
    Function to return indices to cut source arff files to combine
    """
    print('Getting partitions...')
    # multiply n_drift by a factor to account for reduced number of
    # final drifts
    partitions = get_partitions(p_drift, int(n_drift), length)
    w_drift = [partitions[i] for i in range(1, 2*n_drift, 2)]
    w_stream = [partitions[i] for i in range(0, 2*n_drift+1, 2)]
    while min(w_stream) < min(w_drift) * 5 or w_stream[0] < w_drift[0]:
        partitions = get_partitions(p_drift, int(n_drift), length)
        w_drift = [partitions[i] for i in range(1, 2*n_drift, 2)]
        w_stream = [partitions[i] for i in range(0, 2*n_drift+1, 2)]

    print('Getting order of drifts coming after anomaly...')
    seq_drift_after = get_seq_drift_after(p_drift_after, n_drift)

    print('Getting drift center positions...')
    curr_stream = random.randint(0, max_stream)
    streams = [curr_stream]
    positions = []
    init_pos = partitions[0]
    for n in range(n_drift):
        curr_stream = get_next_stream(curr_stream, max_stream)
        streams.append(curr_stream)
        # Note: may not result in exact lengths specified in partitions
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
    for (i, drift_after) in enumerate(seq_drift_after):
        s_prev, s_next = streams[i], streams[i+1]
        if drift_after:
            stream_cuts[s_prev].append(positions[i] + w_drift[i])
            for j in range(max_stream + 1):
                if j != s_prev:
                    stream_cuts[j].append(positions[i])
        else:
            stream_cuts[s_next].append(positions[i] - w_drift[i])
            for j in range(max_stream + 1):
                if j != s_next:
                    stream_cuts[j].append(positions[i])

    #  @Returns
    #    streams: list of int, denoting order of streams in combination
    #    positions: list of int, center position of drift
    #    w_drift: list of int, width of each drift
    #    stream_cuts: list of list of int, where to cut each source arff file
    #    seq_drift_after: list of boolean indicating relative drift position

    return streams, positions, w_drift, stream_cuts, seq_drift_after


#  @param p_drift: float, percentage of target drift
#  @param n_drift: int, number of drift sequences
#  @param length: int, total length of final data stream
#  @Return partitions: list of int, widths corresponding to each alternating
#     non-drift/drift sequence
def get_partitions(p_drift, n_drift, length):
    """
    Randomly generate distribution of source and drift stream based on
    percentage of drift
    """
    p_non_drift = 1 - p_drift
    drift_div = [random.uniform(1, 100) for _ in range(n_drift)]
    non_drift_div = [random.uniform(1, 100) for _ in range(n_drift + 1)]
    total_drift = sum(drift_div)
    total_non_drift = sum(non_drift_div)
    drift_div_norm = [val/total_drift for val in drift_div]
    non_drift_div_norm = [val/total_non_drift for val in non_drift_div]
    partitions = []
    for (i, ddn) in enumerate(drift_div_norm):
        partitions.append(non_drift_div_norm[i] * p_non_drift)
        partitions.append(ddn * p_drift)
    partitions.append(1 - sum(partitions))
    partitions = [int(length * p) for p in partitions]
    return partitions


#  @param curr_stream: int, index of current stream (ex. stream 0, stream 1)
#  @param total_streams: int, total number of streams
#  @Return next_stream: int, index of next stream
def get_next_stream(curr_stream, max_stream=5):
    """
    Randomly generates next different stream in sequence
    """
    next_stream = random.randint(0, max_stream)
    while next_stream == curr_stream:
        next_stream = random.randint(0, max_stream)
    return next_stream


#  Randomly generates sequence of relative drift position (ie. after anomaly)
#  where True indicates that the drift occurs after the anomaly
#  @param p_drift_after: float, percentage of drift transitions with drift
#                     coming after anomaly
#  @param n_drift: int, number of drift sequences
#  @Return seq_drift_after: list of boolean of relative drift position
def get_seq_drift_after(p_drift_after, n_drift):
    n_before = int(p_drift_after * n_drift)
    n_after = n_drift - n_before
    seq_drift_after = [True] * n_before + [False] * n_after
    random.shuffle(seq_drift_after)
    return seq_drift_after


#  @param init_pos: int, initial position to guide selection of drift position
#  @param w_drift: int, width of drift
#  @param drift_after: boolean, relative position of drift surrounding anomaly
#  @param anom_int_curr: list of list of int, anomaly intervals of stream
#  @param anom_int_next: list of list of int, anomaly intervals of stream
#  @Return drift_pos: int, center position of drift
def find_next_drift_pos(
        init_pos,
        w_drift,
        drift_after,
        anom_int_curr,
        anom_int_next
        ):
    """
    Identify position of drift based on an initial position and existing
    anomalies in stream
    """
    end = init_pos + w_drift
    if drift_after:
        try:
            drift_pos = min(a_i[1] for a_i in anom_int_curr if a_i[1] >= end)
        except ValueError:
            drift_pos = -1
        else:
            drift_pos += w_drift // 2
    else:
        try:
            drift_pos = min(a_i[0] for a_i in anom_int_next if a_i[0] >= end)
        except ValueError:
            drift_pos = -1
        else:
            drift_pos -= w_drift // 2
    return drift_pos


#  @param filepath: String representing filepath of .arff file to split
#  @param indices: list of int representing N-1 indices of original data to
#                  split file
#  @param trial_name: String representing identifying name of split
#  @param output_dir: String representing directory to write arff file
#  @Return list of filepath
def split_arff(filepath, indices, trial_name, output_dir):
    """
    Write new .arff files which splits original file from index
    - File 0 contains points from index range [0, index_1)
    - File n contains points from index range [index_{n-1}, index_n) for 0 < n < N
    - File N contains points from index range [index_N - 1, file_length)
    """
    file = filepath.split('/')[-1]
    if output_dir is None:
        output_dir = filepath[:-len(file)]
    filename = file.split('.arff')[0]

    content = []
    output_files = []

    with open(filepath, 'r') as input_file:
        for (i, line) in enumerate(input_file):
            if i > 7:
                num_vals = line.strip().split(',')[:-1]
                num_vals = [float(n) for n in num_vals]
                newline = f"{num_vals[0]},{num_vals[1]},\n"
                content.append(newline)
            else:
                content.append(line)

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
    data = content[indices[-1]+7:]
    output_file_name = f"{output_dir}{filename}_{trial_name}_{len(indices)}.arff"
    with open(output_file_name, 'w') as output_file:
        output_file.writelines(header_lines + data)
    output_files.append(output_file_name)
    return output_files


#  @param moa_file_path: String, path to execute moa
#  @param stream: String representing stream to be generated
#  @Return string, command to run MOA through command line
def generate_moa_command(moa_file_path, stream, output_path, length):
    """
    Generate command to run with MOA CLI to create gradual stream
    """
    command_p1 = f'cd {moa_file_path} && java -cp moa.jar -javaagent:sizeofag-1.0.4.jar'
    command_p2 = \
        f'moa.DoTask "WriteStreamToARFFFile  -s ({stream}) -f {output_path} -m {length}"'
    return f'{command_p1} {command_p2}'


#  @param stream_1: String, first stream in drift
#  @param stream_2: String, second stream in drift
#  @param position: int, center positions of drift
#  @param width: int, width of drift
#  @Return string, representation of resultant MOA stream
def generate_grad_stream_from_stream(
        stream_1,
        stream_2,
        position,
        width
        ):
    """
    Generate command line representation of MOA ConceptDriftStream
    object (gradual drift)
    """
    drift_stream = \
        f'ConceptDriftStream -s ({stream_1}) -d ({stream_2}) -p {position} -w {width}'
    return drift_stream


#  @param file_path: String, filepath to source arff file
#  @Return string, representation of MOA ArffFileStream
def get_stream_from_arff(file_path):
    """
    Generate command line representation of MOA ArffFileStream object
    """
    return f'ArffFileStream -f {file_path} -c 0'


#  @param stream_1: String, first stream in drift
#  @param stream_2: String, second stream in drift
#  @param position: int, center positions of drift
#  @Return string, representation of resultant MOA stream
def generate_abrupt_stream_from_stream(
        stream_1,
        stream_2,
        position
        ):
    """
    Generate command line representation of MOA ConceptDriftStream
    object (abrupt drift)
    """
    drift_stream = f'ConceptDriftStream -s ({stream_1}) -d ({stream_2}) -p {position} -w 1'
    return drift_stream


#  @params X: int iterable, data points
#  @params y: int iterable, anomaly labels
#  @params ax: Axes object to plot graph
#  @params start: int, start of interval to plot
#  @params end: int, endn of interval to plot
#  @params title: string, title of plot
#  @params marker: string, marker type for plot, default '-' (line)
#  @params size: int, fontsize, default=10
def plot_anomaly(X, y, ax, start=0, end=sys.maxsize, title="", marker="-", size=10, show_label=False):
    """
    Plot the data with highlighted anomaly
    """
    ax.plot(np.arange(start, min(X.shape[0], end)), X[start:end],
            f"{marker}b", label="_"*(not show_label) + "Drift Stream")
    for (i, (anom_start, anom_end)) in enumerate(find_anomaly_intervals(y)):
        if start <= anom_end and anom_start <= anom_end:
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            ax.plot(np.arange(anom_start, anom_end),
                    X[anom_start:anom_end],
                    f"{marker}r",
                    label="_"*(i + (not show_label)) + "Anomaly"
                    )
    if len(title) > 0:
        ax.set_title(title, size=size)


#  @param positions: list of int, center position of drift
#  @params y: int iterable, drift labels
#  @params streams: list of int, denoting order of streams in combination
#  @params ax: Axes object to plot graph
#  @params colours: dictionary representing background colours for
#        denoting stream segments
#  @params start: int, start of interval to plot
#  @params end: int, endn of interval to plot
def plot_stream_drift(positions, y, streams, ax, colours=COLOURS, start=0, end=sys.maxsize):
    """
    Plot source stream and drift periods
    """
    positions = positions + [len(y)]
    for i in range(0, len(positions)-1):
        if positions[i] > end:
            break
        elif positions[i+1] < start:
            pass
        else:
            s_start = max(start, positions[i])
            s_end = min(positions[i+1], end)
            colour = colours[streams[i]]
            ax.axvspan(s_start, s_end, facecolor=colour, alpha=0.3)
    drift_ints = [di for di in find_anomaly_intervals(y) if di[0] < end or di[1] > start]
    for (i, (drift_start, drift_end)) in enumerate(drift_ints):
        if drift_start > end:
            break
        elif drift_end < start:
            pass
        else:
            drift_start = max(start, drift_start)
            drift_end = min(end, drift_end)
            ax.axvspan(drift_start, drift_end, facecolor=colours['drift'], alpha=1, label='_'*i + 'drift')


## Alternative: place drift positions according to partitions
#  adjust percentage through width size of each drift
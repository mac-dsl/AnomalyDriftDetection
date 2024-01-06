import arff
import numpy as np
import random
import sys


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


#  @param length: int, total length of new stream
#  @param p_drift: float, target percentage of drift
#  @param n_drift: int, target number of drift sequences
#  @param p_before: float, target percent of drift coming before anomaly
#  @param max_stream: int, maximum stream index
#                 (ex. for 6 streams, stream index 5 is max)
#  @param total_anoms: list of tuple of (int, [int, int]),
#      Represents (stream, [start, end]) for anomaly intervals across all
#      streams ordered by start position
#  @Returns streams, positions, w_drift, stream_cuts, seq_drift_before
def get_split_index(
        length,
        p_drift,
        n_drift,
        p_before,
        max_stream,
        total_anom_ints
        ):
    """
    Function to return indices to cut source arff files to combine
    """
    print('\tGetting partitions...')
    partitions = get_partitions(p_drift, int(n_drift), length)
    w_drift = [partitions[i] for i in range(1, 2*n_drift, 2)]
    w_stream = [partitions[i] for i in range(0, 2*n_drift+1, 2)]
    # Ensure that stream segments are generally longer than drifts
    while min(w_stream) < min(w_drift) * 5 or w_stream[0] < w_drift[0]:
        partitions = get_partitions(p_drift, int(n_drift), length)
        w_drift = [partitions[i] for i in range(1, 2*n_drift, 2)]
        w_stream = [partitions[i] for i in range(0, 2*n_drift+1, 2)]

    # Construct lists representing characteristics of each individual
    # drift
    print('\tGetting order of drifts coming before anomaly...')
    seq_drift_before = get_seq_drift_before(p_before, n_drift)
    print('\tGetting drift center positions...')
    curr_stream = random.randint(0, max_stream)
    streams = [curr_stream]
    positions = []
    init_pos = partitions[0]
    for n in range(n_drift):
        curr_stream, drift_pos = find_next_drift_pos(
            init_pos,
            w_drift[n],
            seq_drift_before[n],
            total_anom_ints,
            curr_stream
        )
        if drift_pos == -1:
            break
        streams.append(curr_stream)
        init_pos = drift_pos + w_stream[n+1]
        positions.append(drift_pos)

    print('\tGetting stream file cuts...')
    n_drift = len(positions)
    seq_drift_before = seq_drift_before[:n_drift]
    w_drift = w_drift[:n_drift]
    streams = streams[:n_drift+1]

    # Determine where ARFF files should be split to create
    # intermediate files
    stream_cuts = [[] for i in range(max_stream + 1)]
    for (i, drift_before) in enumerate(seq_drift_before):
        s_prev, s_next = streams[i], streams[i+1]
        if drift_before:
            stream_cuts[s_next].append(positions[i] - w_drift[i])
            for j in range(max_stream + 1):
                if j != s_next:
                    stream_cuts[j].append(positions[i])
        else:
            stream_cuts[s_prev].append(positions[i] + w_drift[i])
            for j in range(max_stream + 1):
                if j != s_prev:
                    stream_cuts[j].append(positions[i])

    #  @Returns
    #    streams: list of int, denoting order of streams in combination
    #    positions: list of int, center position of drift
    #    w_drift: list of int, width of each drift
    #    stream_cuts: list of list of int, where to cut each source arff file
    #    seq_drift_before: list of boolean indicating relative drift position

    return streams, positions, w_drift, stream_cuts, seq_drift_before


#  @param p_drift: float, percentage of target drift
#  @param n_drift: int, number of drift sequences
#  @param length: int, total length of final data stream
#  @param min_stream: int, minimum length for a stream partition
#  @Return partitions: list of int, widths corresponding to each alternating
#     non-drift/drift sequence
def get_partitions(p_drift, n_drift, length, min_stream=1000):
    """
    Randomly generate distribution of source and drift stream based on
    percentage of drift
    """
    # Randomly generate divisions among drift and non-drift segments
    drift_part = [random.uniform(1, 100) for _ in range(n_drift)]
    non_drift_part = [random.uniform(1, 100) for _ in range(n_drift + 1)]
    while min(non_drift_part) < min_stream / length:
        non_drift_part = [random.uniform(1, 100) for _ in range(n_drift + 1)]
    total_drift = sum(drift_part)
    total_non_drift = sum(non_drift_part)
    # Normalize divisions to sum to 1
    drift_part_norm = [val/total_drift for val in drift_part]
    non_drift_part_norm = [val/total_non_drift for val in non_drift_part]
    # Multiply each list by the percentage of length it needs to sum to,
    # for drift segments and non-drift segments separately
    partitions = []
    p_non_drift = 1 - p_drift
    for (i, drift_part_norm_i) in enumerate(drift_part_norm):
        partitions.append(non_drift_part_norm[i] * p_non_drift)
        partitions.append(drift_part_norm_i * p_drift)
    partitions.append(1 - sum(partitions))
    # Determine the number of data points required to achieve the percentages
    partitions = [int(length * p) for p in partitions]
    return partitions


#  @param anom_ints: list of list of int, list of anomaly intervals
#                 [[s1_start, s1_end], [s2_start, s2_end], ...] of inputs
#  @Return total_anoms: list of tuple of (int, [int, int]),
#      Represents (stream, [start, end]) for anomaly intervals across all
#      streams ordered by start position
def get_total_anoms(anom_ints):
    """
    Create list of anomaly intervals across all streams in increasing order
    of start position
    """
    total_anoms = []
    anom_ints_copy = [a_i.copy() for a_i in anom_ints]
    first_start = [anom_ints_copy[i][0] for i in range(len(anom_ints_copy))]
    while any(len(x) != 0 for x in anom_ints_copy):
        start_ints = [x[0] for x in first_start]
        i = np.argmin(start_ints)
        total_anoms.append((i, anom_ints_copy[i].pop(0)))
        if len(anom_ints_copy[i]) > 0:
            first_start[i] = anom_ints_copy[i][0]
        else:
            first_start[i] = [sys.maxsize, 0]
    return total_anoms


#  @param p_before: float, percentage of drift transitions where drift is
#                     inserted before anomaly
#  @param n_drift: int, number of drift sequences
#  @Return seq_drift_before: list of boolean of relative drift position
def get_seq_drift_before(p_before, n_drift):
    """
    Randomly generates sequence of relative drift position (ie. before anomaly)
    where True indicates that the drift occurs before the anomaly
    """
    n_before = int(p_before * n_drift)
    n_after = n_drift - n_before
    seq_drift_before = [True] * n_before + [False] * n_after
    random.shuffle(seq_drift_before)
    return seq_drift_before


#  @param init_pos: int, initial position to guide selection of drift position
#  @param w_drift: int, width of drift
#  @param drift_before: boolean, relative position of drift surrounding anomaly
#  @param total_anoms: list of tuple of (int, [int, int]),
#      Represents (stream, [start, end]) for anomaly intervals across all
#      streams ordered by start position
#  @param curr_stream: int, index of current stream (ex. stream 0, stream 1)
#  @Return drift_pos: int, center position of drift
def find_next_drift_pos(
        init_pos,
        w_drift,
        drift_before,
        total_anoms,
        curr_stream
        ):
    """
    Identify position of drift based on an initial position and existing
    anomalies in stream
    """
    end = init_pos + w_drift
    if not drift_before:
        try:
            # If drift comes after the anomaly, the center point of drift should occur
            # after the end point for the anomaly with a buffer of w_drift / 2 to avoid
            # interfering with the anomaly during drift generation
            valid_anoms = \
                [a_i for a_i in total_anoms if a_i[1][0] >= end and a_i[0] != curr_stream]
            i = np.argmin(a_i[1] for a_i in valid_anoms)
            drift_pos = valid_anoms[i][1][1]
            next_stream = valid_anoms[i][0]
        except IndexError:
            drift_pos = -1
            next_stream = None
        else:
            drift_pos += w_drift // 2
    else:
        try:
            # If drift comes before the anomaly, the center point of drift should occur
            # before the starting point for the anomaly with a buffer of w_drift / 2 to avoid
            # interfering with the anomaly during drift generation
            valid_anoms = \
                [a_i for a_i in total_anoms if a_i[1][1] >= end and a_i[0] != curr_stream]
            i = np.argmin(a_i[0] for a_i in valid_anoms)
            drift_pos = valid_anoms[i][1][0]
            next_stream = valid_anoms[i][0]
        except IndexError:
            drift_pos = -1
            next_stream = None
        else:
            drift_pos -= w_drift // 2
    return next_stream, drift_pos

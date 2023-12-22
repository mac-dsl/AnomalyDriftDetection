from util.combine_stream import *
import os
import pandas as pd
import subprocess


MOA_FILEPATH = '/Users/tammyz/Desktop/3H03/moa-release-2023.04.0/lib'
SOURCE_DIR = "data/benchmark/ECG"
OUTPUT_DIR = "/Users/tammyz/Desktop/3H03/AnomalyDriftDetection/data/synthetic"


#  Create new gradual drift-injected stream using MOA based on parameters
#  @param selected_streams: list of string representing arff filenames of streams
#  @param n_drift: int, target number of drift sequences
#  @param length: int, total length of new stream
#  @param p_drift_after: float, target percent of drift coming after anomaly
#  @param max_stream: int, maximum stream index (ex. for 6 streams, stream index 5 is max)
#  @param anom_ints: list of list of int, anomaly intervals [start, end] of input streams
#  @param trial_name: string, name of subdirectory to export output
#  @Returns output_path, drift_label, positions, streams, seq_drift_after
def run_generate_grad_stream_moa(
        selected_streams,
        p_drift,
        n_drift,
        length,
        p_drift_after,
        max_stream,
        anom_ints,
        trial_name
        ):
    print('Generating splits...')
    streams, positions, w_drift, stream_cuts, seq_drift_after = \
        get_split_index(p_drift, n_drift, length, p_drift_after, max_stream, anom_ints)

    print('Creating intermediate files...')
    streams_intermed = []
    for i in range(len(selected_streams)):
        input_file = os.path.join(SOURCE_DIR, selected_streams[i])
        split_index = stream_cuts[i]
        trialname = 'p50_inter'
        output_dir = f"{OUTPUT_DIR}/intermediate/"
        streams_intermed.append(split_arff(input_file, split_index, trialname, output_dir))

    print('Recursively generating MOA command...')
    moa_streams = []
    for (i, s) in enumerate(streams):
        moa_streams.append(get_stream_from_arff(streams_intermed[s][i]))

    for i in range(1, len(moa_streams)):
        if i > 1:
            drift_stream = generate_grad_stream_from_stream(drift_stream, moa_streams[i], positions[i-1], w_drift[i-1])
        else:
            drift_stream = generate_grad_stream_from_stream(moa_streams[0], moa_streams[1], positions[0], w_drift[0])

    p_drift = sum(w_drift)/length
    n_drift = len(positions)
    p_drift_after = sum(seq_drift_after)/len(seq_drift_after)
    filename = f"ECG_grad_p{int(p_drift*100)}_n{n_drift}_a{int(p_drift_after*100)}"
    print("Drift filename: ", filename)
    print("Streams:", streams)
    print("Positions:", positions)
    print("w_drift:", w_drift)

    output_path = f'{OUTPUT_DIR}/{trial_name}/{filename}.arff'
    command = generate_moa_command(MOA_FILEPATH, drift_stream, output_path, length)
    print('Running terminal command...')
    subprocess.run(command, shell=True)

    print('Generating drift labels...')
    drift_label = get_drift_labels(positions, w_drift, length)
    drift_label_df = pd.DataFrame(drift_label)
    drift_label_df.to_csv(f'{OUTPUT_DIR}/{trial_name}/{filename}.csv')

    drifts_param_df = pd.DataFrame({
        'streams': streams,
        'positions': [0] + positions
        })
    drifts_param_df.to_csv(f'{OUTPUT_DIR}/{trial_name}/{filename}_params.csv')

    #  @Returns
    #    output_path: string, path to file where generated data stream is exported to
    #    drift_label: list of int, whether or not a point is labelled as drift
    #    streams: list of int, denoting order of streams in combination
    #    positions: list of int, center position of drift
    #    seq_drift_after: list of boolean indicating whether drift comes after or before anomaly

    return output_path, drift_label, streams, positions, seq_drift_after


#  Returns drift labels of dataset
#  @param positions: list of int, center position of drift
#  @param w_drift: list of int, width of each drift
#  @param length: int, total length of new stream
#  @Return drift_label: list of int, whether or not a point is labelled as drift
def get_drift_labels(positions, w_drift, length):
    non_overlap_int = []
    n_0 = positions[0] - w_drift[0]//2
    drift_label = n_0 * [0] + w_drift[0] * [1]
    for i in range(1, len(positions)):
        p_prev, p_curr = positions[i-1], positions[i]
        w_prev, w_curr = w_drift[i-1], w_drift[i]
        n_0 = p_curr - p_prev - w_prev // 2 - w_curr // 2
        non_overlap_int.append(n_0)
        drift_label += n_0 * [0] + w_curr * [1]
    all_pos = all(interv >= 0 for interv in non_overlap_int)
    if not all_pos:
        neg = [interv for interv in non_overlap_int if interv < 0]
        neg_i = non_overlap_int.index(neg[0])
        print('value: ', neg, '| index: ', neg_i)
        print('w_drift:', w_drift[neg_i - 1:neg_i + 2])
        print('positions:', positions[neg_i - 1:neg_i + 2])
    drift_label += [0] * (length - len(drift_label))
    return drift_label


#  Main method to execute script
def main():
    files = os.listdir(SOURCE_DIR)
    files.sort()
    selected_streams = files[2:8]
    anom_ints = []
    for s in selected_streams:
        _, y = get_arff_data_labels(os.path.join(SOURCE_DIR, s))
        anom_ints.append(find_anomaly_intervals(y))
    length = 229900
    n_drift = 20
    p_drift = 0.15
    p_drift_after = 0.5
    max_stream = len(selected_streams)-1
    trial_name = 'p_drift'
    anom_ints = anom_ints
    output_path, drift_label, streams, positions, seq_drift_after = run_generate_grad_stream_moa(
        selected_streams,
        p_drift,
        n_drift,
        length,
        p_drift_after,
        max_stream,
        anom_ints,
        trial_name
        )

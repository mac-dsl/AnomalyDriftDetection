import arff
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
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

class Stream:
    '''
    Object for a stream of data
    '''

    def __init__(self, filepath) -> None:
        filename = filepath.split("/")[-1]
        self.path = "/".join(filepath.split("/")[:-1])
        self.filename = "".join(filename.split(".")[:-1])
        self.data = None
        self.anomaly_labels = None
        self.drift_labels = None
        self.drift_params = None
        self.anomaly_intervals = None
        self.drift_intervals = None

        file_ext = filename.split(".")[-1].lower()
        if file_ext in {"csv", "out"}:
            data = pd.read_csv(filepath, header=None)
            self.data, self.anomaly_labels = self.__get_csv_data_labels(filepath)
            self.drift_labels = self.__get_drift_labels()
            self.length = len(self.anomaly_labels)
        elif file_ext == "arff":
            self.data, self.anomaly_labels = self.__get_arff_data_labels(filepath)
            self.drift_labels = self.__get_drift_labels()
            self.length = len(self.anomaly_labels)


    def to_arff(self, dir="."):
        data_labels = np.concatenate([self.data, self.anomaly_labels], axis=1)
        content = pd.DataFrame(data_labels)
        content = content.to_csv(header=False, index=False).strip("\n").split("\n")
        content = [f"{line},\n" for line in content]
        header = f"@relation '{dir}/{self.filename}'\n\n"
        header += "@attribute att1 numeric\n"
        header += "@attribute class {1.0, 0.0}\n\n"
        header += "@data\n\n"
        arff_filename = f"{self.filename}.arff"
        with open(arff_filename, "w") as output_file:
            output_file.writelines([header] + content)
        return arff_filename


    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, endn of interval to plot, default: sys.maxsize 
    def plot(self, start=0, end=sys.maxsize) -> None:
        end = min(end, self.length)
        fig, ax = plt.subplots(figsize=(10, 3))
        self.__plot_anomaly(
            ax, start, end, self.filename, size=8, show_label=True
        )
        fig.legend(loc='lower right')


    #  @param k: int, to indicate the kth anomaly to plot
    #             value range between 1 and n_drift
    #  @param w: int, half width of interval to view
    def plot_anomaly_k(self, k, w=1000) -> None:
        """
        Plot only drift surrounding the kth anomaly
        """
        anom_center = sum(self.anomaly_intervals[k]) // 2
        start = anom_center - w
        end = anom_center + w
        start = max(0, start)
        end = min(end, self.length)
        self.plot(start, end)


    #  @params X: int iterable, data points
    #  @params y: int iterable, anomaly labels
    #  @params ax: Axes object to plot graph
    #  @params start: int, start of interval to plot
    #  @params end: int, endn of interval to plot
    #  @params title: string, title of plot
    #  @params marker: string, marker type for plot, default '-' (line)
    #  @params size: int, fontsize, default=10
    def __plot_anomaly(
            self, ax, start=0, end=sys.maxsize, title="", marker="-", size=10, show_label=False
    ):
        """
        Plot the data with highlighted anomaly
        """
        start = max(0, start)
        end = min(self.data.shape[0], end)
        ax.plot(np.arange(start, end), self.data[start:end],
                f"{marker}b", label="_"*(not show_label) + "Non-Anomalous")
        if self.anomaly_intervals is None:
            self.__set_anomaly_intervals()
        anom_ints = [
            ai for ai in self.anomaly_intervals if
            (ai[0] < end and ai[0] > start) or
            (ai[1] < end and ai[1] > start) or
            (ai[0] < start and ai[1] > end)
        ]
        for (i, (anom_start, anom_end)) in enumerate(anom_ints):
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            label = "_"*(i + (not show_label)) + "Anomaly"
            ax.plot(
                np.arange(anom_start, anom_end),
                self.data[anom_start:anom_end],
                f"{marker}r",
                label=label
            )
        if len(title) > 0:
            ax.set_title(title, size=size)


    #  @param y: ndarray of shape (N,) corresponding to anomaly labels
    #  @Return list of lists denoting anomaly intervals in the form [start, end)
    def __set_anomaly_intervals(self):
        """
        Method to find the intervals where there is an anomaly
        """
        if sum(self.anomaly_labels) > 0:
            change_indices = np.where(np.diff(self.anomaly_labels, axis=0) != 0)[0]
            if len(change_indices) == 0:
                return []
            anom_intervals = []

            if self.anomaly_labels[change_indices[0]] == 0:
                i = 0
            else:
                i = 1
                anom_intervals.append([0, change_indices[0]+1])

            while i + 1 < len(change_indices):
                anom_intervals.append([change_indices[i]+1, change_indices[i+1]+1])
                i += 2

            if self.anomaly_labels[-1] == 1:
                anom_intervals.append([change_indices[-1]+1, len(self.anomaly_labels)])

            self.anomaly_intervals = anom_intervals
        
        else:
            self.anomaly_intervals = []


    def __get_csv_data_labels(self, filename: str):
        csv_content = pd.read_csv(filename, header=None)
        csv_len = csv_content.shape[0]
        data = csv_content[0].to_numpy().reshape(csv_len, 1)
        anomaly_labels = csv_content[1].to_numpy().reshape(csv_len, 1)
        return data.astype(float), anomaly_labels.astype(float)


    def __get_arff_data_labels(self, filename: str):
        """
        Find ndarray corresponding to data and labels from arff data

        filename : str
            Filename of arff data source
            
        Returns
        tuple 
            ndarrays corresponding to data (N, 1) and labels (N,)
            from arff data
        """
        arff_content = arff.load(f.replace(',\n', '\n') for f in open(filename, 'r'))
        arff_data = arff_content['data']
        data = np.array([i[:1] for i in arff_data])
        anomaly_labels = np.array([i[-1] for i in arff_data])
        anomaly_labels = anomaly_labels.reshape((len(anomaly_labels),1))
        return data.astype(float), anomaly_labels.astype(float)


    def __get_drift_labels(self):
        return np.zeros((len(self.anomaly_labels),1))

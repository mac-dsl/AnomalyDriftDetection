import arff
import os
import numpy as np 
import pandas as pd


## TODO: 
# - make Stream object contain data for stream and label
# - make Stream object able to convert to ARFF file

class Stream:
    '''
    Object for a stream of data
    '''

    def __init__(self, filename) -> None:
        self.filename = "".join(filename.split(".")[:-1])
        self.anomaly_labels = None
        self.drift_labels = None

        file_ext = filename.split(".")[-1].lower()
        if file_ext in {"csv", "out"}:
            data = pd.read_csv(filename, header=None)
            self.data, self.anomaly_labels = self.__get_csv_data_labels(filename)
        elif file_ext == "arff":
            self.data, self.anomaly_labels = self.__get_arff_data_labels(filename)


    #  @param y: ndarray of shape (N,) corresponding to anomaly labels
    #  @Return list of lists denoting anomaly intervals in the form [start, end)
    def find_anomaly_intervals(self):
        """
        Method to find the intervals where there is an anomaly
        """
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

        return anom_intervals
    

    #  @param y: ndarray of shape (N,) corresponding to drift labels
    #  @Return list of lists denoting drift intervals in the form [start, end)
    def find_drift_intervals(self):
        """
        Method to find the intervals where there is an drift
        """
        change_indices = np.where(np.diff(self.drift_labels, axis=0) != 0)[0]
        if len(change_indices) == 0:
            return []
        anom_intervals = []

        if self.drift_labels[change_indices[0]] == 0:
            i = 0
        else:
            i = 1
            anom_intervals.append([0, change_indices[0]+1])

        while i + 1 < len(change_indices):
            anom_intervals.append([change_indices[i]+1, change_indices[i+1]+1])
            i += 2

        if self.drift_labels[-1] == 1:
            anom_intervals.append([change_indices[-1]+1, len(self.drift_labels)])

        return anom_intervals


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
        data = arff_content['data']
        data = np.array([i[:1] for i in data])
        anomaly_labels = np.array([i[-1] for i in data])
        return data.astype(float), anomaly_labels.astype(float)

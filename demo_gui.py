# 
from util.stream import Stream
# importing the anomaly injection libraries
from util.anomaly_intervalsStream import *
from util.anomaly import CollectiveAnomaly, PointAnomaly, PeriodicAnomaly
from util.stream import Stream, DriftStream
from util.drift_generator import DriftGenerator

import matplotlib.pyplot as plt
import os
import yaml
from functools import partial

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tFont
from tkinter import filedialog
from tkinter import messagebox
from tkinter import tix
from PIL import Image


type_txt = ['point', 'collective', 'periodic']
##########################################################
# @param anomalyStream: class createAnomalyIntervals
# @config_param: config parameter, .yaml
def add_anomalies(anomalyStream, config_param):
    """
    Read config_param and add anomalies using util.anomaly_intervalsStream lib.
    """
    start=0
    pAnoms = get_anomaly_params(config_param)
    for step in config_param['anomaly_step']:
        anomalyStream.create_intervals(**step)
        num_input_anoms = step['num_intervals']
        anomalyStream.add_anomalies(*pAnoms[start:start+num_input_anoms])
        start += num_input_anoms
    return anomalyStream

def get_anomaly_params(config_param):
    """
    Read .yaml config to put params into util.anomaly_intervalsStream lib.
    """
    pAnoms = []
    for a_param in config_param['anomaly_params']:
        type = a_param['type']
        p_param = a_param.copy()
        del p_param['type']
        if type == 'point': pAnoms.append(PointAnomaly(**p_param))
        elif a_param['type'] == 'collective': pAnoms.append(CollectiveAnomaly(**p_param))
        elif a_param['type'] == 'periodic': pAnoms.append(PeriodicAnomaly(**p_param))            
        else:
            raise ValueError(
                f"Wrong type of anomalies, {a_param['type']} is not supported")

    return pAnoms


###########################################################################################
## Make a new sub-window and adjust anomaly configuration
class AnomalyConfig:
    def __init__(self, master, ind, len_interval):
        # self.window = type_win
        self.master = tk.Toplevel(master)
        self.frame = tk.Frame(self.master)
        self.ind = ind
        self.len_interval = len_interval
        self.dict_param = {}       

        tk.Label(self.frame, text=f'Interval {ind+1}/{len_interval}', width=20).grid(row=0)
        type_lb = tk.Label(self.frame, text='Type', width=20).grid(row=1, column=0)    
        type_combo = ttk.Combobox(self.frame, width=20, textvariable=type_txt, values=type_txt)
        type_combo.grid(row=1, column=1)
        type_combo.current(0)
        
        ## Button for set type
        type_btn = tk.Button(self.frame, text='Save Type', width=20, command=lambda: self.get_type(type_combo))
        type_btn.grid(row=1, column=2)
        # self.curr += 2
        self.frame.pack()

    def get_type(self, type_combo):
        '''
        Select type of anomaly. For each selection, additional info. would be shown.
        '''
        # global type_win, type_combo, dist
        self.type_num = type_combo.get()
        # print(type_num)
        dist = tk.IntVar()

        ## point and collective
        if self.type_num == 'point' or self.type_num =='collective':
            radio_btn1 = tk.Radiobutton(self.frame, text='Uniform', value=1, variable=dist, command=lambda: self.get_dist(1))
            radio_btn1.grid(row=2, column=0)

            radio_btn2 = tk.Radiobutton(self.frame , text='Gaussian', value=2, variable=dist, command=lambda: self.get_dist(2))
            radio_btn2.grid(row=2, column=1)

            radio_btn3 = tk.Radiobutton(self.frame , text='Skew',  value=3, variable=dist, command=lambda: self.get_dist(3))
            radio_btn3.grid(row=2, column=2)

        elif self.type_num =='periodic':
            ## percentage
            per_lb = tk.Label(self.frame, text='Percentage').grid(row=4, column=0)
            self.per = tk.StringVar()
            self.per.set(0.01)
            per_entry = tk.Entry(self.frame, textvariable=self.per)
            per_entry.grid(row=4, column=1)
            per_entry.insert(0, 0.01)
            ## num_values
            num_val_lb = tk.Label(self.frame, text='Num. Values').grid(row=4, column=2)
            self.num_val = tk.StringVar()
            self.num_val.set(100)
            num_val_entry = tk.Entry(self.frame, textvariable=self.num_val)
            num_val_entry.grid(row=4, column=3)
            num_val_entry.insert(0, 100)

            self.len_seq = tk.StringVar()
            self.len_seq.set(24)

            len_seq_lb = tk.Label(self.frame, text='Length').grid(row=4, column=4)
            len_seq_entry = tk.Entry(self.frame, textvariable=self.len_seq)
            len_seq_entry.grid(row=4, column=5)
            len_seq_entry.insert(0, 24)

    def get_dist(self, dist_num):
        """
        In case of choosing 'point' or 'collective', users need to choose distribution of anomalies.
        It shows additional entity widget corresponding selected distribution.
        """
        print(dist_num, self.type_num)

        ## Distribution
        if dist_num ==1:
            txt_1, txt_2 = 'Upperbound', 'Lowerbound'
            init_1, init_2 = '0.5', '-0.5'
        elif dist_num ==2:
            txt_1, txt_2 = 'mu', 'sigma'
            init_1, init_2 = '1', '0.1'
        elif dist_num ==3:
            txt_1, txt_2 = 'alpha', 'Upperbound'
            init_1, init_2 = '0.5', '0.5'

        lb1 = tk.Entry(self.frame, justify='center')
        lb1.grid(row=3, column=0)
        lb1.insert(0, txt_1)
        lb1.configure(state='readonly')
        self.var1 = tk.StringVar()
        self.var1.set(init_1)
        var1_entry = tk.Entry(self.frame, textvariable=self.var1)
        var1_entry.grid(row=3, column=1)
        # var1_entry.insert(0, init_1)

        lb2 = tk.Entry(self.frame, justify='center')
        lb2.grid(row=3, column=2)
        lb2.insert(0, txt_2)
        lb2.configure(state='readonly')
        self.var2 = tk.StringVar()
        self.var2.set(init_2)
        var2_entry = tk.Entry(self.frame, textvariable=self.var2)
        var2_entry.grid(row=3, column=3)
        # var2_entry.insert(0, init_2)

        ## percentage
        per_lb = tk.Label(self.frame, text='Percentage').grid(row=4, column=0)
        self.per = tk.StringVar()
        self.per.set(0.01)
        per_entry = tk.Entry(self.frame, textvariable=self.per)
        per_entry.grid(row=4, column=1)
        # per_entry.insert(0, 0.01)

        ## num_values
        num_val_lb = tk.Label(self.frame, text='Num. Values').grid(row=4, column=2)
        self.num_val = tk.StringVar()
        self.num_val.set(100)
        num_val_entry = tk.Entry(self.frame, textvariable=self.num_val)
        num_val_entry.grid(row=4, column=3)
        # num_val_entry.insert(0, 100)

        self.len_seq = tk.StringVar()
        self.len_seq.set(24)
        len_seq_lb = tk.Label(self.frame, text='Length').grid(row=4, column=4)
        len_seq_entry = tk.Entry(self.frame, textvariable=self.len_seq)
        len_seq_entry.grid(row=4, column=5)
        # len_seq_entry.insert(0, 24)

        if self.type_num =='point':
            len_seq_entry.configure(state='disabled')
        else:
            len_seq_entry.configure(state='normal')


        ## Button for save    
        save_anomaly_btn = tk.Button(self.frame, text='Save Setting', width=20, command=lambda: self.save_dist(dist_num))
        save_anomaly_btn.grid(row=5, column=3)    

    def save_dist(self, dist_num):
        """
        Save the anomaly injection parameters
        """

        if self.type_num =='point': self.dict_param = {'type':'point'}
        elif self.type_num =='collective': self.dict_param = {'type':'collective'}

        if dist_num ==1:
            self.dict_param['distribution'] = 'uniform'
            self.dict_param['upperbound'] = float(self.var1.get())
            self.dict_param['lowerbound'] = float(self.var2.get())

        elif dist_num ==2:
            self.dict_param['distribution'] = 'gaussian'
            self.dict_param['mu'] = float(self.var1.get())
            self.dict_param['sigma'] = float(self.var2.get())

        elif dist_num ==3:
            self.dict_param['distribution'] = 'skew'
            self.dict_param['skew'] = float(self.var1.get())
            self.dict_param['upperbound'] = float(self.var2.get())

        self.dict_param['percentage'] = float(self.per.get())
        self.dict_param['num_values'] = int(self.num_val.get())
        if self.type_num ==1:
            self.dict_param['length'] = int(self.len_seq.get())

        # print(self.dict_param)
        # self.master.dict_param = self.dict_param
        self.master.destroy()
        
    def display_window(self):
        """
        Holding the main window to get a set of configuration
        """
        self.master.wait_window()
        return self.dict_param

######################################################################
## Make a new window and select interval and gap for anomaly injection
## Based on interval, new window would be popped and get additional parameters 
class IntervalConfig(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = tk.Toplevel(master)
        self.frame = tk.Frame(self.master)
        self.master.title('Config Anomaly Injection')
        # self.master = master
        # self.frame = tk.Frame(self.master)

        self.config_param = {}

        self.config_rows = 0
        ## In this case, we only apply single-step    
        interval_lb = tk.Label(self.frame, text='Interval').grid(row=self.config_rows, column=0)
        # self.interval = tk.StringVar()
        # self.interval.set(1)
        self.interval_en =tk.Entry(self.frame)
        self.interval_en.grid(row=self.config_rows, column=1)    
        self.interval_en.insert(0, 1)    

        gap_lb=tk.Label(self.frame, text='Gap Size').grid(row=self.config_rows, column=2)
        # self.gap_size=tk.StringVar()
        # self.gap_size.set(0)
        self.gap_en=tk.Entry(self.frame)
        self.gap_en.grid(row=self.config_rows, column=3)
        self.gap_en.insert(0, 0)

        ## Button for set interval
        self.interval_btn = tk.Button(self.frame, text='Save Interval', width=22, command=self.set_interval)
        self.interval_btn.grid(row=self.config_rows, column=4)
        self.config_rows +=1

        self.dict_param = {}
        self.frame.pack()

    def set_interval(self):
        """
        After finishing editing of interval, it generates new pop-up window to get corresponding parameters
        """
        ## save entry numbers and show anomaly_params        
        self.interval_btn.configure(state='disable')

        ## Anomaly Params
        len_interval = int(self.interval_en.get())
        gap_size = int(self.gap_en.get())
        self.config_param['anomaly_step'] = []
        self.config_param['anomaly_step'].append({'num_intervals':len_interval, 'gap_size':gap_size})

        self.config_param['anomaly_params'] =[]
        print('len:', len_interval)

        for i in range(len_interval):
            # print('start:', i)
            tk.Label(self.frame, text='Anomaly Params').grid(row=self.config_rows)            
            self.set_type(i, len_interval)
            self.config_rows +=1
            # print('ROW:', self.config_rows)

        self.inject_btn = tk.Button(self.frame, text='Apply Params', width=22, command=self.inject_anomaly)
        self.inject_btn.grid(row=self.config_rows, column=4)

    def set_type(self, ind, len_interval):
        """
        Show new pop-up window to get corresponding parameters (for each interval)
        """
        self.dict_param = AnomalyConfig(self, ind, len_interval).display_window()
        tk.Label(self.frame, text=f'{self.dict_param}').grid(row=self.config_rows, column=1, columnspan=5)
        # print('Only??', self.dict_param)
        self.config_param['anomaly_params'].append(self.dict_param)
        return 0
    
    def inject_anomaly(self):
        self.master.destroy()

    def display_window(self):
        self.master.wait_window()
        return self.config_param

#######################################################################################
## For drift generation: make a new window and get parameters
class DriftConfig(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = tk.Toplevel(master)
        self.frame = tk.Frame(self.master)
        self.master.title('Config Drift Generation')
        # self.master = master
        # self.frame = tk.Frame(self.master)

        self.config_param = {}

        self.config_rows = 0
        ## In this case, we only apply single-step    
        n_drift_lb = tk.Label(self.frame, text='Num. Drift').grid(row=self.config_rows, column=0)
        self.n_drift_en =tk.Entry(self.frame)
        self.n_drift_en.grid(row=self.config_rows, column=1)    
        self.n_drift_en.insert(0, 5)    

        p_drift_lb=tk.Label(self.frame, text='Drift Period').grid(row=self.config_rows, column=2)
        self.p_drift_en=tk.Entry(self.frame)
        self.p_drift_en.grid(row=self.config_rows, column=3)
        self.p_drift_en.insert(0, 0.01)

        p_before_lb=tk.Label(self.frame, text='Before Anomalies').grid(row=self.config_rows, column=4)
        self.p_before_en=tk.Entry(self.frame)
        self.p_before_en.grid(row=self.config_rows, column=5)
        self.p_before_en.insert(0, 0.1)

        self.config_rows +=1
        ## Button for set interval
        self.drift_config_btn = tk.Button(self.frame, text='Save Params', width=22, command=self.set_config)
        self.drift_config_btn.grid(row=self.config_rows, column=5)
        self.config_rows +=1

        self.dict_param = {}
        self.frame.pack()

    def set_config(self):
        """
        To save drift configuration
        """
        ## save entry numbers and show anomaly_params        
        self.drift_config_btn.configure(state='disable')

        n_drift = int(self.n_drift_en.get())
        p_drift = float(self.p_drift_en.get())
        p_before = float(self.p_before_en.get())

        self.config_param['drift_params'] = {'n_drift':n_drift, 'p_drift':p_drift, 'p_before':p_before}
        self.master.destroy()

    def display_window(self):
        self.master.wait_window()
        return self.config_param

class Demo(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.frame = tk.Frame(self.master)

        self.large_font = tFont.Font(family='Arial', size=30)
        self.medium_font = tFont.Font(family='Arial', size=20)
        self.font = tFont.Font(family='Arial', size=15)

        label = tk.Label(self.frame, text='CanGene', font=self.large_font)
        label.pack()

        self.moa_path = '../../moa-release-2023.04.0/lib'
        self.drift_dir = '/data/synthetic'
        self.sub_dir = 'demo'
        self.source_dir ='./data/benchmark/weather'
        self.anomaly_path = None

        ##########################################################
        # Frame
        self.Menu = tk.Frame(self.frame)
        # Menu.grid(row=0, column=0)
        self.Menu.pack(side='top', fill='both', expand=False, padx=5, pady=5)

        self.Graph = tk.Frame(self.frame)
        self.Graph.pack(side='top', fill='both', expand=True, padx=5, pady=5)

        # self.canvas = tk.Canvas(self.Menu)
        # self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
# 
# 
        # self.scrollbar= ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
# 
        # self.canvas.configure(yscrollcommand=self.scrollbar.set)
# 
        # self.canvas.bind(
            # "<Configure>",
            # lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        # )
# 
        # self.Graph = tk.Frame(self.canvas)
        # self.canvas.create_window((0,0), window=self.Graph, anchor='nw')
        # scrollbar.pack(side='right', fill='y')

        ##########################################################
        # Menu Frame
        self.load_btn = tk.Button(self.Menu, text='Load Data', width=18, command=self.load_ts)
        self.load_btn.grid(row=0, column=0)

        # self.show_btn = tk.Button(self.Menu, text='Show All', width=18, state='disable', command=lambda: self.show_ts(0))
        # self.show_btn.grid(row=0, column=1)

        self.yaml_btn = tk.Button(self.Menu, text='Load YAML', width=18, state='normal', command=self.load_yaml)
        self.yaml_btn.grid(row=0, column=1)

        self.anomaly_btn = tk.Button(self.Menu, text='Anomaly Injection', width=18, state='disable', command=self.anomaly_injection)
        self.anomaly_btn.grid(row=0, column=2)

        self.drift_btn = tk.Button(self.Menu, text='Drift Generation', width=18, state='disable', command=self.drift_generation)
        self.drift_btn.grid(row=0, column=3)        

        self.clear_btn = tk.Button(self.Menu, text='Clear All', width=18, state='normal', command=self.clear_all)
        self.clear_btn.grid(row=0, column=4)        

        # info = tk.StringVar()
        # msg_lbl = tk.Label(Menu, textvariable=info, font=font, width=125, height=2 ,foreground='black', background='#CACACA')
        # msg_lbl.grid(row=1, columnspan=4)

        self.file_txt = tk.StringVar()
        file_lbl = tk.Label(self.Menu, textvariable=self.file_txt, font=self.font)
        file_lbl.grid(row=2, columnspan=5)


        ##########################################################
        ## Stream
        self.DataStreams = []
        self.index_stream = 0
        self.AnomalyStreams = []
        self.show_lbls = []
        self.frame.pack()

    def load_ts(self):
        """
        Load time-series into Stream
        """
        
        f = filedialog.askopenfile(
            initialdir=f'{os.getcwd()}/data/benchmark',
            filetypes=(('arff files', '*.arff'), ('csv files', '*.csv'))
        )
        self.source_dir = f.name
        self.DataStreams.append(Stream(f.name))

        ## Draw graph and save temporal img
        self.DataStreams[self.index_stream].plot()
        plt.savefig(f'stream_{self.index_stream}.png', dpi=100)

        txt_msg = 'Loaded Files:'
        for st in self.DataStreams:
            txt_msg += '\n' + st.filename

        self.file_txt.set(txt_msg)    
        self.index_stream +=1

        self.show_ts(0)

        # print('Index:', index_stream, len(DataStreams))
        # self.show_btn.configure(state='normal')
        self.anomaly_btn.configure(state='normal')        
        self.drift_btn.configure(state='normal')
        self.yaml_btn.configure(state='disable')
        
        # self.popup_win(self.index_stream-1)

    #############################################################
    ## Show and Close the loaded popup image 
    def close_win(self):
        global nw
        nw.destroy()

    def popup_win(self):
        """
        Show loaded time-series in a new window
        """
        global nw
        nw=tk.Tk()

        nw.title('Selected file')
        nw.geometry('1050x330')

        # graph_img = tk.PhotoImage(file=f'stream_{index_stream}.png', master=nw)
        graph_img = tk.PhotoImage(file='drift.png', master=nw)
        btn = tk.Button(nw, image=graph_img, command=self.close_win)
        btn.pack()
        nw.mainloop()


    def show_ts(self, sel):
        """
        Show all loaded data stream
        """
        # print(f'Index={self.index_stream}, len_lbl={len(self.show_lbls)}')
        
        if self.index_stream > len(self.show_lbls):
            for i in range(self.index_stream-len(self.show_lbls)):
                self.show_lbls.append(tk.Label(self.Graph))            
                self.show_lbls[-1].grid(row=len(self.show_lbls)-1, columnspan=4)
                # print(f'Len={len(self.show_lbls)}')

        for i in range(self.index_stream):
            if sel ==0: show_img = tk.PhotoImage(file=f'stream_{i}.png', master=self.master)
            elif sel >=1: show_img = tk.PhotoImage(file=f'stream_{i}_a.png', master=self.master)
            self.show_lbls[i].configure(image=show_img)
            self.show_lbls[i].image = show_img

        if sel>1:
            show_img = tk.PhotoImage(file='drift.png', master=self.master)
            self.show_lbls.append(tk.Label(self.Graph))
            self.show_lbls[-1].grid(row=len(self.show_lbls)-1, columnspan=4)
            self.show_lbls[-1].configure(image=show_img)
            self.show_lbls[-1].image = show_img

    def anomaly_injection(self):
        """
        Call a new window to get anomaly injection configuration
        """
        self.config_param = IntervalConfig(self).display_window()

        self.anomaly_path = os.path.dirname(self.source_dir)

        self.anomaly_path = filedialog.askdirectory(parent=self.master, initialdir=self.anomaly_path, title='Choose a path to save data with anomalies')
        for i, dataStream in enumerate(self.DataStreams):
            self.AnomalyStreams.append(createAnomalyIntervals(dataStream))
            self.AnomalyStreams[i] = add_anomalies(self.AnomalyStreams[i], self.config_param)
            dataStream.filename = f"{dataStream.filename}_anomaly"
            dataStream.to_arff(self.anomaly_path)
            dataStream.plot()
            plt.savefig(f'stream_{i}_a.png', dpi=100)

        self.show_ts(1)

    def drift_generation(self):
        """
        Call a new window to get drift configuration
        """
        self.config_param = DriftConfig(self).display_window()

        if self.moa_path == None:
            self.moa_path = filedialog.askdirectory(parent=self.master, initialdir=f'{os.getcwd()}', title='Select a path of MOA program')

        source_stream = self.DataStreams
        dir_path = filedialog.askdirectory(parent=self.master, initialdir=f'{os.getcwd()}{self.drift_dir}', title='Please select a directory')

        self.drift_dir = os.path.dirname(dir_path)
        self.config_param['drift_params']['sub_dir'] = dir_path[len(self.drift_dir)+1:]

        if self.anomaly_path == None:
            source_dir = os.path.dirname(self.source_dir)
        else:
            source_dir = self.anomaly_path

        print(f'Moa: {self.moa_path}')
        print(f'Drift: {self.drift_dir}')
        print(f'Source: {source_dir}')
        print('Params:', self.config_param['drift_params'])

        g = DriftGenerator(source_dir, self.drift_dir, self.moa_path, selected_streams=source_stream)

        length = self.DataStreams[0].length

        ds = g.run_generate_grad_stream_moa(
            length=length,
            dataset='Data',
            mode=0,
            **self.config_param['drift_params']
        )

        ds.plot_drift()
        plt.savefig('drift.png', dpi=100)

        self.show_ts(2)
        self.popup_win()

    def load_yaml(self):
        """
        Run anomaly injection and drift generation using .yaml file
        """
        
        f = filedialog.askopenfile(
            initialdir=f'{os.getcwd()}',
            filetypes=(('yaml files', '*.yaml'),)
        )
        with open(f.name) as fy:
            self.config_param = yaml.load(fy, Loader=yaml.FullLoader)

        ## Load Files
        selected_files = self.config_param['source_files']
        dir = self.config_param['source_dir']
        self.DataStreams = []
        for i, file in enumerate(selected_files):
            self.DataStreams.append(Stream(f"{dir}/{file}"))
            self.DataStreams[i].plot()            
            plt.savefig(f'stream_{i}.png', dpi=100)
            self.index_stream +=1

        txt_msg = 'Loaded Files:'
        for st in self.DataStreams:
            txt_msg += '\n' + st.filename

        self.file_txt.set(txt_msg)    
        
        for i, dataStream in enumerate(self.DataStreams):
            self.AnomalyStreams.append(createAnomalyIntervals(dataStream))
            self.AnomalyStreams[i] = add_anomalies(self.AnomalyStreams[i], self.config_param)
            dataStream.filename = f"{dataStream.filename}_anomaly"
            dataStream.to_arff(self.config_param['source_dir'])
            dataStream.plot()
            plt.savefig(f'stream_{i}_a.png', dpi=100)
        # show_config()

        moa_path = self.config_param['moa_path']
        source_streams = self.DataStreams
        drift_dir = os.getcwd()+ self.config_param['drift_dir']
        g = DriftGenerator(dir, drift_dir, moa_path, selected_streams=source_streams)

        length = self.DataStreams[0].length
        dataset = 'Data'

        ds = g.run_generate_grad_stream_moa(
            length=length,
            dataset = dataset,
            mode=0,  # 0 for variable drift widths and positions, 1 for uniform
            **self.config_param['drift_params']
        )

        ds.plot_drift()
        plt.savefig('drift.png', dpi=100)

        self.show_ts(2)
        self.popup_win()

    def clear_all(self):
        for i in range(self.index_stream):
            self.show_lbls[i].configure(image='')
        self.DataStreams = []
        self.index_stream = 0
        self.AnomalyStreams = []
        self.show_lbls = []
        self.frame.pack()

        self.file_txt.set('')

        self.load_btn.configure(state='normal')
        self.anomaly_btn.configure(state='disable')   
        self.drift_btn.configure(state='disable')
        self.yaml_btn.configure(state='normal')       

def main():
    ##########################################################
    root = tk.Tk()

    root.geometry('1200x900')
    root.title('CanGene')
    app = Demo(root)

    root.mainloop()

if __name__ == "__main__":
    main()
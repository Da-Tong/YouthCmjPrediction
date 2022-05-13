import os
import pandas as pd
import numpy as np


def ResampleLinear1D(original, targetLen):
    '''
    upsample or downsample data

    Parameters
    ----------
    original : list, dataframe or array, data that want to resample (allow multiple columns)
        
    targetLen : int, target length

    Returns
    -------
    interp : array, resampled array

    '''
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem_tmp = (index_arr - index_floor).reshape(-1, 1) #Remain
    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    
    if len(original.shape) > 1:
        index_rem_size = original.shape[1]
    else:
        index_rem_size = 1
        val1 = val1.reshape(-1, 1)
        val2 = val2.reshape(-1, 1)
    index_rem = np.zeros((len(index_rem_tmp), index_rem_size))
    index_rem += index_rem_tmp
    
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp


class getViconData():
    
    def __init__(self, path, sub, testNum, position, testType="cmj"):
        '''
        data description

        Parameters
        ----------
        path : str, data path
        
        sub : int, subject number
        
        position : str, acc postion
        
        testType : str, cmj or ccmj
        
        testNum : int, trial number

        '''
        self.path = path
        self.sub = sub
        self.position = position
        self.testType = testType
        self.testNum = testNum
        self.columnName = ["ax_m/s/s", "ay_m/s/s", "az_m/s/s"]
        
    def __call__(self):
        '''
        return sepcific column data

        Returns
        -------
        data : array of data

        '''
        fn = self.get_file_name()
        data = pd.read_csv(fn)
        data = np.array(data.loc[:, self.columnName])
        # data = self.cut_data(data)
        
        return data
        
    def get_file_name(self):
        '''
        get file name of acc data
        
        '''
        sub_imu = "sb" + "{:02d}".format(self.sub)
        if self.testType == 'cmj':
            fn_imu = os.path.join(self.path, sub_imu,
                                  self.testType.upper() + str(self.testNum),
                                  self.position + '.csv')
        elif self.testType == 'ccmj':
            if self.testNum == 1:
                fn_imu = os.path.join(self.path, sub_imu,
                                      self.testType.upper(), self.pos + '.csv')
            else:
                fn_imu = 1
        return fn_imu
    
    def cut_data(self, data):
        res_acc = get_res_acc(data)
        max_idx = np.argmax(res_acc)
        
        cut_start = max(max_idx - 200, 0)
        cut_end = min(max_idx + 200, len(data))
        
        return data[cut_start:cut_end, :]
    
    
class getGoodix():
    
    def __init__(self, path, sub, testNum):
        self.path = path
        self.sub = sub
        self.testNum = testNum
        
    def __call__(self):
        fn = self.get_file_name()
        if fn:
            data = pd.read_csv(fn)
        else:
            return []
        
        data = self.get_acc(data)
        data = self.cut_data(data)
        return data
    
    def get_file_name(self):
        '''
        get file name of acc data
        
        '''
        subName = "sub{:03d}".format(self.sub)
        testName = "t{:03d}.csv".format(self.testNum)

        fn = os.path.join(self.path, subName,  testName)
        if os.path.isfile(fn):
            return fn
        else:
            print(f"{fn} is not exist")
            return []    
    
    def get_acc_one_axis(self, data, axis, samp_rate=25):
        """
        get timestamp continueous & sampling rate 25 acc of specific axis
        
        if lost data, fill with zero
        
        if lost sampling rate, resample to 25

        Parameters
        ----------
        data : Goodix DataFrame
            
        axis : str, x, y or z

        Returns
        -------
        1D int array with acc

        """
        axis_name = f"ACC_{axis.upper()}"
        acc_array = ResampleLinear1D(data.loc[0][axis_name].split(','),
                                     samp_rate)
        time_old = int(round(data.loc[0].TIMESTAMP / 1000, 0))
        
        for idx in range(1, len(data)):
            data_buf = data.loc[idx]
            time_new = int(round(data_buf.TIMESTAMP / 1000, 0))
            time_diff = time_new - time_old
            time_old = time_new
            
            if time_diff == 1:
                tmp_acc = data_buf[axis_name].split(',')
                acc_array = np.append(acc_array,
                                      ResampleLinear1D(tmp_acc, samp_rate),
                                      axis=0)
                
            elif time_diff > 1:
                acc_array = np.append(acc_array, np.ones(25) * acc_array[-1], 
                                      axis=0)
                
            elif time_diff < 1:
                print(f"idx: {idx} timestamp error!")
                continue
        
        return acc_array.reshape(-1, )
    
    def get_acc(self, data):
        """
        get 3D acc

        Parameters
        ----------
        data : acc dataFrame

        Returns
        -------
        acc: 3D float array

        """
        acc = []
        for axis in ["x", "y", "z"]:
            acc.append(self.get_acc_one_axis(data, axis) / 512 * 9.8)
        
        return  np.array(acc).T
    
    def cut_data(self, data):
        res_acc = get_res_acc(data[:, :2])
        max_idx = np.argmax(res_acc)
        
        cut_start = max(max_idx - 40, 0)
        cut_end = min(max_idx + 40, len(data))
        
        return data[cut_start:cut_end, :]
    
class getNaxsen():
    
    def __init__(self, fn, fs):
        self.fn = fn
        self.fs = fs
        
    def __call__(self):
        if os.path.isfile(self.fn):
            data = self.get_acc(self.fn)
        else:
            print(self.fn, 'not exist!')
            return []
        
        data = self.cut_data(data)
        return data
    
    def get_acc(self, fn):
        data = []
        with open (fn, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == "#":
                    continue
                elif line.split(',')[0].startswith("acc"):
                    col_name = line.split(',')
                    col_accx = col_name.index("accX")
                    col_accy = col_name.index("accY")
                    col_accz = col_name.index("accZ")
                else:
                    accx = line.split(',')[col_accx]
                    accy = line.split(',')[col_accy]
                    accz = line.split(',')[col_accz]
                    data.append([accx, accy, accz])
        return 9.8 * np.array(data).astype(float)
    
    def cut_data(self, data):
        # res_acc = get_res_acc(data)
        max_idx = np.argmin(data[:, 0])
        
        cut_start = int(max(max_idx - 2 * self.fs, 0))
        cut_end = int(min(max_idx + 2 * self.fs, len(data)))
        
        return data[cut_start:cut_end, :]
        
    
class getOh1():
    '''
    from nana's parser
    
    '''
    
    def __init__(self, path, sub, testNum):
        self.path = path
        self.sub = sub
        self.testNum = testNum
        
    def __call__(self):
        fn = self.get_file_name()
        if fn:
            data = pd.read_csv(fn)
        else:
            return []
        
        data = self.get_acc(data)
        data = self.cut_data(data)
        return data
    
    def get_file_name(self):
        '''
        get file name of acc data
        
        '''
        subName = "sub{:03d}".format(self.sub)
        testName = "t{:03d}".format(self.testNum)
        fn = os.path.join(self.path, subName,  testName, "iOS_PolarTest_0.csv")
        if os.path.isfile(fn):
            return fn
        else:
            print(f"{fn} is not exist")
            return []    
    
    def transfer(self, data):
        data = data[1:-1]
        if len(data) > 1:
            data = data.split(',')
            data = np.asarray(data, dtype='int')
            return data
        else:
            return []

    def get_acc(self, data, samp_rate=50):        
        HR = []
        PPG_1 = []
        PPG_2 = []
        PPG_3 = []
        AMB = []
        ACC_X = []
        ACC_Y = []
        ACC_Z = []
        PPG_len = []
        ACC_len = []
        _prev_timestamp = -1
                
        buf_HR = 0
        buf_PPG = 0
        buf_ACC = 0
        
        time_start = data.loc[0]['TIMESTAMP']
        time_end = data.loc[len(data)-1]['TIMESTAMP']
        resample_len = (time_end - time_start) * samp_rate
        
        for i in range(len(data)):
            data_buf = data.loc[i]
            _timestamp = data_buf['TIMESTAMP']
            _HR = self.transfer(data_buf['HR'])
            _PPG_1 = self.transfer(data_buf['PPG_1'])
            _PPG_2 = self.transfer(data_buf['PPG_2'])
            _PPG_3 = self.transfer(data_buf['PPG_3'])
            _AMB = self.transfer(data_buf['AMBIENT_1'])
            _ACC_X = self.transfer(data_buf['ACC_X'])
            _ACC_Y = self.transfer(data_buf['ACC_Y'])
            _ACC_Z = self.transfer(data_buf['ACC_Z'])
            
            # === Deal w/ Data Loss Frame [HR] ===
            if _prev_timestamp != -1 and _timestamp - _prev_timestamp > 1:
                HR.extend(['NaN'] * (_timestamp - _prev_timestamp - 1))
                print('=== Dealing with timestamp drop issues ===')
                print('  - Add 1 [NaN] to HR after timestamp', _prev_timestamp, '\n')
    
            # === Deal w/ Data Loss Frame ===
            if len(_PPG_1) < 60:
            
                # === Deal w/ HR Loss Separately ===
                if len(_HR) == 0:
                    buf_HR += 1
                else:
                    HR.extend(['NaN'] * buf_HR)
                    print('=== Dealing with data loss issues ===')
                    print('  - Add', buf_HR, '[NaN]s to HR. \n')
                    buf_HR = 0
                
                # === Last Frame of file ===
                if i == len(data) - 1:
                    buf_PPG = round(buf_PPG / 18) * 18
                    buf_ACC = round(buf_ACC / 36) * 36
                    
                    HR.extend(['NaN'] * buf_HR)
                    PPG_1.extend(['NaN'] * buf_PPG)
                    PPG_2.extend(['NaN'] * buf_PPG)
                    PPG_3.extend(['NaN'] * buf_PPG)
                    AMB.extend(['NaN'] * buf_PPG)
                    ACC_X.extend(['NaN'] * buf_ACC)
                    ACC_Y.extend(['NaN'] * buf_ACC)
                    ACC_Z.extend(['NaN'] * buf_ACC)
                    buf_HR = 0
                    print('=== Dealing with data loss issues ===')
                    print('  - Add', buf_HR, '[NaN]s to HR. ')
                    print('  - Add', buf_PPG, '[NaN]s to PPG. ')
                    print('  - Add', buf_ACC, '[NaN]s to ACC. \n')
    
                else:
                    buf_PPG += (135 - len(_PPG_1))
                    buf_ACC += (51 - len(_ACC_X))
    
            elif buf_PPG != 0:
                buf_PPG = round(buf_PPG / 18) * 18
                buf_ACC = round(buf_ACC / 36) * 36
                
                PPG_1.extend(['NaN'] * buf_PPG)
                PPG_2.extend(['NaN'] * buf_PPG)
                PPG_3.extend(['NaN'] * buf_PPG)
                AMB.extend(['NaN'] * buf_PPG)
                ACC_X.extend(['NaN'] * buf_ACC)
                ACC_Y.extend(['NaN'] * buf_ACC)
                ACC_Z.extend(['NaN'] * buf_ACC)
    
                print(_timestamp)
                print('=== Dealing with data loss issues ===')
                print('  - Add', buf_PPG, '[NaN]s to PPG. ')
                print('  - Add', buf_ACC, '[NaN]s to ACC. \n')
                
                buf_ACC = 0
                buf_PPG = 0
    
            # === Data Append ===
            HR.extend(_HR)
            PPG_1.extend(_PPG_1)
            PPG_2.extend(_PPG_2)
            PPG_3.extend(_PPG_3)
            AMB.extend(_AMB)
            ACC_X.extend(_ACC_X)
            ACC_Y.extend(_ACC_Y)
            ACC_Z.extend(_ACC_Z)
            PPG_len.append(len(_PPG_1))
            ACC_len.append(len(_ACC_X))
        
            _prev_timestamp = _timestamp
        
        ACC_X = ResampleLinear1D(ACC_X, resample_len) / 1024 * 9.8
        ACC_Y = ResampleLinear1D(ACC_Y, resample_len) / 1024 * 9.8
        ACC_Z = ResampleLinear1D(ACC_Z, resample_len) / 1024 * 9.8
        
        return np.array([ACC_X.reshape(-1, ),
                         ACC_Y.reshape(-1, ),
                         ACC_Z.reshape(-1, )]).T
    
    def cut_data(self, data):
        res_acc = get_res_acc(data)
        max_idx = np.argmax(res_acc)
        
        cut_start = max(max_idx - 50, 0)
        cut_end = min(max_idx + 50, len(data))
        
        return data[cut_start:cut_end, :]
    
 
def get_res_acc(data):
    # data = np.array([data[:, 0], data[:, 2]]).T
    return (np.sqrt((data **2).sum(axis=1)) / 9.8) - 1
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    path_goodix = "data/Goodix"
    path_oh1 = "data/OH1"
    
    sub = 21
    test = 1
    
    getData = getOh1(path_oh1, sub, test)
    data_oh1 = getData()
    if len(data_oh1) > 1:
        fig, ax = plt.subplots()
        
        ax.plot(data_oh1)
        ax.plot(get_res_acc(data_oh1), '--')
        ax.legend(["x", "y", "z", "res"])
        
        ax.set_title(f"OH1, sub: {sub}, test: {test}")
        
    getData = getGoodix(path_goodix, sub, test)
    data_goodix = getData()
    if len(data_goodix) > 1:
        fig, ax = plt.subplots()
        
        ax.plot(data_goodix)        
        ax.plot(get_res_acc(data_goodix), '--')
        ax.legend(["x", "y", "z", "res"])
        ax.set_title(f"Goodix, sub: {sub}, test: {test}")
            
           
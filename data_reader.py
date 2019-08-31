import os
import numpy as np
import copy
import time

inerval = 30 # the interval of the time sequence
fft_n = 16 # number of fft coefficients
dis_rate = 20 # rate of area division
maxlens = [7998,7938,6905,8095,6238,6627,7985,7274,8972,8051,8202,7816,7963,8065,7441]

def read_one_file(filename):
    one_data = open(filename,'r')
    datax = []
    datay = []
    one_data.readline()
    line = one_data.readline()
    # check = (line.split()[3]=="Coral")
    # some files may have some extra spaces. They will affact the split
    if line.split()[3]=="Coral":
        begin = 8
    else:
        begin = 6

    line = line.split()[begin:begin+5]
    seq_start = eval(line[0])
    start = eval(line[0]) - seq_start
    end = eval(line[2]) - seq_start
    x0 = eval(line[3])
    y0 = eval(line[4])

    current_time = 0
    while current_time < end:
        datax.append(float(int(x0)/dis_rate))
        datay.append(float(int(y0)/dis_rate))
        current_time += inerval

    for l in one_data.readlines():
        l = l.split()[begin:begin+5] # fixation start, duration, end, x, y
        start = eval(l[0]) - seq_start
        x1 = float(int(eval(l[3]))/dis_rate)
        y1 = float(int(eval(l[4]))/dis_rate)
        while current_time < start: # points between two fixation
            datax.append(float(int(x1-(x1-x0)*(end-current_time)/(start-end))/dis_rate))
            datay.append(float(int(y1-(y1-y0)*(end-current_time)/(start-end))/dis_rate))
            current_time += inerval

        end = eval(l[2]) - seq_start
        x0 = x1
        y0 = y1
        while current_time < end:
            datax.append(x0)
            datay.append(y0)
            current_time += inerval
    # if check:
    #     print filename
    #     print begin
    #     print line
    #     print datax[:20]
    return datax, datay


def read_one_video(pos_list, neg_list):
    data_seq = []
    data_label = []
    data_name = []
    for filename in pos_list:
        datax, datay = read_one_file(filename)
        datax = np.fft.rfft(datax)[:fft_n] # if you don't need fft, just comment these lines
        datay = np.fft.rfft(datay)[:fft_n]
        real_datax = np.concatenate((np.real(datax), np.imag(datax)))
        real_datay = np.concatenate((np.real(datay), np.imag(datay)))
        data_seq.append(np.concatenate((real_datax, real_datay)))
        data_label.append(1)
        data_name.append(filename)
    for filename in neg_list:
        datax, datay = read_one_file(filename)
        datax = np.fft.rfft(datax)[:fft_n]
        datay = np.fft.rfft(datay)[:fft_n]
        real_datax = np.concatenate((np.real(datax), np.imag(datax)))
        real_datay = np.concatenate((np.real(datay), np.imag(datay)))
        data_seq.append(np.concatenate((real_datax, real_datay)))
        data_label.append(0)
        data_name.append(filename)
    # print data_seq

    return np.array(data_seq)/1000, np.array(data_label), data_name

def read_all():
    # read all data into [vidoe_num, one_file, data/label]
    pos_video_list = [[] for _ in range(15)]
    neg_video_list = [[] for _ in range(15)]
    pos_list = os.listdir("OldPositiveData")
    # print pos_list
    for filename in pos_list:
        num = filename.split("video")[1]
        num = num.split('.')[0]
        num = eval(num)-1
        # print filename
        pos_video_list[num].append("OldPositiveData/"+filename)

    neg_list = os.listdir("OldNegativeData")
    for filename in neg_list:
        num = filename.split("video")[1]
        num = num.split('.')[0]
        num = eval(num) - 1
        neg_video_list[num].append("OldNegativeData/"+filename)

    all_data_seq = []
    all_data_label = []
    all_data_name = []
    for i in range(15):
        seq, label, name = read_one_video(pos_video_list[i],neg_video_list[i])
        all_data_seq.append(seq)
        all_data_label.append(label)
        all_data_name.append(name)

    return all_data_seq, all_data_label, all_data_name


if __name__ == "__main__":
    t1 = time.time()
    read_all()
    t2 = time.time()
    print t2-t1
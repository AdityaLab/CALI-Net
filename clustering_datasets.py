'''
    Note: These are not used for COVID
'''
import numpy as np
import math
import os

# useful for kdd epidepp
def load_mydata(length, first_year, data_region, path = './data'):
    
    
    if data_region != 'X': # if not national region
        str_arr = data_region.split('n')
        data_region = str_arr[0]+'n '+str_arr[1]
        
        
    input_file =  os.path.join( path, 'ILINet.csv')
    x = []

    
    # indexed by region
    all_data = {}
    in_f = open(input_file)
    in_f.readline()
    in_f.readline()
    
    for line in in_f:
        raw = line.strip().split(',')
        region = raw[1].strip()
        year = int(raw[2].strip())
        week = int(raw[3].strip())
        ## upto 20th week belongs to last years cycle
        if(week <= 20):
            year -= 1
        infection = raw[4].strip()
        inf = 0
        if is_number(infection):
            inf = float(infection)
        if region not in all_data:
            all_data[region]={}
        if year not in all_data[region]:
            all_data[region][year] = []
        all_data[region][year].append(inf)
        
    indexDic = {}
    
    raw = all_data[data_region]
    keylist = list(raw.keys())
    keylist.sort()
    
    for year in keylist:
        # if year == 2003 or year == 2008:  # these years have 53
        #     print(year, len(raw[year]))
        if year>=first_year and len(raw[year]) >= 52:  # it was ==52, but some seasons have 53 TODO: check if this is fine
            indexDic[len(x)] = year
            x.append(raw[year][0:length])
    

    return np.array(x)


def load_RNNdata(length, first_year,  data_region, path = './data'):
    
    if data_region != 'X': # if not national region
        str_arr = data_region.split('n')
        data_region = str_arr[0]+'n '+str_arr[1]
        

    input_file =  os.path.join( path, 'ILINet.csv')
    
    x = []
    y = []
    peak = []
    peak_time = []
    onset_time = []

    
    baseline_file = open(os.path.join(path, 'baseline'))
    cdc_baselines = {}
    
    for line in baseline_file:
        arr = line.strip().split()
        #print(arr)
        year = int(arr[0])
        baseline = float(arr[1])
        cdc_baselines[year] = baseline
    
    
    # indexed by region
    all_data = {}
    in_f = open(input_file)
    in_f.readline()
    in_f.readline()
    
    for line in in_f:
        raw = line.strip().split(',')
        region = raw[1].strip()
        year = int(raw[2].strip())
        week = int(raw[3].strip())
        ## upto 20th week belongs to last years cycle
        if(week <= 20):
            year -= 1
        infection = raw[4].strip()
        inf = 0
        if is_number(infection):
            inf = float(infection)
        if region not in all_data:
            all_data[region]={}
        if year not in all_data[region]:
            all_data[region][year] = []
        all_data[region][year].append(inf)
        
    indexDic = {}
    

    raw = all_data[data_region]
    keylist = list(raw.keys())
    keylist.sort()
    peak_time_vals = []
    for year in keylist:
        if year>=first_year and len(raw[year]) >= 52:  # NOTE: same modification as in load_mydata()
            indexDic[len(x)] = year
            x.append(raw[year][0:length])
            y.append(raw[year][length])
            peak.append(max(raw[year]))
            peak_time_val = (raw[year]).index(max(raw[year]))
            peak_time_vec = [0]*52
            if float(peak_time_val) > 13:
                peak_time_val = 13
            peak_time_vec[peak_time_val] = 1
            peak_time_vals.append(peak_time_val)
            peak_time.append(peak_time_vec) #careful the peak time is from the 21st week
                                                                #counts from 0, so 37 means 21+37-52=6 week next year
            onset = -1
            baseline_val = cdc_baselines[year]
            for i in range(len(raw[year])-3):
                trueVals = [raw[year][x]>=baseline_val for x in range(i,i+3)]
                if all(trueVals):
                    onset = i
                    break
            onset_vec = [0]*53
            onset_vec[onset]= 1
            onset_time.append(onset_vec) #careful the peak time is from the 21st week
                                     #counts from 0, so 37 means 21+37-52=6 week next year
                                     # -1 means no onset
            
            
    x = np.array(x)
    x = x[:, :,np.newaxis]

    return x, np.array(y),np.array(peak),np.array(peak_time), np.array(onset_time), np.array(peak_time_vals)
    



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def load_myRegionaldata(length, first_year, path = './data'):
    import os
    input_file =  os.path.join( path, 'ILINetProcessed.csv')

    clusters_file = open( os.path.join(path, 'SeasonClustersFinal'))
    seasonDic = {}
    allSeasons = {}
    
    for line in clusters_file:
        arr = line.strip().split()
        year = int(arr[0])
        season = int(arr[1])
        seasonDic[year] = season
        allSeasons[season] = True
    
    all_data = read_ILINetProccessed(input_file)
        
    indexDic = {}
    
    data = {}
    
    
    region_order = []
    for region, raw in all_data.items():
        region_order.append(region)
        keylist = list(raw.keys())
        keylist.sort()
        x = []
        y = []
    
        for year in keylist:
            if year>=first_year and len(raw[year]) == 52:
                indexDic[len(x)] = year
                x.append(raw[year][0:length])
                y.append(seasonDic[year])
        
        data[region] = [np.array(x), np.array(y)]
    return data


def load_myRegionalRNNdata( length, first_year, path = './data'):
    """
        This one returns labels in classification format
    """
    
    import os
    input_file =  os.path.join( path, 'ILINetProcessed.csv')

    clusters_file = open( os.path.join(path, 'SeasonClustersFinal'))
    seasonDic = {}
    allSeasons = {}
    
    
    for line in clusters_file:
        arr = line.strip().split()
        year = int(arr[0])
        season = int(arr[1])
        seasonDic[year] = season
        allSeasons[season] = True
    
    baseline_file = open(os.path.join(path, 'wILI_Baseline.csv'))
    cdc_baselines = {}
    line = baseline_file.readline()
    for line in baseline_file:
        year = 2000
        
        arr = line.strip().split(',')
        region = arr[0]
        cdc_baselines[region] = {}
        for i in range(1, len(arr)):    
            baseline = float(arr[i])
            cdc_baselines[region][year] = baseline
            year += 1
            
    all_data = read_ILINetProccessed(input_file)
        
    indexDic = {}
    
    data = {}
    
    for region, raw in all_data.items():
        # Note: raw is a dictionary (year is key) that contains yearly sequence 

        keylist = list(raw.keys())
        keylist.sort()
        
        x = []
        y_1 = []
        y_2 = []
        y_3 = []
        y_4 = []
        y_5 = []
        y_6 = []
        peak = []
        peak_time = []
        onset_time = []
        y_1_val_arr = []
        y_2_val_arr = []
        y_3_val_arr = []
        y_4_val_arr = []
    
        for year in keylist:
            if year>=first_year and len(raw[year]) == 52:
                indexDic[len(x)] = year
                x.append(raw[year][0:length])
                
                y1_val, y2_val, y3_val, y4_val, y5_val, y6_val = raw[year][length:length+6]


                y1_vec = [0]*131
                y1_val *= 10
                if(y1_val<130):
                    y1_vec[int(math.floor(y1_val))]=1
                else:
                    y1_vec[-1]= 1
                    
                y2_vec = [0]*131
                y2_val *= 10
                if(y2_val<130):
                    y2_vec[int(math.floor(y2_val))]=1
                else:
                    y2_vec[-1]= 1
                    
                y3_vec = [0]*131
                y3_val *= 10
                if(y3_val<130):
                    y3_vec[int(math.floor(y3_val))]=1
                else:
                    y3_vec[-1]= 1
                    
                y4_vec = [0]*131
                y4_val *= 10
                if(y4_val<130):
                    y4_vec[int(math.floor(y4_val))]=1
                else:
                    y4_vec[-1]= 1
                
                y5_vec = [0]*131
                y5_val *= 10
                if(y5_val<130):
                    y5_vec[int(math.floor(y5_val))]=1
                else:
                    y5_vec[-1]= 1
                
                y6_vec = [0]*131
                y6_val *= 10
                if(y6_val<130):
                    y6_vec[int(math.floor(y6_val))]=1
                else:
                    y6_vec[-1]= 1
                
                y_1.append(y1_vec)
                y_2.append(y2_vec)
                y_3.append(y3_vec)
                y_4.append(y4_vec)
                y_5.append(y5_vec)
                y_6.append(y6_vec)
                y_1_val_arr.append([y1_val])
                y_2_val_arr.append([y2_val])
                y_3_val_arr.append([y3_val])
                y_4_val_arr.append([y4_val])
                peak_val = max(raw[year])
                peak_val_vec = [0]*131
                peak_val *= 10
                if(peak_val<130):
                    peak_val_vec[int(math.floor(peak_val))]=1
                else:
                    peak_val_vec[-1]= 1
                peak.append(peak_val_vec)
                
                peak_time_val = (raw[year]).index(max(raw[year]))
                peak_time_vec = [0]*52
                peak_time_vec[peak_time_val] = 1
                peak_time.append(peak_time_vec[19:]) #careful the peak time is from the 21st week
                                                                    #counts from 0, so 37 means 21+37-52=6 week next year
                onset = -1
                offset = -1
                baseline_val = cdc_baselines[region][year]
                for i in range(len(raw[year])-3):
                    trueVals = [raw[year][x]>=baseline_val for x in range(i,i+3)]
                    if all(trueVals):
                        onset = i
                        break
                onset_vec = [0]*53
                onset_vec[onset]= 1
                onset_time.append(onset_vec[19:]) #careful the peak time is from the 21st week
                                         #counts from 0, so 37 means 21+37-52=6 week next year
                                         # -1 means no onset
                
        x = np.array(x)
        x = x[:, :,np.newaxis]
        data[region] = [x, np.array(y_1),np.array(y_2), np.array(y_3), np.array(y_4), np.array(y_5), np.array(y_6),np.array(peak),np.array(peak_time), np.array(onset_time) ]  # previously we had np.array(y_1_val_arr), np.array(y_2_val_arr), np.array(y_3_val_arr) , np.array(y_4_val_arr)

    # print(data); quit()
    # print(y_1); quit()
    #print("y=",y)
    #print("peak =",peak)
    #print("peak_time =", peak_time)

    return data



def load_myRegionalRNNdata_NumericwILI(length, first_year, path = './data'):
    """
        This one returns labels in regression format
        Based on load_RNNdata
    """
    import os
    input_file =  os.path.join( path, 'ILINetProcessed.csv')

    clusters_file = open( os.path.join(path, 'SeasonClustersFinal'))
    seasonDic = {}
    allSeasons = {}
    
    
    for line in clusters_file:
        arr = line.strip().split()
        year = int(arr[0])
        season = int(arr[1])
        seasonDic[year] = season
        allSeasons[season] = True
    
    baseline_file = open(os.path.join(path, 'wILI_Baseline.csv'))
    cdc_baselines = {}
    line = baseline_file.readline()
    for line in baseline_file:
        year = 2000
        
        arr = line.strip().split(',')
        region = arr[0]
        cdc_baselines[region] = {}
        for i in range(1, len(arr)):    
            baseline = float(arr[i])
            cdc_baselines[region][year] = baseline
            year += 1
            
    all_data = read_ILINetProccessed(input_file)
        
    indexDic = {}
    
    data = {}
    
    
    for region, raw in all_data.items():
        # Note: raw is a dictionary (year is key) that contains yearly sequence 

        keylist = list(raw.keys())
        keylist.sort()
        x = []
        y_1 = []
        y_2 = []
        y_3 = []
        y_4 = []
        y_5 = []
        y_6 = []
        peak = []
        peak_time = []
        onset_time = []
        offset_time = []
        peak_time_vals = []
        for year in keylist:
            if year>=first_year and len(raw[year]) >= 52:  # NOTE: same modification as in load_mydata()
                indexDic[len(x)] = year
                x.append(raw[year][0:length])
                y_1.append(raw[year][length])
                y_2.append(raw[year][length+1])
                y_3.append(raw[year][length+2])
                y_4.append(raw[year][length+3])
                y_5.append(raw[year][length+4])
                y_6.append(raw[year][length+5])
                
                peak.append(max(raw[year]))
                peak_time_val = (raw[year]).index(max(raw[year]))
                peak_time_vec = [0]*52
                peak_time_vec[peak_time_val] = 1
                peak_time_vals.append(peak_time_val)
                peak_time.append(peak_time_vec[19:]) #careful the peak time is from the 21st week
                                                                    #counts from 0, so 37 means 21+37-52=6 week next year
                # peak_time.append(peak_time_vec) #careful the peak time is from the 21st week
                #                                                     #counts from 0, so 37 means 21+37-52=6 week next year
                onset = -1
                baseline_val = cdc_baselines[region][year]
                for i in range(len(raw[year])-3):
                    trueVals = [raw[year][x]>=baseline_val for x in range(i,i+3)]
                    if all(trueVals):
                        onset = i
                        break
                onset_vec = [0]*53
                onset_vec[onset]= 1
                # onset_time.append(onset_vec) #careful the peak time is from the 21st week
                #                         #counts from 0, so 37 means 21+37-52=6 week next year
                #                         # -1 means no onset
                onset_time.append(onset_vec[19:]) #careful the peak time is from the 21st week
                                         #counts from 0, so 37 means 21+37-52=6 week next year
                                         # -1 means no onset
                # NOTE: offset - for COVID
                for i in range(34,len(raw[year])-3):  # by week 34, we have passed the onset
                    trueVals = [raw[year][x]<=baseline_val for x in range(i,i+3)]
                    if all(trueVals):
                        offset = i
                        break
                offset_vec = [0]*53
                offset_vec[offset]= 1
                # offset_time.append(offset_vec) #careful the peak time is from the 21st week
                                        #counts from 0, so 37 means 21+37-52=6 week next year
                                        # -1 means no onset
                offset_time.append(offset_vec[19:]) #careful the peak time is from the 21st week
                                         #counts from 0, so 37 means 21+37-52=6 week next year
                                         # -1 means no onset
        x = np.array(x)
        x = x[:, :,np.newaxis]
        data[region] = [x, np.array(y_1),np.array(y_2), np.array(y_3), np.array(y_4), np.array(y_5), np.array(y_6), np.array(peak), np.array(peak_time), np.array(onset_time), np.array(offset_time) ]

    return data
    


def load_myRegionalRNNdata_Prediction( length, first_year, path = './data'):
    import os
    input_file =  os.path.join( path, 'ILINetProcessed.csv')

    clusters_file = open( os.path.join(path, 'SeasonClustersFinal'))
    seasonDic = {}
    allSeasons = {}
    
    
    for line in clusters_file:
        arr = line.strip().split()
        year = int(arr[0])
        season = int(arr[1])
        seasonDic[year] = season
        allSeasons[season] = True
    
    baseline_file = open(os.path.join(path, 'baseline'))
    cdc_baselines = {}
    
    for line in baseline_file:
        arr = line.strip().split()
        #print(arr)
        year = int(arr[0])
        baseline = float(arr[1])
        cdc_baselines[year] = baseline
    
    
    all_data = read_ILINetProccessed(input_file)
        
    indexDic = {}
    
    data = {}
    
    for region, raw in all_data.items():
        keylist = list(raw.keys())
        keylist.sort()
        
        x = []
    
        for year in keylist:
            if year==2019:
                indexDic[len(x)] = year
                x.append(raw[year][0:length])
                
            
        x = np.array(x)
        x = x[:, :,np.newaxis]
        
        data[region] = [x]

    #print("y=",y)
    #print("peak =",peak)
    #print("peak_time =", peak_time)

    return data

def load_myRegionaldata_Prediction(length, first_year, path = './data'):
    import os
    input_file =  os.path.join( path, 'ILINetProcessed.csv')

    clusters_file = open( os.path.join(path, 'SeasonClustersFinal'))
    seasonDic = {}
    allSeasons = {}
    
    for line in clusters_file:
        arr = line.strip().split()
        year = int(arr[0])
        season = int(arr[1])
        seasonDic[year] = season
        allSeasons[season] = True
    
    all_data = read_ILINetProccessed(input_file)
        
    indexDic = {}
    
    data = {}
    
    region_order = []
    for region, raw in all_data.items():
        region_order.append(region)
        keylist = list(raw.keys())
        keylist.sort()
        x = []
    
        for year in keylist:
            if year==2019:
                indexDic[len(x)] = year
                x.append(raw[year][0:length])
        
        data[region] = [np.array(x)]
    return data
    
def read_ILINetProccessed(input_file):
    # indexed by region
    all_data = {}
    in_f = open(input_file)
    in_f.readline()
    in_f.readline()
    
    for line in in_f:
        raw = line.strip().split(',')
        region = raw[1].strip()
        year = int(raw[2].strip())
        week = int(raw[3].strip())
        ## upto 20th week belongs to last years cycle
        if(week <= 20):
            year -= 1
        infection = raw[4].strip()
        inf = 0
        if is_number(infection):
            inf = float(infection)
        if region not in all_data:
            all_data[region]={}
        if year not in all_data[region]:
            all_data[region][year] = []
        all_data[region][year].append(inf)
    return all_data


if __name__ == "__main__":
    # load_myRegionalRNNdata(35, 2015)

    # rnn_data, rnn_label_wILI_1, rnn_label_wILI_2, rnn_label_wILI_3, rnn_label_wILI_4,\
    #     rnn_label_wILI_5, rnn_label_wILI_6, rnn_label_peak, rnn_label_peak_time, rnn_label_onset_time,\
    #          = load_myRegionalRNNdata(35, 2015)['X']
    
    length=21
    first_year=2004; RegionName='Region 1'
    rnn_data, rnn_label_wILI_1, rnn_label_wILI_2, rnn_label_wILI_3, rnn_label_wILI_4,\
        rnn_label_wILI_5, rnn_label_wILI_6, rnn_label_peak, rnn_label_peak_time, rnn_label_onset_time,\
            rnn_label_offset_time = load_myRegionalRNNdata_NumericwILI(length, first_year)[RegionName]
    
    print(rnn_data)
    print(rnn_label_wILI_1)
    print(rnn_label_wILI_2)
    print(rnn_label_wILI_3)
    print(rnn_label_wILI_4)
    print(rnn_label_wILI_5)
    print(rnn_label_wILI_6)
    print(rnn_label_onset_time)
    print(rnn_label_offset_time)
    # print(np.asarray(rnn_label_onset_time==1).nonzero())
    # print(np.asarray(rnn_label_offset_time==1).nonzero())

    print(load_ILI_as_time_series(RegionName))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:56:38 2019

@author: ubuntu
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
import wfdb

def getIndexOfRRSpO2HR(flname):
    PML = open(flname,"r")
    lnum = 0
    gotSpO2 = False
    gotRR = False
    gotHR = False
    for line in PML:
        if lnum == 0:
            firstline = line
            samples = firstline.split(' ')[3]
            samplingrate = int(float((firstline.split(' ')[2]).split('/')[0])*60)
            hours = float(samples)/(60*float(samplingrate))
            firstline = firstline + 'Sampling Rate (per min) :'+str(samplingrate)+'  Total Hours : '+str(hours) 
        if('RESP' in line and (not gotRR)):
            r = lnum-1
            gotRR = True
        if('SpO2' in line and (not gotSpO2)):
            s = lnum-1
            gotSpO2 = True
        if('HR' in line and (not gotHR)):
            h = lnum-1
            gotHR = True
        lnum = lnum + 1
    return int(samples),r,s,h,firstline


def outlierRejector(x,t,f,default=0):
    tnew = t
    xb = []
     # remove values that are zero and nan and create a new series
    for i in range(len(x)):
        if ((not np.isnan(x[i])) and x[i] > 0):
            xb.append(x[i])
    
    if(len(xb)==0):
        xb = [default] * len(x)
        x = [default] * len(xb)
    
    # calculate std and mean of this series  
    m = np.mean(xb)
    s = np.std(xb)
    if(s==0):
        s = 0.0001
    xnew = []
    for i in range(len(x)):
        if((not np.isnan(x[i])) and (abs((x[i] - m)/s) < f) and (x[i] > 0)):
            xnew.append(x[i])
        else:
            xnew.append(np.nan)
    
    ser = pd.Series(xnew)
    newser = ser.interpolate().fillna(method='bfill')
    xnew = newser.values
    return xnew,tnew

def applyLowess(x,t,points):
    f = (float(points)/len(x))
    z = sm.nonparametric.lowess(x,t,frac=f)
    return z

def fitRegressionLines(t,x,points):
    m = []
    c = []
    r = []
    # if 'points' is not an integral multiple of 'len(x)' then it will use just the points left out after second last 
    # iteration to get the line, we need not worry about it because arr[8:12] will return last two values if 
    # size of array arr is let us say 10
    for i in range(0,len(x),points):
        fit = np.polyfit(t[i:i+points],x[i:i+points],1)
        m.append(fit[0])
        c.append(fit[1])
        fit_fn = np.poly1d(fit)
        #plt.plot(t[i:i+points], fit_fn(t[i:i+points]),color,linewidth=3.0)
        r.append(r2_score(x[i:i+points], fit_fn(t[i:i+points])))  
    return m,c,r

def getFilteredLowAlarm(x,t,THRESHOLD):
    filtered_alarms = []
    ALARM_ON = 0
    for i in range(60,len(x)):
        if((np.mean(x[i-60:i]) < THRESHOLD) and ALARM_ON==0):
            ALARM_ON = 1
            filtered_alarms.append(t[i])
        if((np.mean(x[i-60:i]) > THRESHOLD) and ALARM_ON==1):
            ALARM_ON = 0
    return filtered_alarms


def getFilteredHighAlarm(x,t,THRESHOLD):
    filtered_alarms = []
    ALARM_ON = 0
    for i in range(60,len(x)):
        if((np.mean(x[i-60:i]) > THRESHOLD) and ALARM_ON==0):
            ALARM_ON = 1
            filtered_alarms.append(t[i])
        if((np.mean(x[i-60:i]) < THRESHOLD) and ALARM_ON==1):
            ALARM_ON = 0
    return filtered_alarms

def getRawHighAlarm(x,t,THRESHOLD):
    raw_alarms = []
    ALARM_ON = 0
    for i in range(0,len(x)):
        if(x[i] > THRESHOLD and ALARM_ON==0):
            ALARM_ON = 1
            raw_alarms.append(t[i])
        if(x[i] < THRESHOLD and ALARM_ON==1):
            ALARM_ON = 0
    return raw_alarms

def getRawLowAlarm(x,t,THRESHOLD):
    raw_alarms = []
    ALARM_ON = 0
    for i in range(0,len(x)):
        if(x[i] < THRESHOLD and ALARM_ON==0):
            ALARM_ON = 1
            raw_alarms.append(t[i])
        if(x[i] > THRESHOLD and ALARM_ON==1):
            ALARM_ON = 0
    return raw_alarms



def getRDAlarm(xLR, xLGB, xMLP, xSVM, xCNN, xLSTM, bLR, bLGB, bMLP, bSVM, bCNN, bLSTM, ts):
    v = bLR*xLR + bLGB*xLGB + bMLP*xMLP + bSVM*xSVM + bCNN*xCNN + bLSTM*xLSTM - 10.0
    RD_alarms = []
    ALARM_ON = 0
    for i in range(len(v)):
        if(v[i]>=0 and ALARM_ON==0):
            ALARM_ON = 1
            RD_alarms.append(ts[i])
        if(v[i]<0 and ALARM_ON==1):
            ALARM_ON = 0
    return RD_alarms
    
def getLongFeatures(rr,spo2):
    LENGTH = len(rr)
    REGRESSIONWINDOW_VAL = 1
    RR_THRESH_VAL = 25
    RR_THRESH_VAL_2 = 20
    SPO2_THRESH_VAL_2 = 95
    SPO2_THRESH_VAL = 93
    RR_SLOPE_THRESH_VAL = 1
    SPO2_SLOPE_THRESH_VAL = 1
    
    
    t = np.arange(LENGTH,dtype=np.float64)
    mRR,cRR,rRR = fitRegressionLines(t/60,rr,int(REGRESSIONWINDOW_VAL*60))
    mSPO2,cSPO2,rSPO2 = fitRegressionLines(t/60,spo2,int(REGRESSIONWINDOW_VAL*60))
    points = int(REGRESSIONWINDOW_VAL*60)
    
    rr_breach = 0 # will take value as 0 or 1
    spo2_breach = 0 # will take value as 0 or 1
    
    stage1_occurence_L = 0 # number of times stage 1 occurs when both RR < 20 and SpO2 > 95
    stage2_occurence_L = 0 # number of times stage 2 occurs when both RR < 20 and SpO2 > 95
    stage3_occurence_L = 0 # number of times stage 3 occurs when both RR < 20 and SpO2 > 95
    
    stage1_occurence_H = 0 # number of times stage 1 occurs when both RR >= 20 or SpO2 <= 95
    stage2_occurence_H = 0 # number of times stage 2 occurs when both RR >= 20 or SpO2 <= 95
    stage3_occurence_H = 0 # number of times stage 3 occurs when both RR >= 20 or SpO2 <= 95
    
    rr_high_regions = 0 # number of regions with avg RR greater than 25
    spo2_low_regions = 0 # number of regions with avg SpO2 less than 93
    
    rr_unsafe_regions = 0 # number of regions with 20<=RR<=25
    spo2_unsafe_regions = 0 # number of regions with 93<=SpO2<=95
    
    feature_sum = 0
    
    if(max(rr) >= RR_THRESH_VAL):
        rr_breach = 1
    if(min(spo2) <= SPO2_THRESH_VAL):
        spo2_breach = 1
    
    for i in range(len(mRR)):
        unsafe = False
        START = i*points
        END = START + points
        if (END >= len(t)):
            END = -1
        if(np.mean(rr[START:END]) > RR_THRESH_VAL):
            rr_high_regions = rr_high_regions + 1
            unsafe = True
            
        if(np.mean(rr[START:END]) >= RR_THRESH_VAL_2 and np.mean(rr[START:END]) <= RR_THRESH_VAL):
            rr_unsafe_regions = rr_unsafe_regions + 1
            unsafe = True
        
        if(np.mean(spo2[START:END]) < SPO2_THRESH_VAL):
            spo2_low_regions = spo2_low_regions + 1
            unsafe = True
        
        if(np.mean(spo2[START:END]) >= SPO2_THRESH_VAL and np.mean(spo2[START:END]) <= SPO2_THRESH_VAL_2):
            spo2_unsafe_regions = spo2_unsafe_regions + 1
            unsafe = True
            
        if(not unsafe):
            if(mRR[i] > RR_SLOPE_THRESH_VAL and mSPO2[i] >= -SPO2_SLOPE_THRESH_VAL and mSPO2[i] <= SPO2_SLOPE_THRESH_VAL):
                stage1_occurence_L = stage1_occurence_L + 1
            elif(mRR[i] > RR_SLOPE_THRESH_VAL and mSPO2[i] < -SPO2_SLOPE_THRESH_VAL):
                stage2_occurence_L = stage2_occurence_L + 1
            elif(mRR[i] < -RR_SLOPE_THRESH_VAL and mSPO2[i] < -SPO2_SLOPE_THRESH_VAL):
                stage3_occurence_L = stage3_occurence_L + 1
        if(unsafe):
            if(mRR[i] > RR_SLOPE_THRESH_VAL and mSPO2[i] >= -SPO2_SLOPE_THRESH_VAL and mSPO2[i] <= SPO2_SLOPE_THRESH_VAL):
                stage1_occurence_H = stage1_occurence_H + 1
            elif(mRR[i] > RR_SLOPE_THRESH_VAL and mSPO2[i] < -SPO2_SLOPE_THRESH_VAL):
                stage2_occurence_H = stage2_occurence_H + 1
            elif(mRR[i] < -RR_SLOPE_THRESH_VAL and mSPO2[i] < -SPO2_SLOPE_THRESH_VAL):
                stage3_occurence_H = stage3_occurence_H + 1
            
    
    feature_sum = (rr_breach + spo2_breach + rr_high_regions + spo2_low_regions 
                   + stage1_occurence_L + stage2_occurence_L + stage3_occurence_L 
                   + rr_unsafe_regions + spo2_unsafe_regions
                   + stage1_occurence_H + stage2_occurence_H + stage3_occurence_H)
                   

    
    return [rr_breach, spo2_breach, rr_high_regions, spo2_low_regions, 
            stage1_occurence_L, stage2_occurence_L, stage3_occurence_L, 
            rr_unsafe_regions, spo2_unsafe_regions,
            stage1_occurence_H, stage2_occurence_H, stage3_occurence_H,
            feature_sum]

def addFeatureVectorToSegFrame(i,dfOuranno,starting_time,label,segFrame):
    LOWESSWINDOW = 3
    OUTLIERSD = 2.5
    record_name = dfOuranno.iloc[i]['RecordNum'] 
    flname = './FinalRecords/'+str(record_name)+'n.hea'
    recname = './FinalRecords/'+str(record_name)+'n'
    [samples,R,S,H,firstline] = getIndexOfRRSpO2HR(flname)
    rec =  wfdb.io.rdrecord(str(recname))
    xrr = rec.p_signal[:,R]
    xspo2 = rec.p_signal[:,S]
    TOTAL_LEN = len(xrr)
    t = np.arange(0,TOTAL_LEN,1)

    xrr = xrr[int(starting_time*60):int(starting_time*60+24*60)]
    xspo2 = xspo2[int(starting_time*60):int(starting_time*60+24*60)]
    t = t[int(starting_time*60):int(starting_time*60+24*60)]

    [xrrnew,trrnew] = outlierRejector(xrr,t,OUTLIERSD, default=15.0)
    [xspo2new,tspo2new] = outlierRejector(xspo2,t,OUTLIERSD, default=98.0)

    zrrnew = applyLowess(xrrnew,trrnew,LOWESSWINDOW*60)
    zspo2new = applyLowess(xspo2new,tspo2new,LOWESSWINDOW*60)

    [rr_breach, spo2_breach, rr_high_regions, spo2_low_regions, 
     stage1_occurence_L, stage2_occurence_L, stage3_occurence_L, 
     rr_unsafe_regions, spo2_unsafe_regions,
     stage1_occurence_H, stage2_occurence_H, stage3_occurence_H,
     feature_sum] = getLongFeatures(zrrnew[:,1],zspo2new[:,1])

    seg_num = np.shape(segFrame)[0]
    seg_id = 'Seg'+str(seg_num)
    segFrame.loc[seg_num+1] = [seg_id, record_name,starting_time,
                               rr_breach, spo2_breach, rr_high_regions, spo2_low_regions,
                               stage1_occurence_L, stage2_occurence_L, stage3_occurence_L,
                               rr_unsafe_regions, spo2_unsafe_regions,
                               stage1_occurence_H, stage2_occurence_H, stage3_occurence_H,
                               feature_sum,label]

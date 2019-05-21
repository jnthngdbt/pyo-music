#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd

matplotlib.style.use('ggplot')

plt.close('all')

day0num = dates.datestr2num('2017-06-23')
yesterday = dates.date2num(dates.datetime.date.today()) - 1

numdays = int(yesterday - day0num + 1)
nummins = 60*24

days = np.arange(numdays) + day0num
hours = np.arange(nummins)/60

daysdata = np.zeros((nummins, numdays))
dayssmooth = np.zeros((nummins, numdays))

for i, day in enumerate(days):
    
    daydate = dates.num2date(day)
    daystr = daydate.strftime('%Y-%m-%d')
    
    url = 'https://api.fitbit.com/1/user/-/activities/steps/date/{}/{}.json'.format(daystr, daystr)                                                    
    
    # - go to https://apigee.com/me3/embed/console/fitbit
    # - make a get request
    # - get the bearer token in the bottom left window
    headers = {"Authorization":"Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI1VENDOVAiLCJhdWQiOiIyMjhUTVAiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd3dlaSB3c29jIHdhY3Qgd3NldCB3bG9jIiwiZXhwIjoxNDk4NzgyNDc5LCJpYXQiOjE0OTg3NTM2Nzl9.rQ1eofmgMQNkpUNBeuROW_xqDGywJnpEMPaKsx9df1o"}
    x = requests.get(url, headers=headers).json()
    
    xi = x['activities-steps-intraday']
    xd = xi['dataset']
    
    df = pd.DataFrame.from_dict(xd)
    
    daysdata[:,i] = df.value
    dayssmooth[:,i] = np.convolve(df.value, np.ones(5), 'same')
            
daystrs = []
for day in days:
    daydate = dates.num2date(day)
    daystrs.append(daydate.strftime('%Y-%m-%d'))
    
hourstrs = []
for hour in np.arange(24):
    hourstrs.append('{}:00'.format(hour))
    hourstrs.append('{}:30'.format(hour))

hourticks = nummins * np.arange(len(hourstrs))/len(hourstrs)

plt.figure()

plt.plot(daysdata, '.-k', alpha=0.1)
plt.xticks(hourticks, hourstrs, rotation=90)
plt.ylim([1,np.max(daysdata)])

ax = plt.gca()
ax.set_facecolor('white')

plt.tight_layout() # to make sure date tick labels are visible

plt.figure(figsize=(3.20, 6.80))

plt.imshow(np.sqrt(dayssmooth))
plt.axis('tight')
plt.grid('off')

plt.xticks(np.arange(numdays), daystrs, rotation=90)
plt.yticks(hourticks, hourstrs)

plt.tight_layout() # to make sure date tick labels are visible



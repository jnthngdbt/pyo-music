#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:15:00 2017

@author: jonathangodbout
"""

import json
import requests
import pandas as pd

url = 'https://api.fitbit.com/1/user/-/activities/steps/date/2017-06-26/2017-06-26.json'

headers = {"Authorization":"Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI1VENDOVAiLCJhdWQiOiIyMjhUTVAiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd3dlaSB3c29jIHdhY3Qgd3NldCB3bG9jIiwiZXhwIjoxNDk4Njg5MTc1LCJpYXQiOjE0OTg2NjAzNzV9.L-ila7yVJW1MKeGKKQYnK2Ju-k3Zv39JY58WkZXDHaQ"}
x = requests.get(url, headers=headers).json()

xi = x['activities-steps-intraday']
xd = xi['dataset']

df = pd.DataFrame.from_dict(xd)

df.plot('time','value')
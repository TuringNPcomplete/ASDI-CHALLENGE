# Import statements for packages used...
import os, glob, shutil, sys, requests, json
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from botocore import UNSIGNED
from botocore.config import Config
from io import StringIO
from datetime import datetime
from types import SimpleNamespace
from IPython.display import clear_output



# CELL #3: Review the Variables and Classes defined in this cell to see defaults and how ASDI access occurs...

# class AQParam => Used to define attributes for the (6) main OpenAQ parameters.
class AQParam:
    def __init__(self, id, name, unit, unhealthyThresholdDefault, desc):
        self.id                        = id
        self.name                      = name
        self.unit                      = unit
        self.unhealthyThresholdDefault = unhealthyThresholdDefault
        self.desc                      = desc
    
    def isValid(self):
        if(self is not None and self.id > 0 and self.unhealthyThresholdDefault > 0.0 and 
           len(self.name) > 0 and len(self.unit) > 0 and len(self.desc) > 0):
            return True
        else:
            return False
            
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

# class AQScenario => Defines an ML scenario including a Location w/ NOAA Weather Station ID 
#                     and the target OpenAQ Param.
# Note: OpenAQ data mostly begins sometime in 2016, so using that as a default yearStart value.
class AQScenario:
    def __init__(self, location=None, noaaStationID=None, aqParamTarget=None, unhealthyThreshold=None, 
                 yearStart=2016, yearEnd=2022, aqRadiusMiles=10, featureColumnsToDrop=None):
        self.location           = location
        self.name               = location + "_" + aqParamTarget.name
        self.noaaStationID      = noaaStationID
        self.noaaStationLat     = 0.0
        self.noaaStationLng     = 0.0
        self.openAqLocIDs       = []
        
        self.aqParamTarget      = aqParamTarget
        
        if unhealthyThreshold and unhealthyThreshold > 0.0:
            self.unhealthyThreshold = unhealthyThreshold
        else:
            self.unhealthyThreshold = self.aqParamTarget.unhealthyThresholdDefault
        
        self.yearStart          = yearStart
        self.yearEnd            = yearEnd
        self.aqRadiusMiles      = aqRadiusMiles
        self.aqRadiusMeters     = aqRadiusMiles * 1610 # Rough integer approximation is fine here.
        
        self.modelFolder        = "AutogluonModels"
            
    def getSummary(self):
        return f"Scenario: {self.name} => {self.aqParamTarget.desc} ({self.aqParamTarget.name}) with UnhealthyThreshold > {self.unhealthyThreshold} {self.aqParamTarget.unit}"
    
    def getModelPath(self):
        return f"{self.modelFolder}/aq_{self.name}_{self.yearStart}-{self.yearEnd}/"
    
    def updateNoaaStationLatLng(self, noaagsod_df_row):
        # Use a NOAA row to set Lat+Lng values used for the OpenAQ API requests...
        if(noaagsod_df_row is not None and noaagsod_df_row['LATITUDE'] and noaagsod_df_row['LONGITUDE']):
            self.noaaStationLat = noaagsod_df_row['LATITUDE']
            self.noaaStationLng = noaagsod_df_row['LONGITUDE']
            print(f"NOAA Station Lat,Lng Updated for Scenario: {self.name} => {self.noaaStationLat},{self.noaaStationLng}")
        else:
            print("NOAA Station Lat,Lng COULD NOT BE UPDATED.")
    
    def isValid(self):
        if(self is not None and self.aqParamTarget is not None and
           self.yearStart > 0 and self.yearEnd > 0 and self.yearEnd >= self.yearStart and 
           self.aqRadiusMiles > 0 and self.aqRadiusMeters > 0 and self.unhealthyThreshold > 0.0 and 
           len(self.name) > 0 and len(self.noaaStationID) > 0):
            return True
        else:
            return False
            
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

# class AQbyWeatherApp => Main app class with settings, AQParams, AQScenarios, and data access methods...
class AQbyWeatherApp:
    def __init__(self, mlTargetLabel='isUnhealthy', mlEvalMetric='accuracy', mlTimeLimitSecs=None):
        self.mlTargetLabel   = mlTargetLabel
        self.mlEvalMetric    = mlEvalMetric
        self.mlTimeLimitSecs = mlTimeLimitSecs
        self.mlIgnoreColumns = ['DATE','NAME','LATITUDE','LONGITUDE','day','unit','average','parameter']
        
        self.defaultColumnsNOAA   = ['DATE','NAME','LATITUDE','LONGITUDE',
                                     'DEWP','WDSP','MAX','MIN','PRCP','MONTH'] # Default relevant NOAA columns
        self.defaultColumnsOpenAQ = ['day','parameter','unit','average']       # Default relevant OpenAQ columns
        
        self.aqParams    = {} # A list to save AQParam objects
        self.aqScenarios = {} # A list to save AQScenario objects
        
        self.selectedScenario = None
    
    def addAQParam(self, aqParam):
        if aqParam and aqParam.isValid():
            self.aqParams[aqParam.name] = aqParam
            return True
        else:
            return False
    
    def addAQScenario(self, aqScenario):
        if aqScenario and aqScenario.isValid():
            self.aqScenarios[aqScenario.name] = aqScenario
            if(self.selectedScenario is None):
                self.selectedScenario = self.aqScenarios[next(iter(self.aqScenarios))] # Default selectedScenario to 1st item.
            return True
        else:
            return False
    
    def getFilenameNOAA(self):
        if self and self.selectedScenario and self.selectedScenario.isValid():
            return f"dataNOAA_{self.selectedScenario.name}_{self.selectedScenario.yearStart}-{self.selectedScenario.yearEnd}_{self.selectedScenario.noaaStationID}.csv"
        else:
            return ""
    
    def getFilenameOpenAQ(self):
        if self and self.selectedScenario and self.selectedScenario.isValid() and len(self.selectedScenario.openAqLocIDs) > 0:
            idString = ""
            for i in range(0, len(self.selectedScenario.openAqLocIDs)):
                idString = idString + str(self.selectedScenario.openAqLocIDs[i]) + "-"
            idString = idString[:-1]
            return f"dataOpenAQ_{self.selectedScenario.name}_{self.selectedScenario.yearStart}-{self.selectedScenario.yearEnd}_{idString}.csv"
        else:
            return ""
    
    def getFilenameOther(self, prefix):
        if self and self.selectedScenario and self.selectedScenario.isValid():
            return f"{prefix}_{self.selectedScenario.name}_{self.selectedScenario.yearStart}-{self.selectedScenario.yearEnd}.csv"
    
    def getNoaaDataFrame(self):
        # ASDI Dataset Name: NOAA GSOD
        # ASDI Dataset URL : https://registry.opendata.aws/noaa-gsod/
        # NOAA GSOD README : https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt
        # NOAA GSOD data in S3 is organized by year and Station ID values, so this is straight-forward
        # Example S3 path format => s3://noaa-gsod-pds/{yyyy}/{stationid}.csv
        # Let's start with a new DataFrame and load it from a local CSV or the NOAA data source...
        noaagsod_df = pd.DataFrame()
        filenameNOAA = self.getFilenameNOAA()

        if os.path.exists(filenameNOAA):
            # Use local data file already accessed + prepared...
            print('Loading NOAA GSOD data from local file: ', filenameNOAA)
            noaagsod_df = pd.read_csv(filenameNOAA)
        else:
            # Access + prepare data and save to a local data file...
            noaagsod_bucket = 'noaa-gsod-pds'
            print(f'Accessing and preparing data from ASDI-hosted NOAA GSOD dataset in Amazon S3 (bucket: {noaagsod_bucket})...')
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

            for year in range(self.selectedScenario.yearStart, self.selectedScenario.yearEnd + 1):
                key = f'{year}/{self.selectedScenario.noaaStationID}.csv'                                    # Compute the key to get
                csv_obj = s3.get_object(Bucket=noaagsod_bucket, Key=key)                                     # Get the S3 object
                csv_string = csv_obj['Body'].read().decode('utf-8')                                          # Read object contents to a string
                noaagsod_df = pd.concat([noaagsod_df, pd.read_csv(StringIO(csv_string))], ignore_index=True) # Use the string to build the DataFrame

            # Perform some Feature Engineering to append potentially useful columns to our dataset... (TODO: Add more to optimize...)
            # It may be true that Month affects air quality (ie: seasonal considerations; tends to have correlation for certain areas)
            noaagsod_df['MONTH'] = pd.to_datetime(noaagsod_df['DATE']).dt.month

            # Trim down to the desired key columns... (do this last in case engineered columns are to be removed)
            noaagsod_df = noaagsod_df[self.defaultColumnsNOAA]
            
        return noaagsod_df
        
    def getOpenAqDataFrame(self):
        # ASDI Dataset Name: OpenAQ
        # ASDI Dataset URL : https://registry.opendata.aws/openaq/
        # OpenAQ API Docs  : https://docs.openaq.org/#/v2/
        # OpenAQ S3 data is only organized by date folders, so each folder is large and contains all stations.
        # Because of this, it's better to query ASDI OpenAQ data using the CloudFront-hosted API.
        # Note that some days may not have values and will get filtered out via an INNER JOIN later.
        # Let's start with a new DataFrame and load it from a local CSV or the NOAA data source...
        aq_df = pd.DataFrame()
        aq_reqUrlBase = "https://api.openaq.org/v2" # OpenAQ ASDI API Endpoint URL Base (ie: add /locations OR /averages)
        
        if self.selectedScenario.noaaStationLat == 0.0 or self.selectedScenario.noaaStationLng == 0.0:
            print("NOAA Station Lat/Lng NOT DEFINED. CANNOT PROCEED")
            return aq_df
        
        if len(self.selectedScenario.openAqLocIDs) == 0:
            # We must start by querying nearby OpenAQ Locations for their IDs...
            print('Accessing ASDI-hosted OpenAQ Locations (HTTPS API)...')
            aq_reqParams = {
                'limit': 10,
                'page': 1,
                'offset': 0,
                'sort': 'desc',
                'order_by': 'location',
                'parameter': self.selectedScenario.aqParamTarget.name,
                'coordinates': f'{self.selectedScenario.noaaStationLat},{self.selectedScenario.noaaStationLng}',
                'radius': self.selectedScenario.aqRadiusMeters,
                'isMobile': 'false',
                'sensorType': 'reference grade',
                'dumpRaw': 'false'
            }
            aq_resp = requests.get(aq_reqUrlBase + "/locations", aq_reqParams)
            aq_data = aq_resp.json()
            if aq_data['results'] and len(aq_data['results']) >= 1:
                for i in range(0, len(aq_data['results'])):
                    self.selectedScenario.openAqLocIDs.append(aq_data['results'][i]['id'])
                print(f"OpenAQ Location IDs within {self.selectedScenario.aqRadiusMiles} miles ({self.selectedScenario.aqRadiusMeters}m) " +
                      f"of NOAA Station {self.selectedScenario.noaaStationID} at " +
                      f"{self.selectedScenario.noaaStationLat},{self.selectedScenario.noaaStationLng}: {self.selectedScenario.openAqLocIDs}")
            else:
                print(f"NO OpenAQ Location IDs found within {self.selectedScenario.aqRadiusMiles} miles ({self.selectedScenario.aqRadiusMeters}m) " +
                      f"of NOAA Station {self.selectedScenario.noaaStationID}. CANNOT PROCEED.")
        
        if len(self.selectedScenario.openAqLocIDs) >= 1:
            filenameOpenAQ = self.getFilenameOpenAQ()

            if os.path.exists(filenameOpenAQ):
                # Use local data file already accessed + prepared...
                print('Loading OpenAQ data from local file: ', filenameOpenAQ)
                aq_df = pd.read_csv(filenameOpenAQ)
            else:
                # Access + prepare data (NOTE: calling OpenAQ API one year at a time to avoid timeouts)
                print('Accessing ASDI-hosted OpenAQ Averages (HTTPS API)...')

                for year in range(self.selectedScenario.yearStart, self.selectedScenario.yearEnd + 1):
                    aq_reqParams = {
                        'date_from': f'{year}-01-01',
                        'date_to': f'{year}-12-31',
                        'parameter': [self.selectedScenario.aqParamTarget.name],
                        'limit': 366,
                        'page': 1,
                        'offset': 0,
                        'sort': 'asc',
                        'spatial': 'location',
                        'temporal': 'day',
                        'location': self.selectedScenario.openAqLocIDs,
                        'group': 'true'
                    }
                    aq_resp = requests.get(aq_reqUrlBase + "/averages", aq_reqParams)
                    aq_data = aq_resp.json()
                    if aq_data['results'] and len(aq_data['results']) >= 1:
                        aq_df = pd.concat([aq_df, pd.json_normalize(aq_data['results'])], ignore_index=True)

                # Using the joined dataframe, trim down to the desired key columns...
                aq_df = aq_df[self.defaultColumnsOpenAQ]

                # Perform some Label Engineering to add our binary classification label => {0=OKAY, 1=UNHEALTHY}
                aq_df[self.mlTargetLabel] = np.where(aq_df['average'] <= self.selectedScenario.unhealthyThreshold, 0, 1)
        
        return aq_df
    
    def getMergedDataFrame(self, noaagsod_df, aq_df):
        if len(noaagsod_df) > 0 and len(aq_df) > 0:
            merged_df = pd.merge(noaagsod_df, aq_df, how="inner", left_on="DATE", right_on="day")
            merged_df = merged_df.drop(columns=self.mlIgnoreColumns)
            return merged_df
        else:
            return pd.DataFrame()
    
    def getConfusionMatrixData(self, cm):
        cmData = SimpleNamespace()
        cmData.TN = cm[0][0]
        cmData.TP = cm[1][1]
        cmData.FN = cm[1][0]
        cmData.FP = cm[0][1]
        
        cmData.TN_Rate = cmData.TN/(cmData.TN+cmData.FP)
        cmData.TP_Rate = cmData.TP/(cmData.TP+cmData.FN)
        cmData.FN_Rate = cmData.FN/(cmData.FN+cmData.TP)
        cmData.FP_Rate = cmData.FP/(cmData.FP+cmData.TN)
        
        cmData.TN_Output = f"True Negatives  (TN): {cmData.TN} of {cmData.TN+cmData.FP} => {round(cmData.TN_Rate * 100, 2)}%"
        cmData.TP_Output = f"True Positives  (TP): {cmData.TP} of {cmData.TP+cmData.FN} => {round(cmData.TP_Rate * 100, 2)}%"
        cmData.FN_Output = f"False Negatives (FN): {cmData.FN} of {cmData.FN+cmData.TP} => {round(cmData.FN_Rate * 100, 2)}%"
        cmData.FP_Output = f"False Positives (FP): {cmData.FP} of {cmData.FP+cmData.TN} => {round(cmData.FP_Rate * 100, 2)}%"
        
        return cmData
            
print("Classes and Variables are ready.")
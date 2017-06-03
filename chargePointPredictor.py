import numpy as np
import pandas as pd

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score, KFold

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

import time
from datetime import datetime

#======================== Estimators class ===========================#
class Estimators:
	"""
	Estimators class. This class
	(i) fits charging duration and energy consumption model 
	with designated regressor type (Random Forest, Extra-Random Forest, or 
	Decision Tree Regressor). 
	(ii) predicts charging duration and energy consumption with trained models

	"""

	def __init__(self, filePath, durationModelType = "RF", energyEstimatorType = "RF"):
		
		# Load and process data
		data = self.loadData(filePath)

		# Set regression type for charging duration and energy consumption
		self.durationModelType 		= durationModelType
		self.energyEstimatorType 	= energyEstimatorType

		# Parse attributes and target columns
		attrColumns = ['Start Time Seconds From Midnight','Vehicle Battery Capacity']
		self.X 		= data[attrColumns] 	# Vehicle Battery Capacity
											# Station Start time
											# OPTIONAL: User Id?
											# OPTIONAL: Vehicle Model Year?
						
		self.durationData 	= pd.to_numeric(data["Charging Time Secs"], downcast='float')/60.0 # mins
		self.energyData 	= pd.to_numeric(data["Total Charge"], downcast='float') # kW?

		# Fit estimators
		self.fitDurationEstimator(durationModelType)
		self.fitEnergyEstimator(energyEstimatorType)

		### Validate the models ### -- TO BE DEVELOPED...
		# K = 5 # k-folds cross-validation
		# y = self.energyData
		# R2 = cross_val_score(self.energyEstimator, self.X, y=np.ravel(y), cv=KFold(y.size, K), n_jobs=1, scoring="accuracy").mean()
		# self.R2 = R2
		# print "The %d-Folds estimate of the coefficient of determination is R2 = %s" % (K, R2)

		print "Done: fitting estimators"

	def loadData(self, filePath):
		""" Load and process raw data """

		### Load data ###
		print "loading data..."	
		timeStart = time.clock()
		if "xlsx" in filePath:
			data = pd.read_excel(filePath)
		elif "csv" in filePath:
			data = pd.read_csv(filePath)
		else:
			raise ValueError("Wrong file path: " + filePath)
		print "Done: loading data. Time: "  + str(time.clock()-timeStart)

		
		### Process data ###
		# - convert time in string to seconds from midnight
		# - filter out rows containing nan elements
		print "processing data..."
		timeStart = time.clock()
		
		# Get Start Time in seconds from midnight
		startTimeSec = [(datetime.strptime(startTime,"%m/%d/%Y %H:%M")-datetime.strptime(startTime,"%m/%d/%Y %H:%M").replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() for startTime in data["Station Start Time"]]
		print "Done: processing data. Time: "  + str(time.clock()-timeStart)
		
		# Add new coloumn for start time in seconds from midnight
		data['Start Time Seconds From Midnight'] = startTimeSec
		
		# Remove indices with NaN elements from columns that you want to use in training
		nanIndices = data["Vehicle Battery Capacity"].isnull().values
		dataFiltered = data[np.invert(nanIndices)]
		# {COPY THE ABOVE TWO LINES AND CHANGE THE INDEX KEY IF YOU HAVE ANOTHER COLUMN TO FILTER OUT}

		return dataFiltered

	def fitDurationEstimator(self, modelType = "RF"):
		""" Fit duration model with specified regressor type (Random forest by default) """	
		
		print "fitting charging duration model..."
		
		if modelType == "RF":
			self.durationEstimator = RandomForestRegressor(random_state = 0, n_estimators = 50, max_depth = 50)
			self.durationEstimator.fit(self.X, self.durationData)

		# {ADD OTHER IF STATEMENTS FOR OTHER REGRESSOR MODELS, E.G., EXTRA-TREE REGRESSOR}
	
	def fitEnergyEstimator(self, modelType = "RF"):
		""" Fit energy consumption model with specified regressor type (Random forest by default) """	

		print "fitting energy consumption model..."

		# Stack energy consumption data to attribute data before fitting the energy model. 
		# i.e., we use a charging duration to predict an energy consumption
		X = self.X
		X["chargingDuration"] = self.durationData

		# Fit energy consumption regressor
		if modelType == "RF":
			self.energyEstimator = RandomForestRegressor(random_state = 0, n_estimators = 50, max_depth = 50)
			self.energyEstimator.fit(X, self.energyData)

		# {ADD OTHER IF STATEMENTS FOR OTHER REGRESSOR MODELS, E.G., EXTRA-TREE REGRESSOR}


	def estimateChargingDuration(self, Xq):
		return self.durationEstimator.predict(Xq)
	
	def estimateEnergyConsumptions(self, Xq):
		return self.energyEstimator.predict(Xq)

	def predict(self, df):
		"""
		Returns predicted charging duration and energy consumption based on the trained estimators
		
		Params
		df: dataframe with 'Start Time Seconds From Midnight', 'Vehicle Battery Capacity' columns NOTE: add more if needed
		
		"""
		
		# Get start time in seconds from midnight
		startTime = df["Station Start Time"][0]
		startTimeSec = (datetime.strptime(startTime,"%m/%d/%Y %H:%M")-datetime.strptime(startTime,"%m/%d/%Y %H:%M").replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
		
		# Build input Dataframe 
		Xq = pd.DataFrame()
		Xq['Start Time Seconds From Midnight'] = [startTimeSec]
		Xq['Vehicle Battery Capacity'] = df["Vehicle Battery Capacity"]

		# Estimate charging duration
		estDuration = self.estimateChargingDuration(Xq)
		
		# Estimate energy consumption
		Xq["ChargingDuration"] = [estDuration]
		estEnergy = self.estimateEnergyConsumptions(Xq)
		
		return estDuration[0], estEnergy[0]




#======================== Main scripts ===========================#
#### CHANGE INPUT ARGUMENTS HERE ###
trainFilePath 		= "/Users/mygreencar/Documents/workspace/python/MGC_ChargePoint/data/session-data.csv"
predictionFilePath 	= "/Users/mygreencar/Documents/workspace/python/MGC_ChargePoint/data/session-data.csv"

### Initialize & Fit charging duration and energy consumption model ###
estimators = Estimators(trainFilePath)

### YOUR PREDICTION HERE ###
#=========== < Example > ==========#
# Customize data length
examDataLen = 100

# Read data to predict
data = pd.read_csv(predictionFilePath)

# Filter out nan elements
nanIndices 		= data["Vehicle Battery Capacity"].isnull().values
dataFiltered 	= data[np.invert(nanIndices)]

# Target columns that are used as input of the estimator
columns 		= ['Station Start Time','Vehicle Battery Capacity'] # ADD MORE

# Trim data with customized data length
attrs 			= dataFiltered[columns].head(examDataLen)

# Initialize lists that store predicted values
estDurationList = []
estEnergyList 	= []

# Estimate charging duration and energy consumption iteratively
for startTimeSec, battCap in zip(attrs[columns[0]], attrs[columns[1]]):
	df 						= pd.DataFrame()
	df[columns[0]] 			= [startTimeSec]
	df[columns[1]] 			= [battCap]
	# {ADD MORE COLUMNS AS YOU HAVE MORE ATTRIBUTES}
	
	# Predict
	estDuration, estEnergy 	= estimators.predict(df)
	
	# Append the lists
	estDurationList 		+= [estDuration]
	estEnergyList 			+= [estEnergy]

# Visualization
pl = figure(title="Energy Consumption Prediction Example")
pl.grid.grid_line_alpha = 0.3
pl.xaxis.axis_label = 'Event'
pl.yaxis.axis_label = 'Energy consumption (kW)'

pl.line(range(len(estEnergyList)), pd.to_numeric(dataFiltered["Total Charge"][:examDataLen]), color='#A6CEE3', legend='Observed')
pl.line(range(len(estEnergyList)), estEnergyList, color='#B2DF8A', legend='Predicted')
pl.legend.location = "top_left"

output_file("prediction_example.html", title="Prediction example")
show(gridplot([[pl]],plot_width=400, plot_height=400))



# TODO: CROSS-VALIDATION OF THE MODEL
# TODO: PREDICTION SCORE
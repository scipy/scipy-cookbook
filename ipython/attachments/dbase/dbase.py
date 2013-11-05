from scipy import c_, arange, array, unique, kron, ones, eye
from numpy.random import randn
from __future__ import division
import pylab, cPickle, shelve, csv, copy, os

class dbase:
	"""
	TODO:
		- Check if shelve loading/saving works
		- Only tested on Mac OS X 10.4.8, with pytz

	DOC:
	A simple data-frame, that reads and writes csv/pickle/shelve files with variable names.
	Data is stored in a dictionary. 

	To use the class:

		>>> from dbase import dbase
		>>> y = dbase('your_filename.csv')

	or for a previously created dbase object stored in a pickle file

		>>> from dbase import dbase
		>>> y = dbase('your_filename.pickle')

		or without importing the dbase class

		>>> import cPickle 
		>>> f = open('your_filename.pickle','rb')
		>>> y = cPickle.load(f)
		>>> data_key = cPickle.load(f)
		>>> f.close()

	or for a dictionary stored in a shelf file

		>>> from dbase import dbase
		>>> y = dbase('your_filename.pickle')

	To return a list of variable names and an array of data

		>>> varnm, data = y.get()
	
	For usage examples of other class methods see the class tests at the bottom of this file. To see the class in action
	simply run the file using 'python dbase.py'. This will generate some simulated data (data.csv) and save instance data
	of the class to a pickle file.
	"""

	def __init__(self,fname,*var,**date):
		"""
		Initializing the dbase class. Loading file fname.

		If you have have a column in your csv file that is a date-string use:

			>>> x = dbase('myfile.csv',date = 0)

		where 0 is the index of the date column

		If you have have an array in your pickle file that is a date variable use:

			>>> x = dbase('myfile.pickle',date = 'date')

		where 'date' is the key of the date array
		"""
		self.load(fname,var,date)

	def load(self,fname,var,date):
		"""
		Loading data from a csv or a pickle file of the dbase class.
		If this is csv file use pylab's load function. Seems much faster
		than scipy.io.read_array.
		"""
		# setting the ascii/csv file name used for input
		self.DBname = os.getcwd() + '/' + fname

		# assuming self.date_key = None unless otherwise given
		self.date_key = date.values()

		# getting the file extension
		fext = self.__ext(fname)

		# opening the file for reading
		if fext == 'csv':
			f = open(fname,'r')
			self.load_csv(f)
			f.close()
		elif fext == 'pickle':
			f = open(fname,'rb')
			self.load_pickle(f)
			f.close()
		elif fext == 'she':
			self.load_shelve(fname,var)
		else:
			raise 'This class only works on csv, pickle, and shelve files'

		# specifying nobs in self.data
		self.nobs = self.data[self.data.keys()[0]].shape[0]

	def load_csv(self,f):
		"""
		Loading data from a csv file. Uses pylab's load function. Seems much faster
		than scipy.io.read_array.
		"""
		varnm = f.readline().split(',')

		# what is the date variable's key if any, based on index passed as argument
		if self.date_key != []:
			rawdata = pylab.load(f, delimiter=',',converters={self.date_key[0]:pylab.datestr2num})			# don't need to 'skiprow' here
			self.date_key = varnm[self.date_key[0]]
		else:
			rawdata = pylab.load(f, delimiter=',')															# don't need to 'skiprow' here

		# making sure that the variable names contain no leading or trailing spaces
		varnm = [i.strip() for i in varnm]

		# transforming the data into a dictionary
		self.data = dict(zip(varnm,rawdata.T))

	def load_pickle(self,f):
		"""
		Loading data from a created earlier using the the dbase class.
		"""
		self.data = cPickle.load(f)					# loading the data dictionary

		# what is the date variable's key if any
		if self.date_key == []:
			try:
				self.date_key = cPickle.load(f)		# if nothing given assume it is in the pickle file
			except:
				print "No date series in pickle file"
		else:
			self.date_key = self.date_key[0]		# assumes formatting using pylab.datestr2num already applied

	def load_shelve(self,fname,var):
		"""
		Loading data from a created earlier using the the dbase class.
		"""
		data = shelve.open(fname)				# loading the data dictionary

		# find out if a variable list is provided
		if var == ():
			var = data.keys()

		# making sure the date variable is fetched from shelve
		if self.date_key != []:
			if not self.date_key[0] in var: var = var + self.date_key
			self.date_key = self.date_key[0]		# assumes formatting using pylab.datestr2num already applied

		self.data = dict([(i,data[i]) for i in var])
		data.close()

	def save(self,fname):
		"""
		Dumping the class data dictionary into a csv or pickle file
		"""
		fext = self.__ext(fname)
		if fext == 'csv':
			f = open(fname,'w')
			self.save_csv(f)
			f.close()
		elif fext == 'pickle':
			f = open(fname,'wb')
			self.save_pickle(f)
			f.close()
		elif fext == 'she':
			self.save_shelve(fname)
		else:
			raise 'This class only works on csv, pickle, and shelve files'

	def save_csv(self,f):
		"""
		Dumping the class data dictionary into a csv file
		"""
		writer = csv.writer(f)
		writer.writerow(self.data.keys())

		data = self.data						# a reference to the data dict
		if self.date_key != []:
			data = dict(data)				# making a copy so the dates can be changed to strings
			dates = pylab.num2date(data[self.date_key])
			dates = array([i.strftime('%d %b %y') for i in dates])
			data[self.date_key] = dates

		writer.writerows(array(data.values()).T)

	def save_pickle(self,f):
		"""
		Dumping the class data dictionary and date_key into a binary pickle file
		"""
		cPickle.dump(self.data,f,2)
		cPickle.dump(self.date_key,f,2)

	def save_shelve(self,fname):
		"""
		Dumping the class data dictionary into a shelve file
		"""
		f = shelve.open('data.she','c') 
		f = self.data
		f.close()

	def add_trend(self,tname = 'trend'):
		# making a trend based on nobs in arbitrary series in dictionary
		self.data[tname] = arange(self.nobs)

	def add_dummy(self,dum, dname = 'dummy'):
		if self.data.has_key(dname):
			print "The variable name '" + str(dname) + "' already exists. Please select another name."
		else:
			self.data[dname] = dum

	def add_seasonal_dummies(self,freq=52,ndum=13):
		"""
		This function will only work if the freq and ndum 'fit. That is,
		weeks and 4-weekly periods will work. Weeks and months/quarters
		will not.
		"""
		if self.date_key == []:
			print "Cannot create seasonal dummies since no date array is known"
		else:
			# list of years
			years = array([pylab.num2date(i).year for i in self.data[self.date_key]])

			# how many periods in does the data start
			start = freq - sum(years ==	min(years))

			# how many unique years
			nyear = unique(years).shape[0]

			# using kronecker products to make a big dummy matrix
			sd = kron(ones(nyear),kron(eye(ndum),ones(freq/ndum))).T;
			sd = sd[start:start+self.nobs]		# slicing the dummies to fit the data	
			sd = dict([(("sd"+str(i+1)),sd[:,i]) for i in range(1,ndum)])
			self.data.update(sd)				# adding the dummies to the main dict

	def delvar(self,*var):
		"""
		Deleting specified variables in the data dictionary, changing dictionary in place
		"""
		[self.data.pop(i) for i in var]

	def keepvar(self,*var):
		"""
		Keeping specified variables in the data dictionary, changing dictionary in place
		"""
		[self.data.pop(i) for i in self.data.keys() if i not in var]

	def delvar_copy(self,*var):
		"""
		Deleting specified variables in the data dictionary, making a copy
		"""
		return dict([(i,self.data[i]) for i in self.data.keys() if i not in var])

	def keepvar_copy(self,*var):
		"""
		Keeping specified variables in the data dictionary, making a copy
		"""
		return dict([(i,self.data[i]) for i in var])

	def delobs(self,sel):
		"""
		Deleting specified observations, changing dictionary in place
		"""
		for i in self.data.keys(): self.data[i] = self.data[i][sel]

		# updating the value of self.nobs
		self.nobs -= sum(sel)

	def keepobs(self,sel):
		"""
		Keeping specified observations, changing dictionary in place
		"""
		# updating the value of self.nobs
		self.nobs -= sum(sel)

		sel -= 1				# making true, false and vice-versa
		self.delobs(sel)

	def delobs_copy(self,sel):
		"""
		Deleting specified observations, making a copy
		"""
		return dict([(i,self.data[i][sel]) for i in self.data.keys()])

	def keepobs_copy(self,sel):
		"""
		Keeping specified observations, making a copy
		"""
		sel -= 1				# making true, false and vice-versa
		self.delobs_copy(sel)

	def get(self,*var,**sel):
		"""
		Copying data and keys of selected variables for further analysis
		"""
		# calling convenience function to clean-up input parameters
		var, sel = self.__var_and_sel_clean(var, sel)

		# copying the entire dictionary (= default)
		d = dict((i,self.data[i][sel]) for i in var)

		return d.keys(), array(d.values()).T

	def info(self,*var, **adict):
		"""
		Printing descriptive statistics on selected variables
		"""
			
		# calling convenience functions to clean-up input parameters
		var, sel = self.__var_and_sel_clean(var, adict)
		dates, nobs = self.__dates_and_nobs_clean(var, sel)
			
		# setting the minimum and maximum dates to be used
		mindate = pylab.num2date(min(dates)).strftime('%d %b %Y')
		maxdate = pylab.num2date(max(dates)).strftime('%d %b %Y')

		# number of variables (excluding date if present)
		nvar = len(var)

		print '\n=============================================================='
		print '==================== Database information ===================='
		print '==============================================================\n'

		print 'file:				%s' % b.DBname
		print '# obs:				%s' % nobs
		print '# variables:		%s' % nvar 
		print "Start date:			%s" % mindate
		print "End date:			%s" % maxdate

		print '\nvar				min			max			mean		std.dev'
		print '=============================================================='
		
		for i in var:
			_min = self.data[i][sel].min(); _max = self.data[i][sel].max(); _mean = self.data[i][sel].mean(); _std = self.data[i][sel].std()
			print '''%-5s			%-5.2f		%-5.2f		%-5.2f		%-5.2f''' % tuple([i,_min,_max,_mean,_std]) 
	
	def dataplot(self,*var, **adict):
		"""
		Plotting the data with variable names
		"""
		# calling convenience functions to clean-up input parameters
		var, sel = self.__var_and_sel_clean(var, adict)
		dates, nobs = self.__dates_and_nobs_clean(var, sel)

		for i in var:
			pylab.plot_date(dates,self.data[i][sel],'o-') 

		pylab.xlabel("Time (n = " + str(nobs) + ")") 
		pylab.title("Data plot of " + self.DBname)
		pylab.legend(var)
		if adict.has_key('file'):
			pylab.savefig(adict['file'],dpi=600)
		pylab.show()

	def __var_and_sel_clean(self, var, sel, dates_needed = True):
		"""
		Convenience function to avoid code duplication
		"""
		# find out if a variable list is provided
		if var == ():
			var = self.data.keys()
			
		# removing the date variable if it is present
		var = [x for x in var if x != self.date_key]

		# report variable label in alphabetical order
		var.sort()

		# find out if a selection rule is being used
		# if not, set to empty tuple
		if not sel.has_key('sel'):
			sel = ()
		else:
			sel = sel['sel']

		return var, sel

	def __dates_and_nobs_clean(self, var, sel):
		"""
		Convenience function to avoid code duplication
		"""
		nobs = self.nobs
		if len(sel):
			nobs = nobs - (nobs - sum(sel))

		if self.date_key != None and self.data.has_key(self.date_key):
			# selecting dates from data base
			dates = self.data[self.date_key][sel]
		else:
			# setting date series to start on 1/1/1950
			dates = range(711858,nobs+711858)

		return dates, nobs

	def __ext(self,fname):
		"""
		Finding the file extension of the filename passed to dbase
		"""
		return fname.split('.')[-1].strip()

if __name__ == '__main__':

	########################
	### Testing dbase class
	########################

	import sys
	from scipy import c_

	# making a directory to store simulate data
	if not os.path.exists('./dbase_test_files'): os.mkdir('./dbase_test_files')

	# creating simulated data and variable labels
	varnm = ['date','a','b','c']			# variable labels
	nobs = 100
	data =	randn(nobs,3)					# the data array
	dates = pylab.num2date(arange(730493,730493+(nobs*7),7))
	dates = [i.strftime('%d %b %y') for i in dates]
	data = c_[dates,data]

	# saving simulated data to a csv file
	f = open('./dbase_test_files/data.csv','w')
	writer = csv.writer(f)
	writer.writerow(varnm)
	writer.writerows(data)
	f.close()

	# loading the data from the csv file
	a = dbase("./dbase_test_files/data.csv",date = 0)
	# saving the dbase instance data to a pickle file
	a.save("./dbase_test_files/data.pickle")
	# saving the dbase data to a shelve file
	### a.save("./dbase_test_files/data.she")

	# loading a sub-section of the data from a shelve file
	### print "\nLoading 2 variables from a shelve file\n"
	### b = dbase("./dbase_test_files/data.she",'a','b',date = 'date')

	# showing data and variable names, from load_shelve
	### varnm, data = b.get()
	### print "Variable names from shelve file\n", varnm
	### print "\nData selected from shelve file\n", data
	### print "\nDate series", b.data[b.date_key]
	### del b		# cleaning up

	# loading the object from the pickle file
	print "\nLoading the dbase object from a pickle file\n"
	b = dbase("./dbase_test_files/data.pickle")

	# getting the name of the file you are working on
	print "\nWorking on file: " + b.DBname

	# showing data and variable names
	varnm, data = b.get()
	print "Variable names from dbase class\n", varnm
	print "\nData from dbase class\n", data
	print "\nDate series", b.data[b.date_key]

	# viewing selected data columns
	varnm, data = b.get('a','c')
	print "\nTwo columns selected using variable names\n", varnm, "\n", data 

	# saving to a csv file
	print "\nSaving data and variable names to a different csv file\n", b.save("./dbase_test_files/data_save.csv")

	# adding variables/data
	x1 = b.data['a'] * b.data['b']
	x2 = b.data['a'] * b.data['c']
	xdict = {'x1':x1,'x2':x2}
	b.data.update(xdict)				# using a dictionaries own 'add/extend method'

	varnm, data = b.get()
	print "\nTwo variable names added\n", varnm
	print "\nTwo columns added\n", data

	# using copy.deepcopy to make a complete copy of the class instance data
	import copy
	c = copy.deepcopy(b)

	# making the database smaller, inplace, by deleting selected variables
	c.delvar('a','x2')
	varnm, data = c.get()
	print "\nTwo variable names deleted\n", varnm
	print "\nTwo columns deleted\n", data

	# making the database smaller, inplace, by keeping only selected variables
	c = copy.deepcopy(b)
	c.keepvar('a','x2')
	varnm, data = c.get()
	print "\nAll but two variable names deleted\n", varnm
	print "\nAll but Two columns deleted\n", data

	# specifying a selection rule
	sel_rule = b.data['date'] > pylab.datestr2num("8/1/2001")

	# making the database smaller, inplace, by delecting selected observation
	c = copy.deepcopy(b)
	c.delobs(sel_rule)

	varnm, data = c.get()
	print "\nReduced number of observations following the selection rule\n", data

	# making the database smaller, inplace, by delecting all but the selected observation
	c = copy.deepcopy(b)
	c.keepobs(sel_rule)

	varnm, data = c.get()
	print "\nReduced number of observations following the inverse of the selection rule\n", data

	# making a copy of of just the dictionary for selected variables
	x = b.keepvar_copy('a')

	# making a copy of of just the dictionary for everything but the selected variables
	x = b.delvar_copy('a')

	# making a copy of of just the dictionary for selected observations
	x = b.keepobs_copy(sel_rule)

	# making a copy of of just the dictionary for everything but the selected observation
	x = b.delobs_copy(sel_rule)

	# descriptive information on the database
	b.info()

	# plotting series
	b.dataplot(file = './dbase_test_files/full_plot.png')

	# adding a trend component
	b.add_trend('mytrend')			# or b.data.update({'mytrend':range(100)})

	# adding a dummy
	dummy_rule = b.data['a'] > 0
	b.add_dummy(dummy_rule,'mydummy')			# or b.data.update({'mydummy':dummy_rule})

	# add seasonal dummies
	b.add_seasonal_dummies(52,13)

	# descriptive information on the database for selected time period
	b.info('b','c', sel = sel_rule)

	# plotting series
	b.dataplot('b','c', sel = sel_rule, file = './dbase_test_files/partial_plot.png')

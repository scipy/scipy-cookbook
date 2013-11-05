from scipy import c_, arange
from scipy.io import read_array
from numpy.random import randn
from pylab import plot, show, figure
import pickle, csv, os

class dbase:
	"""
	A simple data-frame, that reads and write csv/pickle files with variable names.
	Columns in the data can be accessed using x.get('a','c') where 'a' and 'c' are
	variable names.
	"""
	def __init__(self,f):
		"""
		Initializing the dbase class. Loading file f.
		"""
		self.load(f)
		self.DBname = os.getcwd() + '/' + f

	def load(self,fname):
		"""
		Loading data from a csv or a pickle file of the dbase class
		"""
		fext = self.__ext(fname)
		f = open(fname,'r')
		if fext == 'csv':
			self.varnm = self.__vardic(f.readline().split(','))
			self.data = read_array(f, separator=',', lines=(0,-1))
		elif fext == 'pickle':
			a = pickle.load(f)
			self.varnm = a.varnm
			self.data = a.data
		else:
			raise 'This class only works on csv and pickle files'
		f.close()

	def dump(self,fname):
		"""
		Dumping the instance of the class into a csv or pickle file
		"""
		fext = self.__ext(fname)
		f = open(fname,'w')
		if fext == 'csv':
			writer = csv.writer(f)
			writer.writerow(self.__sort_keys())
			writer.writerows(self.data)
		elif fext == 'pickle':
			pickle.dump(self,f)
		else:
			raise 'This class only outputs csv or pickle files'
		f.close()

	def get(self,*var):
		"""
		Selecting a column based on variable labels. Assumes data are in columns.
		"""

		a = self.data[:,self.varnm[var[0]]]				# getting the data for the 1st element in self.data

		for i in var[1:]:						
			a = c_[a,self.data[:,self.varnm[i]]]		# concatenate column-wise, along last axis
	
		return a

	def addvar(self,a,v):
		"""
		Adding columns of data
		"""
		self.data = c_[self.data,a]			# concatenation the data at end

		j = max(self.varnm.values()) + 1	# starting index past max index
		if isinstance(v,str): v = [v]
		for i in v:						
			self.varnm[i] = j
			j += 1

	def delvar(self,*v):
		"""
		Deleting columns of data
		"""
		# removing the variables listed 
		for i in v:						
			del self.varnm[i]

		# index list for the remaining variables
		index = self.varnm.values()
		index.sort()

		# selecting the remain columns
		self.data = self.data[:,index]

		# updating the index number 
		self.varnm = self.__vardic(self.__sort_keys(range(len(index))))

	def info(self,axis=0):
		"""
		Printing descriptive statistics on selected variables
		"""
		nobs = self.data.shape[axis]
		nvar = len(self.varnm.keys())
		min = self.data.min(axis)
		max = self.data.max(axis)
		mean = self.data.mean(axis)
		std = self.data.std(axis)
		vars = self.__sort_keys()
		
		print '\n=========================================================='
		print '================== Database information =================='
		print '==========================================================\n'

		print '''file:			%s''' % b.DBname
		print '''# obs:			%s''' % nobs
		print '''# variables:	%s\n''' % nvar

		print 'var			min			max			mean		std.dev'
		print '=========================================================='
		
		for i in range(nvar):
			print '''%s			%-5.2f		%-5.2f		%-5.2f		%-5.2f''' % tuple([vars[i],min[i],max[i],mean[i],std[i]]) 

	def dataplot(self,var):
		"""
		Plotting the data with variable names
		"""
		a = self.get(var)

		# plot a single column
		title = "Plot of series " + var
		ax = figure().add_axes([.1,.1,.8,.8])
		ax.plot(a); 
		ax.set_title(title)
		show()

	def __vardic(self,L):
		"""
		Making a dictionary with variable names and indices
		"""
		dic = {}; j = 0

		# reading only the 1st line in the file and extracting variables names
		# names are linked in the dictionary to their, and the data's, index
		# making sure to strip leading and trailing white space
		for i in L:
			dic[i.strip()] = j
			j += 1
	
		return dic

	def __ext(self,fname):
		"""
		Finding the file extension of the filename passed to dbase
		"""
		return fname.split('.')[-1].strip()

	def __sort_keys(self,v = []):
		"""
		Sorting the keys in the variable name dictionary so they are in the correct order
		"""
		k = self.varnm.keys()
		if v == []: v = self.varnm.values()

		return [k[i] for i in v]

########################
### Testing the class
########################

if __name__ == '__main__':

	# creating simulated data and variable labels
	varnm = ['a','b','c']			# variable labels
	data =	randn(5,3)				# the data array

	# saving simulated data to a csv file
	f = open('data.csv','w')
	writer = csv.writer(f)
	writer.writerow(varnm)
	writer.writerows(data)
	f.close()

	# loading the data from the csv file and dumping the dbase class instance to a pickle file
	a = dbase("data.csv")
	a.dump("data.pickle")

	# loading the object from the pickle file
	print "\nLoading the dbase object from a pickle file\n"

	b = dbase("data.pickle")

	print "Data from dbase class\n", b.data
	print "\nVariable names from dbase class\n", b.varnm
	print "\nTwo columns selected using variable names\n", b.get('a','c')
	print "\nSaving data and variable names to a different csv file\n", b.dump("data_dump.csv")

	# making the database bigger
	xtra1 = b.get('a') * b.get('b')
	xtra2 = b.get('a') * b.get('c')
	xtra = c_[xtra1,xtra2]
	xtra_varnm = ('x1','x2')

	b.addvar(xtra,xtra_varnm)
	print "\nTwo columns added\n", b.data
	print "\nTwo variable names added\n", b.varnm

	# making the database smaller
	b.delvar('a','x2')
	print "\nTwo columns deleted\n", b.data
	print "\nTwo variable names deleted\n", b.varnm

	# getting the name of the file you are working on
	print "\nWorking on file: " + b.DBname

	# descriptive information on the database, or selected variables in the databse
	b.info()

	# plotting a series
	b.dataplot('b')

## Copyright (c) 2001-2006, Andrew Straw. All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:

##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.

##     * Redistributions in binary form must reproduce the above
##       copyright notice, this list of conditions and the following
##       disclaimer in the documentation and/or other materials provided
##       with the distribution.

##     * Neither the name of the Andrew Straw nor the names of its
##       contributors may be used to endorse or promote products derived
##       from this software without specific prior written permission.

## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import sets
from scipy import *
import Numeric
import cStringIO as StringIO
import csv # with Python 2.3
class access2000(csv.Dialect):
	"""A dialect to properly interpret Microsoft Access2000 CSV exports for international languages.
	"""
	delimiter = ';'
	quotechar = '"'
	doublequote = True
	quoting = csv.QUOTE_NONNUMERIC
	lineterminator = '\n'
	skipinitialspace = True

class DataFrame:
    def __init__(self, value_dict=None, fields_order=None ):
        if value_dict is None:
            value_dict = {}
        self.value_dict = value_dict
        num_rows = 0
        for column in self.value_dict.values():
            try:
                num_rows = max(num_rows,len(column))
            except:
                pass
        for key in self.value_dict.keys():
            if self.value_dict[key] is None:
                self.value_dict[key] = (None,)*num_rows
        for field in self.value_dict:
            if len(self.value_dict[field]) != num_rows:
                raise ValueError("field %s has wrong number of rows"%str(field))
        self.num_rows = num_rows
        if fields_order is None:
            self.fields_order = self.value_dict.keys()
        else:
            for key in fields_order:
                assert self.value_dict.has_key(key)
            self.fields_order = fields_order

    def copy_empty(self):
        vd = {}
        for k in self.value_dict.keys():
            vd[k] = None
        return DataFrame(vd,self.fields_order)

    def __add__(self,other):
        nd = {}
        for k in other.fields_order:
            if k not in self.fields_order:
                raise NotImplementedError("no fix yet for when not all fields are in both frames")
        for k in self.fields_order:
            nd[k] = list(self.value_dict[k]) + list(other.value_dict[k])
        res = DataFrame(nd,self.fields_order)
        return res

    def insert_row(self, value_dict, new_fields_ok=False):
        if not new_fields_ok:
            for v in value_dict:
                assert(v in self.fields_order)
            for v in self.fields_order:
                assert(v in value_dict)
                try:
                    self.value_dict[v].append( value_dict[v] )
                except AttributeError:
                    tmp = list(self.value_dict[v])
                    tmp.append( value_dict[v] )
                    self.value_dict[v] = tmp
            self.num_rows += 1
        else:
            all_fields = list(sets.Set(self.fields_order).union(value_dict.keys()))
            all_fields.sort()
            all_fields.reverse()
            for v in all_fields:
                if v in value_dict:
                    if v not in self.value_dict:
                        self.value_dict[v] = [None]*self.num_rows
                    try:
                        self.value_dict[v].append(value_dict[v])
                    except AttributeError:
                        tmp = list(self.value_dict[v])
                        tmp.append( value_dict[v] )
                        self.value_dict[v] = tmp
                else:
                    try:
                        self.value_dict[v].append(None)
                    except AttributeError:
                        tmp = list(self.value_dict[v])
                        tmp.append( None )
                        self.value_dict[v] = tmp
            self.fields_order = all_fields
            self.num_rows += 1

    def insert_column(self, field_name, values, position='last'):
        assert len(values) == self.num_rows
        if position == 'last':
            self.fields_order.append(field_name)
        else:
            self.fields_order.insert(position,field_name)
        self.value_dict[field_name] = values

    def drop_column(self, field_name):
        self.fields_order.remove(field_name)
        del self.value_dict[field_name]

    def drop_all_columns_except(self, *field_names):
        save_names = list(field_names)
	for field_name in self.fields_order[:]:
            if field_name not in save_names:
                self.drop_column( field_name )

    def __str__(self):
        def cc(s,width=10,just='center'):
            if len(s) > width:
                s = s[:width]
            if just=='center':
                return s.center(width)
            elif just=='left':
                return s.ljust(width)
            elif just=='right':
                return s.rjust(width)
        buf = StringIO.StringIO()
        print >> buf, cc('',width=5,just='right'),
        for field in self.fields_order:
            print >> buf, cc( field),
        print >> buf
        for row in range(self.num_rows):
            print >> buf, cc(str(row),width=5,just='right'), # row number
            for field in self.fields_order:
                v = self.value_dict[field][row]
                if v is not None:
                    v_str = str(v)
                else:
                    v_str = ''
                print >> buf, cc( v_str ),
            print >> buf
        buf.seek(0)
        return buf.read()

    def get_row_dict(self,row_idx):
        return self[row_idx]

    def __getitem__(self, i):
        result = {}
        for field in self.fields_order:
            result[field] = self.value_dict[field][i]
        return result

    def __getitems__(self, idxs):
        result = []
        for i in idxs:
            result.append( self[i] )
        return result

    def __len__(self):
        return self.num_rows

    def __get_row(self,row_idx):
        result = []
        for field in self.fields_order:
            result.append( self.value_dict[field][row_idx] )
        return result

    def __get_rows(self,row_idxs):
        if len(row_idxs) == 0:
            return
        rows = [self.__get_row(row_idx) for row_idx in row_idxs]
        by_col = zip(*rows)
        result = {}
        for i,field in enumerate(self.fields_order):
            result[field] = by_col[i]
        return result

    def where_field_cmp(self, field, bool_func_of_value):
        col = self.value_dict[field]
        indices = []
        for i in range(len(col)):
            if bool_func_of_value(col[i]):
                indices.append(i)
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_equal(self, field, value, eps=None):
        col = self.value_dict[field]
        indices = []
        if eps is None:
            for i in range(len(col)):
                if col[i] == value:
                    indices.append(i)
        else:
            # this is probably faster because it assumes data is numeric
            a = asarray(col)
            abs_diff = abs(a-value)
            indices = nonzero( less( abs_diff, eps ) )
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_not_equal(self, field, value, eps=None):
        col = self.value_dict[field]
        indices = []
        if eps is None:
            for i in range(len(col)):
                if col[i] != value:
                    indices.append(i)
        else:
            # this is probably faster because it assumes data is numeric
            a = numpy.asarray(col)
            abs_diff = abs(a-value)
            indices = numpy.nonzero( numpy.greater_equal( abs_diff, eps ) )
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_less(self, field, value):
        col = self.value_dict[field]
        indices = []
        for i in range(len(col)):
            if col[i] < value:
                indices.append(i)
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_lessequal(self, field, value):
        col = self.value_dict[field]
        indices = []
        for i in range(len(col)):
            if col[i] <= value:
                indices.append(i)
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_greater(self, field, value):
        col = self.value_dict[field]
        indices = []
        for i in range(len(col)):
            if col[i] > value:
                indices.append(i)
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_greaterequal(self, field, value):
        col = self.value_dict[field]
        indices = []
        for i in range(len(col)):
            if col[i] >= value:
                indices.append(i)
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def where_field_in(self, field, values):
        col = self.value_dict[field]
        indices = []
        for i in range(len(col)):
            if col[i] in values:
                indices.append(i)
        results = self.__get_rows(indices)
        if results is not None:
            return DataFrame(results,fields_order=self.fields_order)
        else:
            return None

    def enumerate_on(self, field, cmp_func=None):
        values = self.get_unique_values(field)
        result_frames = []
        values.sort(cmp_func)
        for value in values:
            result_frames.append( (value, self.where_field_equal(field,value)))
        return iter(result_frames)

    def enumerate_crude_bins(self, field, eps=None, eps_domain='linear'):
        if eps is None:
            return self.enumerate_on(field)
        if eps_domain == 'linear':
            def filt(x):
                return x
        elif eps_domain == 'log10':
            def filt(x):
                return numpy.log10(x)
        else:
            raise NotImplementedError
        vs = self.get_unique_values(field)
        vs.sort()
        bins = {}
        current_starter_v = None
        for cv in vs:
            if current_starter_v is not None and abs(filt(cv)-filt(current_starter_v))<eps:
                bins[current_starter_v].append(cv)
            else:
                bins.setdefault(cv,[]).append(cv)
                current_starter_v = cv
        results = []
        keys = bins.keys()
        keys.sort()
        for close_v in keys:
            running_sum = 0
            running_n = 0
            accum = self.copy_empty() # new DataFrame with same fields
            for v in bins[close_v]:
                add_frame = self.where_field_equal(field,v)
                n = add_frame.num_rows
                running_sum += (v*n)
                running_n += n
                accum = accum + add_frame
            avg_value = running_sum/float(running_n)
            results.append(( avg_value, accum ))
        return iter(results)

    def enumerate_crude_2_dims(self, field1, field2,
                               eps1=None, eps2=None,
                               eps1_domain='linear', eps2_domain='linear'):
        axis1_vs = {}
        for v,vf in self.enumerate_crude_bins(field1,eps=eps1,eps_domain=eps1_domain):
            axis1_vs[v] = vf.get_unique_values(field1)

        axis2_vs = {}
        for v,vf in self.enumerate_crude_bins(field2,eps=eps2,eps_domain=eps2_domain):
            axis2_vs[v] = vf.get_unique_values(field2)

        results = []

        v1s = axis1_vs.keys()
        v1s.sort()

        v2s = axis2_vs.keys()
        v2s.sort()
        for v1 in v1s:
            for v2 in v2s:
                this_result = self.copy_empty()
                for v1r in axis1_vs[v1]:
                    for v2r in axis2_vs[v2]:
                        tmp1 = self.where_field_equal(field1,v1r)
                        if tmp1 is not None:
                            tmp2 = tmp1.where_field_equal(field2,v2r)
                            if tmp2 is not None:
                                this_result = this_result + tmp2
                if this_result.num_rows > 0:
                    results.append((v1,v2,this_result))
        return iter(results)

    def mean(self, field):
        return mean(self.value_dict[field])

    def mean_and_std(self, field):
        values = self.value_dict[field]
        return mean(values), std(values)

    def mean_and_sem(self, field):
        values = self.value_dict[field]
        n = len(values)
        return mean(values), std(values)/float(numpy.sqrt(n))

    def get_fields(self):
        return self.fields_order[:] # return copy

    def sort_by(self,field,ascending=True):
        orig = self.value_dict[field]
        sorted = list(orig[:]) # play with copy
        sorted.sort() # make it sorted
        if not ascending:
            sorted.reverse() # make it sorted
        my_copy = list(orig[:]) # play with copy
        new_order = []
        for i in range(len(sorted)):
            sorted_value = sorted[i]
            index = my_copy.index( sorted_value )
            new_order.append( index )
            my_copy[index] = int # just set it equal to anything that can't be in list so it won't match again
        for field in self.fields_order:
            orig_list = self.value_dict[field]
            new_list = []
            for i in new_order:
                new_list.append( orig_list[i] )
            self.value_dict[field] = new_list

    def sorted(self,field,ascending=True):
        result = DataFrame(self.value_dict, self.fields_order)
        result.sort_by(field,ascending=ascending)
        return result

    def get_unique_values(self, field):
        unique_values = list(sets.Set(self.get_all_values(field)))
        try:
            unique_values.sort()
        except:
            pass
        return unique_values

    def get_all_values(self, field):
        values = [ v for v in self.value_dict[field] ]
        return values

    def write_csv(self,filename,dialect='excel'):
        writer = csv.writer(open(filename,'w'),dialect=dialect)
        writer.writerow( self.fields_order )
        for i in range(self.num_rows):
            writer.writerow( self.__get_row(i) )

def read_csv(file,header=True,dialect='excel'):
    if not hasattr(file,'readlines'):
        # inspired by R's read.csv
        reader = csv.reader(open(file,"r"),dialect=dialect)
    else:
        reader = csv.reader(file,dialect=dialect)
    split_lines = [ row for row in reader ] # newline problems sometimes happen here - de-Macify...
    if header:
        fields = split_lines.pop(0)
    else:
        num_fields = len(split_lines[0])
        num_chars = int(math.ceil(math.log10(num_fields)))
        name_string = "column%0"+str(num_chars)+"d"
        fields = [ name_string%(i+1) for i in range(len(split_lines[0])) ]
    # check/convert values
    for i in range(len(split_lines)):
        split_line = split_lines[i]
        for j in range(len(split_line)):
            value = split_line[j]
            if value == '':
                converted_value = None
            else:
                try:
                    converted_value = int(value)
                except ValueError:
                    try:
                        converted_value = float(value)
                    except:
                        converted_value = value
            split_line[j] = converted_value
        split_lines[i] = split_line
    columns = zip(*split_lines)
    results = {}
    for i in range(len(fields)):
        field = fields[i]
        try:
            results[field] = list(columns[i])
        except IndexError:
            results[field] = None
#        print i, field, results[field], columns[i]
    return DataFrame( results, fields_order=fields )

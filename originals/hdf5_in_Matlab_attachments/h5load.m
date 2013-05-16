function data=h5load(filename, path)
%
% data = H5LOAD(filename)
% data = H5LOAD(filename, path_in_file)
%
% Load data in a HDF5 file to a Matlab structure.
%
% Parameters
% ----------
%
% filename
%     Name of the file to load data from
% path_in_file : optional
%     Path to the part of the HDF5 file to load
%

% Author: Pauli Virtanen <pav@iki.fi>
% This script is in the Public Domain. No warranty.

if nargin > 1
  path_parts = regexp(path, '/', 'split');
else
  path = '';
  path_parts = [];
end

loc = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
try
  data = load_one(loc, path_parts, path);
  H5F.close(loc);
catch exc
  H5F.close(loc);
  rethrow(exc);
end


function data=load_one(loc, path_parts, full_path)
% Load a record recursively.

while ~isempty(path_parts) & strcmp(path_parts{1}, '')
  path_parts = path_parts(2:end);
end

data = struct();

num_objs = H5G.get_num_objs(loc);

% 
% Load groups and datasets
%
for j_item=0:num_objs-1,
  objtype = H5G.get_objtype_by_idx(loc, j_item);
  objname = H5G.get_objname_by_idx(loc, j_item);
  
  if objtype == 1
    % Group
    name = regexprep(objname, '.*/', '');
  
    if isempty(path_parts) | strcmp(path_parts{1}, name)
      if ~isempty(regexp(name,'^[a-zA-Z].*'))
	group_loc = H5G.open(loc, name);
	try
	  sub_data = load_one(group_loc, path_parts(2:end), full_path);
	  H5G.close(group_loc);
	catch exc
	  H5G.close(group_loc);
	  rethrow(exc);
	end
	if isempty(path_parts)
	  data = setfield(data, name, sub_data);
	else
	  data = sub_data;
	  return
	end
      end
    end
   
  elseif objtype == 2
    % Dataset
    name = regexprep(objname, '.*/', '');
  
    if isempty(path_parts) | strcmp(path_parts{1}, name)
      if ~isempty(regexp(name,'^[a-zA-Z].*'))
	dataset_loc = H5D.open(loc, name);
	try
	  sub_data = H5D.read(dataset_loc, ...
	      'H5ML_DEFAULT', 'H5S_ALL','H5S_ALL','H5P_DEFAULT');
	  H5D.close(dataset_loc);
	catch exc
	  H5D.close(dataset_loc);
	  rethrow(exc);
	end
	
	sub_data = fix_data(sub_data);
	
	if isempty(path_parts)
	  data = setfield(data, name, sub_data);
	else
	  data = sub_data;
	  return
	end
      end
    end
  end
end

% Check that we managed to load something if path walking is in progress
if ~isempty(path_parts)
  error(sprintf('Path "%s" not found in the HDF5 file', full_path));
end


function data=fix_data(data)
% Fix some common types of data to more friendly form.

if isstruct(data)
  fields = fieldnames(data);
  if length(fields) == 2 & strcmp(fields{1}, 'r') & strcmp(fields{2}, 'i')
    if isnumeric(data.r) & isnumeric(data.i)
      data = data.r + 1j*data.i;
    end
  end
end

if isnumeric(data) & ndims(data) > 1
  % permute dimensions
  data = permute(data, fliplr(1:ndims(data)));
end

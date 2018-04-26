#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""



import pickle
import os


path = "C:\\Users\\Administrator\\Documents\\GitHub\\my_ud120\\naive_bayes"
os.chdir(path)

#########################################################
### word_data to unix ###

original = "../final_project/final_project_dataset.pkl"
destination = "../final_project/final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))

### preprocess ###

pkl_file = open('../final_project/final_project_dataset_unix.pkl', 'rb')
dataset = pickle.load(pkl_file)
pkl_file.close()



# Open the file with read only permit
f = open('../final_project/poi_names.txt ', "r")
# use readlines to read all lines in the file
# The variable "lines" is a list containing all lines in the file
lines = f.readlines()[2:]
# close the file after reading the lines.
f.close()


### answers ###

len(dataset)

len(dataset["ALLEN PHILLIP K"])

n = 0
for k, v in dataset.items():
    n = n + v["poi"]
print(n)

len(lines)

dataset["PRENTICE JAMES"]['total_stock_value']

dataset["COLWELL WESLEY"]["from_this_person_to_poi"]
        
dataset["SKILLING JEFFREY K"]["exercised_stock_options"]


Jeffrey Skilling 



SKILLING JEFFREY K















import csv
import numpy as np
import pandas as pd

def create_csv(path):
    path = path#"classification_test.csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["filename","backpack","bag","handbag","clothes","down","up","hair","hat","gender","upblack","upwhite","upred","uppurple","upyellow","upgray","upblue","upgreen","downblack","downwhite","downpink","downpurple","downyellow","downgray","downblue","downgreen","downbrown","upmulticolor","downmulticolor","young","teenager","adult","old"]
        csv_write.writerow(csv_head)


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def get_csv(csv_path):

    testcsv = pd.read_csv(csv_path, sep=',', header=0)
    testcsv.columns = ["filename","backpack","bag","handbag","clothes","down","up","hair","hat","gender","upblack","upwhite","upred","uppurple","upyellow","upgray","upblue","upgreen","downblack","downwhite","downpink","downpurple","downyellow","downgray","downblue","downgreen","downbrown","upmulticolor","downmulticolor","0","1","2","3"]
    testcsv["age"] = np.ones(19679).astype(int)
    testcsv["age"] = testcsv.iloc[:, 29:33].idxmax(axis="columns").astype(int)
    testcsv = testcsv.drop("0",axis=1) 
    testcsv = testcsv.drop("1",axis=1)
    testcsv = testcsv.drop("2",axis=1)
    testcsv = testcsv.drop("3",axis=1)
    testcsv = testcsv.reindex(columns=["filename","age","backpack","bag","handbag","clothes","down","up","hair","hat","gender","upblack","upwhite","upred","uppurple","upyellow","upgray","upblue","upgreen","downblack","downwhite","downpink","downpurple","downyellow","downgray","downblue","downgreen","downbrown","upmulticolor","downmulticolor"])
    testcsv.iloc[:, 1:] +=1
    testcsv.to_csv('classification_test.csv', index=False)#
    #print(testcsv)


    

#get_csv('test.csv')
        

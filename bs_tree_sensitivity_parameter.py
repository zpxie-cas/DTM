# This Pyhton script was used to test the sensitivity of DTM to the selected feature variables
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO  
from IPython.display import Image  
import pydotplus
import numpy as np

col_names1    = ['site_name','timestamp', 'ws', 'ws_max','ws_5','ws_5_max','wd_min','wd_max','wd', 't', 'rh', 'precp', 'sd1','sd2','snow_drift','snow_drift_modify','if_snowfall', 'if_snow_cover', 'if_dry_snow','label']
col_names2    = ['timestamp', 'ws', 'ws_max','ws_5','ws_5_max','wd_min','wd_max','wd', 't', 'rh', 'precp', 'sd1','sd2','snow_drift','snow_drift_modify','if_snowfall', 'if_snow_cover', 'if_dry_snow','label']

sta_name     = ['all','fmor1','fcmb1','fber1','fhue1','fgie1','fmon1','fche1','fbon1','fcel1']
files        = ['_snowfall.csv','_no_snowfall.csv','_no_snowfall_no_snow.csv','_no_snowfall_with_snow.csv','_no_snowfall_with_snow_wet.csv','_no_snowfall_with_snow_dry.csv']
files_all     = ['_snowfall_rh.csv','_no_snowfall_rh.csv','_no_snowfall_no_snow_rh.csv','_no_snowfall_with_snow_rh.csv','_no_snowfall_with_snow_wet_rh.csv','_no_snowfall_with_snow_dry_rh.csv']
feature_cols1 = [['ws_5'],['ws_5_max'],['t'],['ws_5', 'ws_5_max'],['ws_5', 't'],['ws_5_max','t'],['ws_5','ws_5_max','t']]
feature_cols2 = [['ws_5'],['ws_5_max'],['t'],['ws_5', 'ws_5_max'],['ws_5', 't'],['ws_5_max','t'],['ws_5','ws_5_max','t'],['rh'],['ws_5','ws_5_max','t','rh']]

accuracy_max = 0.0
accuracy_min = 0.0

#print(accuracy_list)
file = open('accuracy_list.csv','w')
file.close()

file_count = 0
for i_file in files:
    count        = 0
    for i_sta in sta_name:
        print(i_file+" "+i_sta)

        if(count == 0):
            pima = pd.read_csv("../datafile/"+i_sta+i_file, header=0, names=col_names1)
        else:
            pima = pd.read_csv("../datafile/"+i_sta+i_file, header=0, names=col_names2)
        #pima = pd.read_csv("../datafile/"+i_sta+"_no_snowfall_with_snow_dry_rh.csv", header=0, names=col_names)
        pima.head() 

        if(count <4):
            feature_cols = feature_cols2
        else:
            feature_cols = feature_cols1
        
        fea_count =0
        for i_fea in feature_cols:
            #if(count == 0 and fea_count >6):
            if(count == 0):
               pima = pd.read_csv("../datafile/"+i_sta+files_all[file_count], header=0, names=col_names1)
                
            X = pima[feature_cols[fea_count]] # Features
            y = pima.label # Target variable        

            features_string = i_file+","+i_sta+","
            for i in feature_cols[fea_count]:
                features_string = features_string+i+"="     

            for i_count in range(0,20):                 

                # Split dataset into training set and test set
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)                  
                clf = DecisionTreeClassifier(max_depth=len(feature_cols)+1)
                #else:
                #	clf = DecisionTreeClassifier(max_depth=5)         

                # Train Decision Tree Classifer
                clf = clf.fit(X_train,y_train)                  

                y_pred = clf.predict(X_test)
                      	
               # print(X_test,y_test,y_pred)
                aa= metrics.accuracy_score(y_test, y_pred)
                #print(str(aa))
                if(i_count == 0):
                	ss=features_string+","+str(aa)
                else:
                    ss=ss+','+str(aa)       	           

                if(i_count == 0):
                	accuracy_min = aa
                	clf_min      = clf
                	dot_data_min = StringIO()              

                	accuracy_max = aa
                	clf_max      = clf
                	dot_data_max = StringIO()
                else:
                	if(aa > accuracy_max):
                		accuracy_max = aa
                		clf_max      = clf
                		dot_data_max = StringIO()
                	else:
                		accuracy_min = aa
                		clf_min      = clf
                		dot_data_min = StringIO() 

            fea_count = fea_count+1
            df= pd.DataFrame([ss])
            df.to_csv('accuracy_list.csv', mode='a', header=False)
        
        count = count+1  
    file_count = file_count+1          
    #        dot_data = StringIO()
    #        export_graphviz(clf, out_file=dot_data,  
    #                        filled=True, rounded=True,
    #                        special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    #        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    #        graph.write_png(i_sta+'_bs_snowfall'+str(i_count)+'.png')
    #        Image(graph.create_png())
        	   



    #print(ss)

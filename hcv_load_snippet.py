from ucimlrepo import fetch_ucirepo 
  
# fetch dataset (HCV data)
hcv_data = fetch_ucirepo(id=571) 
  
# data (as pandas dataframes) 
X = hcv_data.data.features 
y = hcv_data.data.targets 
  
# metadata 
print(hcv_data.metadata) 
  
# variable information 
print(hcv_data.variables) 

from data import *

for i in range(10):
    count= 0
    for j in range(16):
        if(cipher_text[j]!= fault_ct[i][j]):
            count+= 1
        
    print(i,count)
    

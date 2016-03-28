
# coding: utf-8

# In[76]:

import csv
data_lists = {}
i = 0
with open('ExperimentRunner/src/experiment1_1.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data_lists[i] = list(row)
        i=i+1

# print data_lists
timesteps = {}
events = {}
compartments = [{}]*((len(data_lists))/50-2)

s = {}
e = {}
i = {}
d = {}
r = {}
dr = {}
k = -1
c=0
for j in data_lists:
    if j%8==0:
        timesteps[k] = [int(data_lists[j][m]) for m in range(len(data_lists[j])) if m!=0 and data_lists[j][m]!='']
        k=k+1
    else:
        if j%8==1:
            events[k] = [int(data_lists[j][m]) for m in range(len(data_lists[j])) if m!=0 and data_lists[j][m]!='']
        else:
            compartments[c%len(compartments)][k] = list([int(data_lists[j][m]) for m in range(len(data_lists[j])) if m!=0 and data_lists[j][m]!=''])
            print c%len(compartments), k, compartments[c%len(compartments)]
            c=c+1
            
print compartments[0]
        
max_time = 0
for j in timesteps:
    local_max = max(timesteps[j])
    if local_max>max_time:
        max_time = local_max

for j in timesteps:
    index = 0
    local_s = [0]*(max_time+1)
    local_e = [0]*(max_time+1)
    local_i = [0]*(max_time+1)
    local_d = [0]*(max_time+1)
    local_r = [0]*(max_time+1)
    local_dr = [0]*(max_time+1)
    last_value_s = s[j][0]
    last_value_e = e[j][0]
    last_value_i = i[j][0]
    last_value_d = d[j][0]
    last_value_r = r[j][0]
    last_value_dr = dr[j][0]
    for m in range(0, max_time+1):
        if index<len(timesteps[j]) and timesteps[j][index]==m:
            local_s[m] = s[j][index]
            last_value_s = s[j][index]
            local_e[m] = e[j][index]
            last_value_e = e[j][index]
            local_i[m] = i[j][index]
            last_value_i = i[j][index]
            local_d[m] = d[j][index]
            last_value_d = d[j][index]
            local_r[m] = r[j][index]
            last_value_r = r[j][index]
            local_dr[m] = dr[j][index]
            last_value_dr = dr[j][index]
            index=index+1
        else:
            local_s[m] = last_value_s
            local_e[m] = last_value_e
            local_i[m] = last_value_i
            local_d[m] = last_value_d
            local_r[m] = last_value_r
            local_dr[m] = last_value_dr
    s[j] = local_s
    e[j] = local_e
    i[j] = local_i
    d[j] = local_d
    r[j] = local_r
    dr[j] = local_dr
    


# In[47]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
for j in s:
    plt.plot(range(0,max_time+1), s[j], 'y-')
    plt.plot(range(0,max_time+1), e[j], 'b-')
    plt.plot(range(0,max_time+1), i[j], 'r-')
    plt.plot(range(0,max_time+1), d[j], 'k-')
    plt.plot(range(0,max_time+1), [x + y for x, y in zip(r[j], dr[j])], 'g-')


# In[50]:

s_average = [0]*(max_time+1)
num_of_experiments = len(s)
for i in range(0, max_time+1):
    local_sum_s = 0
    for j in s:
        local_sum_s=local_sum_s+s[j][i]
    s_average[i]=(local_sum_s+0.0)/num_of_experiments
print s_average


# In[51]:

plt.plot(range(0,max_time+1), s_average, 'y-')


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[46]:


from matplotlib import pyplot as plt
import numpy as np
import math
from matplotlib.patches import Polygon
from scipy.special import gamma
import scipy.stats as stats
import csv
import random
import pandas as pd
import pylab as py
import statsmodels.api as sm


# ## Helper Functions

# In[47]:


def open_file():
    with open('/Users/vaibhavseshadri/Desktop/untitled folder/archive/application_record.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data
file = pd.read_csv('/Users/vaibhavseshadri/Desktop/untitled folder/archive/application_record.csv')

incomes_total = file['AMT_INCOME_TOTAL'].tolist()

pop_mean = np.mean(incomes_total)
pop_std = np.std(incomes_total)

data = open_file()


def rand_lst(data, size):
    return random.sample(range(1, len(data)), size)

def create_rand_sample(data, n):
    sample = []
    rand_lst = set()
    count = 0
    prev_size = 0
    while count < n:
        
        x = random.randint(1, len(data) - 1)
        rand_lst.add(x)
        curr_size = len(rand_lst)
        
        if curr_size > prev_size:
            count += 1
            sample.append(float(data[x][5]))
        prev_size = curr_size
    
    return sample

def create_rand_sample_male(data):
    sample_male = []
    rand_5000 = set()
    count = 0
    prev_size = 0
    while count < 5000:
        
        x = random.randint(1, len(data) - 1)
        if data[x][1] == 'M':
            rand_5000.add(x)
            curr_size = len(rand_5000)
        
            if curr_size > prev_size:
                count += 1
                sample_male.append(float(data[x][5]))
            prev_size = curr_size
    
    return sample_male

def create_rand_sample_female(data):
    sample_female = []
    rand_6000 = set()
    count = 0
    prev_size = 0
    while count < 6000:
        
        x = random.randint(1, len(data) - 1)
        if data[x][1] == 'F':
            rand_6000.add(x)
            curr_size = len(rand_6000)
        
            if curr_size > prev_size:
                count += 1
                sample_female.append(float(data[x][5]))
            prev_size = curr_size
    
    return sample_female

def CI_mean(sample, CI):
    return stats.t.interval(alpha=CI, df=len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))

def CI_var(sample, CI, n):
    alpha = 1 - CI                              
    var = np.var(sample, ddof=1)                

    upper = (n-1) * (var / stats.chi2.ppf(alpha / 2, n-1))
    lower = (n-1) * (var / stats.chi2.ppf(1 - alpha / 2, n-1))
    
    return (lower, upper)

def CI_diff_mean(sample1, sample2, n1, n2, CI):
    alpha = 1 - CI
    
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    num_sp2 = ((n1-1)*var1) + ((n2-1)*var2)
    den_sp2 = n1 + n2 - 2
    sp2 = math.sqrt(num_sp2/den_sp2)
    
    t_score = -1*(stats.t.ppf(alpha/2, den_sp2))
    
    upper = (mean1 - mean2) + (t_score*sp2*(math.sqrt((1/n1) + (1/n2))))
    lower = (mean1 - mean2) - (t_score*sp2*(math.sqrt((1/n1) + (1/n2))))
    
    return (lower, upper)

def CI_proportion(p, n, CI):
    alpha = 1 - CI
    q = 1 - p
    t_score = -1*(stats.t.ppf(alpha/2, n-1))
    
    upper = p + (t_score*(math.sqrt(p*q/n)))
    lower = p - (t_score*(math.sqrt(p*q/n)))
    
    return (lower, upper)


def plot_n_CI(n, m, data, CI):
    
    plt.xlabel('Number of samples', fontsize = 15)
    plt.ylabel('Income', fontsize = 15)
    txt = 'Mean and 95% confidence interval for 1000 samples of size ' + str(m)
    plt.title(txt, fontsize = 15)
    
    for i in range(n):
        
        sample = create_rand_sample(data, m)
        interval = CI_mean(sample, CI)
        
        if interval[0] < 187524.2860095039 and interval[1] > 187524.2860095039:
           
            plt.plot([i+1, i+1], [interval[0], interval[1]], marker = 'o', markersize = 2, color = 'g')
            plt.plot([i+1], [np.mean(sample)], marker = 'o', markersize = 5, color = 'b')
        
        else:
            plt.plot([i+1, i+1], [interval[0], interval[1]], marker = 'o', markersize = 2, color = 'r')
            plt.plot([i+1], [np.mean(sample)], marker = 'o', markersize = 5, color = 'b')

def get_n_means(n, size, data):
    mean_list = []
    for i in range(n):
        sample = create_rand_sample(data, size)
        sample_mean = np.mean(sample)
        mean_list.append(sample_mean)
    return mean_list

def get_z_score(mean_list, mean, std):
    z_score = []
    for i in mean_list:
        z_score.append((i-mean)/std)
    return z_score

def plot_hist(n, size, data):
    mean_list = get_n_means(n, size, data)
    mean_of_sample_means = np.mean(mean_list)
    sd_of_sample_means = np.std(mean_list)
    z_score = get_z_score(mean_list, mean_of_sample_means, sd_of_sample_means)

    plt.ylabel('Proportion of Population', fontsize = 15)
    plt.xlabel('Z-Score of sample means', fontsize = 15)
    txt = 'Histogram of sample means of ' + str(n) + ' samples of size ' + str(size)
    plt.title(txt, fontsize = 12)
    plt.hist(z_score, density = True, range = (-3,3))
    
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), label = 'Standard Normal Curve from random normally distributed data')
    plt.legend()

    


# # ACTIVITY 1

# ## 1A

# In[48]:


lst = create_rand_sample(data, 1000)

CI_mean_90 = CI_mean(lst, 0.90)
CI_mean_95 = CI_mean(lst, 0.95)
CI_mean_98 = CI_mean(lst, 0.98)

CI_var_90 = CI_var(lst, 0.90, 1000)
CI_var_95 = CI_var(lst, 0.95, 1000)
CI_var_98 = CI_var(lst, 0.98, 1000)

print("90% Confidence interval for mean of random sample of size 1000: ", CI_mean_90)
print("95% Confidence interval for mean of random sample of size 1000: ", CI_mean_95)
print("98% Confidence interval for mean ofrandom sample of size 1000: ", CI_mean_98)
print("\n90% Confidence interval for variance of random sample of size 1000: ", CI_var_90)
print("95% Confidence interval for variance of random sample of size 1000: ", CI_var_95)
print("98% Confidence interval for variance of random sample of size 1000: ", CI_var_98)


# ## 1B

# In[49]:


random_bootstrap_sample = create_rand_sample(data, 1000)
bootstrap_means = []

for i in range(1000):
    bootstrapped_sample = random.choices(random_bootstrap_sample, k =1000)
    bootstrap_means.append(np.mean(bootstrapped_sample))

bootstrap_means.sort()
interval = (bootstrap_means[50], bootstrap_means[-51])
print("The 90% Confidence interval for the mean using bootstrapping is: ", interval)
print("90% Confidence interval for mean of random sample of size 1000: ", CI_mean(random_bootstrap_sample, 0.90))


# ## 1C

# In[50]:


plot_n_CI(1000, 15, data, 0.95)
x1, y1 = [0,1000], [187524.2860095039 , 187524.2860095039]
plt.plot(x1, y1, color = 'y', label = 'Population mean')
plt.legend()
plt.show()

plot_n_CI(1000, 50, data, 0.95)
x1, y1 = [0,1000], [187524.2860095039 , 187524.2860095039]
plt.plot(x1, y1, color = 'y', label = 'Population mean')
plt.legend()
plt.show()

plot_n_CI(1000, 200, data, 0.95)
x1, y1 = [0,1000], [187524.2860095039 , 187524.2860095039]
plt.plot(x1, y1, color = 'y', label = 'Population mean')
plt.legend()
plt.show()

plot_n_CI(1000, 1000, data, 0.95)
x1, y1 = [0,1000], [187524.2860095039 , 187524.2860095039]
plt.plot(x1, y1, color = 'y', label = 'Population mean')
plt.legend()
plt.show()

plot_n_CI(1000, 10000, data, 0.95)
x1, y1 = [0,1000], [187524.2860095039 , 187524.2860095039]
plt.plot(x1, y1, color = 'y', label = 'Population mean')
plt.legend()
plt.show()


# Based on the graphs it is apparent that as we increase the sample size, the width of the 95% confidence interval decreases. In other words, for the same level of confidence, larger sample sizes have a more precise prediciton of the true mean. (Note: y-axis is scaled by 1E6 on first graph)

# ## 1D

# In[51]:


male_income = create_rand_sample_male(data)
female_income = create_rand_sample_female(data)

interval_diff_mean_90 = CI_diff_mean(male_income, female_income, len(male_income), len(female_income), 0.90)
interval_diff_mean_95 = CI_diff_mean(male_income, female_income, len(male_income), len(female_income), 0.95)
interval_diff_mean_98 = CI_diff_mean(male_income, female_income, len(male_income), len(female_income), 0.98)

print("90% CI for difference in means between males and females income: ",interval_diff_mean_90)
print("95% CI for difference in means between males and females income: ",interval_diff_mean_95)
print("98% CI for difference in means between males and females income: ",interval_diff_mean_98)


# ## 1E

# In[52]:


z = 1.96

interval_size = 4000

n = ((2*z*pop_std)/(interval_size))**2

print("The sample size required for a confidence interval of 95% with an interval width of 4000 is: ", n)

test_sample = create_rand_sample(data, n)

CI_test_sample = CI_mean(test_sample, 0.95)

test_interval_size = CI_test_sample[1] - CI_test_sample[0]

print("Drawing a random sample of size n creates a confidence interval of width: ", test_interval_size)


# ## 1F (i)

# In[53]:


male_female_sample = rand_lst(data, 500)

num_male = 0
num_female = 0

for i in male_female_sample:
    if data[i][1] == 'M':
        num_male += 1
    else:
        num_female += 1
p = num_female / 500

print("The proportion of female participants in this random sample is: ", p)
print("The proportion of male participants in this random sample is: ", 1-p)

CI_proportion_90 = CI_proportion(p, 500, 0.90)

print("\nThe 90% confidence interval for the proportion of female participants is: ", CI_proportion_90)
print("")


# The findings in part 1F (i) show how based on a random sample size of 500, the 90% confidence interval depicts the range, with 90% confidence, that the true proportion of female participants lies inbetween.

# ## 1F (ii)

# In[54]:


print("The 99% confidence interval of the true proportion of male participants, given sample size 3000 and sample male proportion of 68%: ", CI_proportion(0.68, 3000, 0.99))


# ## 2: n = 15

# In[55]:


plot_hist(50, 15, data)
plt.show()

plot_hist(100, 15, data)
plt.show()

plot_hist(1000, 15, data)
plt.show()


# ## 2: n = 50

# In[56]:


plot_hist(50, 50, data)
plt.show()

plot_hist(100, 50, data)
plt.show()

plot_hist(1000, 50, data)
plt.show()


# ## 2: n = 200

# In[57]:


plot_hist(50, 200, data)
plt.show()

plot_hist(100, 200, data)
plt.show()

plot_hist(1000, 200, data)
plt.show()


# ## 2: n = 1000

# In[58]:


plot_hist(50, 1000, data)
plt.show()

plot_hist(100, 1000, data)
plt.show()

plot_hist(1000, 1000, data)
plt.show()


# ## 2: n = 10000

# In[59]:


plot_hist(50, 10000, data)
plt.show()

plot_hist(100, 10000, data)
plt.show()

plot_hist(1000, 10000, data)
plt.show()


# ## 2: n = 100000

# In[60]:


plot_hist(50, 10000, data)
plt.show()

plot_hist(100, 10000, data)
plt.show()

plot_hist(1000, 10000, data)
plt.show()


# The above graphs help visualize the CLT by increasing the number of samples and plotting the z-score of the means on x-axis and the proportion of the total population that has those z-scores on the y-axis. The CLT states that although one sample is not necessarily randomly distributed. If we take multiple samples, the means of these samples will be normally distributed. The findings show that the sample size does not effect the normaility of the data as much as the number of samples do. Take n1 = 50 with 1000 samples vs n1 = 10000 with 50 samples. The 1000 samples of smaller sample size fits the normal curve better than the 50 samples with larger sample size. 

# ## 3

# In[61]:


incomes_total.sort()
percentile = []
plt.grid()

for i in range (len(incomes_total)):
    p = (i+1) / (len(incomes_total) + 1)
    z_score = stats.norm.ppf(p)
    percentile.append(z_score - p)

table = pd.DataFrame(incomes_total, percentile)
print(table)

plt.xlabel('Actual Incomes', fontsize = 15)
plt.ylabel('Theoretical Values', fontsize = 15)
plt.title('Q-Q plot of Annual Incomes', fontsize = 12)
plt.scatter(incomes_total, percentile)
plt.show()


# The findings show that when we plot the actual income vs the theoretical values, the output is not linear. Thus, the population is not normally distributed. 

# ## 4a

# In[62]:


male_list = []
female_list = []

for i in range (1, len(data)):
    if data[i][1] == 'M':
        male_list.append(float(data[i][5]))
    else:
        female_list.append(float(data[i][5]))


var1 = np.var(male_list)
var2 = np.var(female_list)

mean1 = np.mean(male_list)
mean2 = np.mean(female_list)

print(mean1)
print(mean2)

size1 = len(male_list)
size2 = len(female_list)

T = (mean1 - mean2)/(math.sqrt((var1/size1) + (var2/size2)))
print(T)
dof = (((var1/size1) + (var2/size2))**2) / ((((var1/size1)**2)/(size1-1)) + (((var2/size2)**2)/(size2-1)))
print(dof)
t_score = (stats.t.ppf(0.95, dof))

if abs(T) > t_score:
    print("Based on the t-test, there is a difference in the incomes of male and female")
else:
    print("Based on the t-test, there is no difference in the incomes of male and females")


# Mean income of male participants: 214086.63886841942
# Mean income of female participants: 174523.04091044015
# 
# Test Statistic: 102.19552813426439
# 
# Since the Test Statistic < t-score (with alpha = 0.1 and 222033.1445060923 degrees of freedom)
# 
# There is a difference in the incomes of males and females
# 

# ## 4b

# In[63]:


less_than_2_children = []
more_than_2_children = []

for i in range (1, len(data)):
    
    if int(data[i][4]) < 2 and data[i][8]:
        less_than_2_children.append(float(data[i][5]))
    else:
        more_than_2_children.append(float(data[i][5]))


var1 = np.var(less_than_2_children)
var2 = np.var(more_than_2_children)

mean1 = np.mean(less_than_2_children)
mean2 = np.mean(more_than_2_children)
print(mean1)
print(mean2)

size1 = len(less_than_2_children)
size2 = len(more_than_2_children)

T = (mean1 - mean2)/(math.sqrt((var1/size1) + (var2/size2)))
print(T)
dof = (((var1/size1) + (var2/size2))**2) / ((((var1/size1)**2)/(size1-1)) + (((var2/size2)**2)/(size2-1)))
print(dof)
t_score = (stats.t.ppf(0.9, dof))
print(t_score)
if T > t_score:
    print("Based on the t-test, Families with less than 2 children make more than families with 2 or more children")
else:
    print("Based on the t-test, Families with less than 2 children make less than families with 2 or more children")


# Assumption: Only married people are taken into account for "families"
# 
# Mean income of families with less than 2 children: 187310.91068337078
# Mean income of families with two or more children: 189347.01335973368
# 
# Test Statistic: -3.8678510964456136
# 
# Since the Test Statistic < t-score (with alpha = 0.1 and 58207.902619469925 degrees of freedom)
# 
# Families with less than 2 children make less than families with 2 or more children
# 

# # ACTIVITY 2

# In[64]:


datapoints = [[655.5, 788.3, 734.3, 721.4, 679.1, 699.4],
[789.2, 772.5, 786.9, 686.1, 732.1, 774.8], 
[737.1, 639.0, 696.3, 671.7, 717.2, 727.1],
[535.1, 628.7, 542.4, 559.0, 586.9, 520.0]]

figure = plt.figure()
axis = figure.add_axes([0, 0, 1, 1])

plot = axis.boxplot(datapoints)

plt.xlabel('Box number', fontsize = '15')
plt.ylabel('Compression Strength (lb.)', fontsize = '15')
plt.title('Box plots for compression strength', fontsize = 20)

plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport


# In[5]:


df = pd.read_csv('adult.data.csv')


# In[7]:


df.keys()


# In[8]:


df.head(20)


# In[9]:


df.tail()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[13]:


df.describe(include=['O'])


# In[14]:


prof = ProfileReport(df)
prof.to_file('adult.data.html')


#  ### How many of each race are represented in this dataset? This should be a Pandas series with race names as the index labels.

# In[20]:


race_type = df.race.unique()
race_type


# In[19]:


race_count = df.race.value_counts()
race_count


#  ### What is the average age of men?

# In[33]:


male_data= df['sex']=='Male'
df_male=df[male_data]
average_age_men=df_male.age.mean()
average_age_men


#  ### What is the percentage of people who have a Bachelor's degree?

# In[64]:


percentage_education=df['education'].value_counts(normalize=True)
percentage_bachelors=percentage_education['Bachelors']*100
#percentage_bachelors = round(df[df['education'] == 'Bachelors'].shape[0] / df.shape[0] * 100, 1)
percentage_bachelors


#  ### What percentage of people with advanced education (`Bachelors`, `Masters`, or `Doctorate`) make more than 50K?

# In[65]:


higher_education = df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])
rich_people = df['salary'] == '>50K'


# In[66]:


higher_education_rich = round((higher_education & rich_people).sum() / higher_education.sum() * 100, 1)
higher_education_rich


# In[67]:


lower_education_rich = round((~higher_education & rich_people).sum() / (~higher_education).sum() * 100, 1)
lower_education_rich


# In[73]:


people_50k=df['salary']=='>50K'
df_people_50k=df[people_50k]
percentage_education_50k=df_people_50k['education'].value_counts(normalize=True)
higher_education_rich_2=(percentage_education_50k['Bachelors']+percentage_education_50k['Masters']+percentage_education_50k['Doctorate'])*100
higher_education_rich_2


# ### What percentage of people without advanced education make more than 50K?

# In[76]:


percentage_education_50k
lower_education_rich_2=100-higher_education_rich_2
lower_education_rich_2


# ### What is the minimum number of hours a person works per week (hours-per-week feature)?

# In[77]:


min_work_hours = df['hours-per-week'].min()
min_work_hours


# ### What percentage of the people who work the minimum number of hours per week have a salary of >50K?

# In[86]:


people_50k=df['salary']=='>50K'
df_people_50k=df[people_50k]
people_50k_1h=df_people_50k['hours-per-week']==1
df_people_50k_1h=df_people_50k[people_50k_1h]
df_people_50k_1h


# In[97]:


num_min_workers = df_people_50k_1h['salary'].count()
num_min_workers


# In[96]:


rich_percentage = df_people_50k_1h['salary'].count()*100/df['salary'].count()
rich_percentage


#  ### What country has the highest percentage of people that earn >50K?

# In[114]:


country_rich_percentage = (df[df["salary"] == ">50K"]["native-country"].value_counts() / df["native-country"].value_counts())*100
highest_earning_country = country_rich_percentage.sort_values(ascending=False).index[0]
highest_earning_country


# In[119]:


highest_earning_country_percentage = country_rich_percentage.sort_values(ascending=False).iloc[0]
highest_earning_country


# ### Identify the most popular occupation for those who earn >50K in India.

# In[130]:


df_indians = df[df["native-country"] == "India"]
df_rich_indians = df_indians[df_indians["salary"] == ">50K"]
top_IN_occupation = df_rich_indians['occupation'].value_counts().index[0]
top_IN_occupation


# In[ ]:


# Identify the most popular occupation for those who earn >50K in India.
top_IN_occupation = None

    # DO NOT MODIFY BELOW THIS LINE

    if print_data:
        print("Number of each race:\n", race_count) 
        print("Average age of men:", average_age_men)
        print(f"Percentage with Bachelors degrees: {percentage_bachelors}%")
        print(f"Percentage with higher education that earn >50K: {higher_education_rich}%")
        print(f"Percentage without higher education that earn >50K: {lower_education_rich}%")
        print(f"Min work time: {min_work_hours} hours/week")
        print(f"Percentage of rich among those who work fewest hours: {rich_percentage}%")
        print("Country with highest percentage of rich:", highest_earning_country)
        print(f"Highest percentage of rich people in country: {highest_earning_country_percentage}%")
        print("Top occupations in India:", top_IN_occupation)


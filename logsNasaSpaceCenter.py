#!/usr/bin/env python
# coding: utf-8

# <pre>
# <b>HTTP requests to the NASA Kennedy Space Center WWW server</b>
# 
# <b>Fonte oficial do dateset:</b>  http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
# Dados:
#     
#  <a href = 'ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz'>
#     ● Jul 01 to Jul 31, ASCII format, 20.7 MB gzip compressed, 205.2 MB</a>
#  <a href = 'ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz'>
#     ● Aug 04 to Aug 31, ASCII format, 21.8 MB gzip compressed, 167.8 MB.</a>
# 
# Sobre o dataset​: Esses dois conjuntos de dados possuem todas as requisições HTTP para o servidor da NASA Kennedy
# Space Center WWW na Flórida para um período específico.
# 
# Os logs estão em arquivos ASCII com uma linha por requisição com as seguintes colunas:
# <b>● Host fazendo a requisição.</b> Um hostname quando possível, caso contrário o endereço de internet se o nome
# não puder ser identificado.
# <b>● Timestamp</b> no formato "DIA/MÊS/ANO:HH:MM:SS TIMEZONE"
# <b>● Requisição(entre aspas)</b>
# <b>● Código do retorno HTTP</b>
# <b>● Total de bytes retornados</b>
# 
# Questões
# Responda as seguintes questões devem ser desenvolvidas em Spark utilizando a sua linguagem de preferência.
# 
# <b>1. Número de hosts únicos.</b>
# <b>2. O total de erros 404.</b>
# <b>3. Os 5 URLs que mais causaram erro 404.</b>
# <b>4. Quantidade de erros 404 por dia.</b>
# <b>5. O total de bytes retornados.</b>
# 
# </pre>
# 
# 

# <pre>
# Fontes de pesquisa:
#     
#     <a href='https://opensource.com/article/19/5/log-data-apache-spark'>How to wrangle log data with Python and Apache Spark</a>
#     <a href='https://opensource.com/article/19/5/visualize-log-data-apache-spark'>How to analyze log data with Python and Apache Spark</a>
#     
#     
# </pre>
#         

# In[1]:


from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import regexp_extract
from pyspark.sql.functions import udf
from pyspark.sql import functions as F

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import pandas as pd
import glob
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sc = SparkContext()
sqlContext = SQLContext(sc)
spark = SparkSession(sc)


# In[3]:


raw_data_files = glob.glob('data/*')
raw_data_files


# In[4]:


base_df = spark.read.text(raw_data_files)
base_df.printSchema()


# In[5]:


type(base_df)


# In[6]:


base_df_rdd =  base_df.rdd


# In[7]:


type(base_df_rdd)


# In[8]:


base_df.show(10, truncate=False)


# In[9]:


base_df_rdd.take(10)


# In[10]:


print((base_df.count(), len(base_df.columns)))


# In[11]:


sample_logs = [item['value'] for item in base_df.take(15)]
sample_logs


# <pre><b>Extraindo os Hostnames</b></pre>

# In[12]:


host_pattern = r'(^\S+\.[\S+\.]+\S+)\s'
hosts = [re.search(host_pattern, item).group(1)
           if re.search(host_pattern, item)
           else 'no match'
           for item in sample_logs]


# In[13]:


hosts


# <pre><b>Extraindo os horários</b></pre>

# In[14]:


ts_pattern = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'
timestamps = [re.search(ts_pattern, item).group(1) for item in sample_logs]
timestamps


# <pre><b>Extraindo os requisições HTTP, URL's e protocolos</b></pre>

# In[15]:


method_uri_protocol_pattern = r'\"(\S+)\s(\S+)\s*(\S*)\"'
method_uri_protocol = [re.search(method_uri_protocol_pattern, item).groups()
               if re.search(method_uri_protocol_pattern, item)
               else 'no match'
              for item in sample_logs]
method_uri_protocol


# <pre><b>Extraindo os códigos HTTP's</b></pre>

# In[16]:


status_pattern = r'\s(\d{3})\s'
status = [re.search(status_pattern, item).group(1) for item in sample_logs]
print(status)


# <pre><b>Extraindo as respostas HTTP's</b></pre>

# In[17]:


content_size_pattern = r'\s(\d+)$'
content_size = [re.search(content_size_pattern, item).group(1) for item in sample_logs]
print(content_size)


# <pre><b>Juntando todos os dados</b></pre>

# In[18]:


logs_df = base_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                         regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                         regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                         regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                         regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                         regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                         regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'))
logs_df.show(10, truncate=True)
print((logs_df.count(), len(logs_df.columns)))


# In[19]:


nullValue = (base_df.filter(base_df['value'].isNull()).count())
nullValue


# In[20]:


bad_rows_df = logs_df.filter(logs_df['host'].isNull()|
                             logs_df['timestamp'].isNull() |
                             logs_df['method'].isNull() |
                             logs_df['endpoint'].isNull() |
                             logs_df['status'].isNull() |
                             logs_df['content_size'].isNull()|
                             logs_df['protocol'].isNull())
bad_rows_df.count()


# <pre><b>Contando valores nulos</b></pre>

# In[21]:



def count_null(col_name):
    return spark_sum(col(col_name).isNull().cast('integer')).alias(col_name)

# Build up a list of column expressions, one per column.
exprs = [count_null(col_name) for col_name in logs_df.columns]

# Run the aggregation. The *exprs converts the list of expressions into
# variable function arguments.
logs_df.agg(*exprs).show()


# <pre><b>Manipulando status HTTP's nulos </b></pre>

# <pre><b>a=</b></pre>

# In[22]:


regexp_extract('value', r'\s(\d{3})\s', 1).cast('integer').alias( 'status')
null_status_df = base_df.filter(~base_df['value'].rlike(r'\s(\d{3})\s'))
null_status_df.count()


# In[23]:


null_status_df.show(truncate=False)


# In[24]:


bad_status_df = null_status_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                                      regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                                      regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                                      regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                                      regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                                      regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                                      regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'))
bad_status_df.show(truncate=False)


# In[25]:


logs_df = logs_df[logs_df['status'].isNotNull()]
exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()


# In[26]:


regexp_extract('value', r'\s(\d+)$', 1).cast('integer').alias('content_size')
null_content_size_df = base_df.filter(~base_df['value'].rlike(r'\s\d+$'))
null_content_size_df.count()


# In[27]:


null_content_size_df.take(10)


# In[28]:


logs_df = logs_df.na.fill({'content_size': 0})
exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()


# In[29]:



month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(text):

    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(text[7:11]),
      month_map[text[3:6]],
      int(text[0:2]),
      int(text[12:14]),
      int(text[15:17]),
      int(text[18:20])
    )
udf_parse_time = udf(parse_clf_time)

logs_df = (logs_df.select('*', udf_parse_time(logs_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp'))


# In[30]:


logs_df.show(10, truncate=True)


# In[31]:


logs_df.printSchema()


# In[32]:


logs_df.cache()


# In[33]:


logs_df.show()


# In[34]:


content_size_summary_df = logs_df.describe(['content_size'])
content_size_summary_df.toPandas()


# In[35]:


(logs_df.agg(F.min(logs_df['content_size']).alias('min_content_size'),
             F.max(logs_df['content_size']).alias('max_content_size'),
             F.mean(logs_df['content_size']).alias('mean_content_size'),
             F.stddev(logs_df['content_size']).alias('std_content_size'),
             F.count(logs_df['content_size']).alias('count_content_size'))
        .toPandas())


# In[36]:


status_freq_df = (logs_df
                     .groupBy('status')
                     .count()
                     .sort('status')
                     .cache())
print('Códigos HTTP:', status_freq_df.count())  


# In[37]:


status_freq_pd_df = (status_freq_df
                         .toPandas()
                         .sort_values(by=['count'],
                                      ascending=False))
status_freq_pd_df


# In[38]:


sns.catplot(x='status', y='count', data=status_freq_pd_df,
            kind='bar', order=status_freq_pd_df['status'])


# In[39]:


log_freq_df = status_freq_df.withColumn('log(count)',
                                        F.log(status_freq_df['count']))
log_freq_df.show()


# In[40]:


log_freq_pd_df = (log_freq_df
                    .toPandas()
                    .sort_values(by=['log(count)'],
                                 ascending=False))
sns.catplot(x='status', y='log(count)', data=log_freq_pd_df,
            kind='bar', order=status_freq_pd_df['status'])


# In[41]:


host_sum_df =(logs_df
               .groupBy('host')
               .count()
               .sort('count', ascending=False).limit(10))

host_sum_df.show(truncate=False)


# In[42]:


host_sum_pd_df = host_sum_df.toPandas()
host_sum_pd_df.iloc[8]['host']


# In[43]:


paths_df = (logs_df
            .groupBy('endpoint')
            .count()
            .sort('count', ascending=False).limit(20))

paths_pd_df = paths_df.toPandas()
paths_pd_df  


# In[44]:


not200_df = (logs_df
               .filter(logs_df['status'] != 200))

error_endpoints_freq_df = (not200_df
                               .groupBy('endpoint')
                               .count()
                               .sort('count', ascending=False)
                               .limit(10)
                          )
                         
error_endpoints_freq_df.show(truncate=False)  


# In[45]:


unique_host_count = (logs_df
                     .select('host')
                     .distinct()
                     .count())
unique_host_count


# In[46]:


logs_df.show()


# In[47]:


logs_df.host


# In[48]:


host_day_df = logs_df.select(logs_df.host,F.dayofmonth('time').alias('day'))
host_day_df.show(5, truncate=False)


# In[49]:


host_day_df = logs_df.select(logs_df.host,
                             F.dayofmonth('time').alias('day'))
host_day_df.show(5, truncate=False)


# In[82]:


def_mr = pd.get_option('max_rows')
pd.set_option('max_rows', 10)

daily_hosts_df = (host_day_df.groupBy('day').count().sort("day"))
daily_hosts_df = daily_hosts_df.toPandas()


# In[83]:


daily_hosts_df


# In[84]:


type(daily_hosts_df)


# In[85]:


c = sns.catplot(x='day', y='count',
                data=daily_hosts_df,
                kind='point', height=5,
                aspect=1.5)


# In[88]:


daily_hosts_df = (host_day_df
                     .groupBy('day')
                     .count()
                     .select(col("day"),
                                      col("count").alias("total_hosts")))

total_daily_reqests_df = (logs_df
                              .select(F.dayofmonth("time")
                                          .alias("day"))
                              .groupBy("day")
                              .count()
                              .select(col("day"),
                                      col("count").alias("total_reqs")))

avg_daily_reqests_per_host_df = total_daily_reqests_df.join(daily_hosts_df, 'day')
avg_daily_reqests_per_host_df = (avg_daily_reqests_per_host_df
                                    .withColumn('avg_reqs', col('total_reqs') / col('total_hosts'))
                                    .sort("day"))
avg_daily_reqests_per_host_df = avg_daily_reqests_per_host_df.toPandas()


# In[92]:


avg_daily_reqests_per_host_df


# In[87]:


c = sns.catplot(x='day', y='avg_reqs',
                data=avg_daily_reqests_per_host_df,
                kind='point', height=5, aspect=1.5)


# In[68]:


not_found_df = logs_df.filter(logs_df["status"] == 404).cache()
print(('Total 404 responses: {}').format(not_found_df.count()))


# In[69]:


endpoints_404_count_df = (not_found_df
                          .groupBy("endpoint")
                          .count()
                          .sort("count", ascending=False)
                          .limit(20))

endpoints_404_count_df.show(truncate=False)


# In[70]:


hosts_404_count_df = (not_found_df
                          .groupBy("host")
                          .count()
                          .sort("count", ascending=False)
                          .limit(20))

hosts_404_count_df.show(truncate=False)


# <pre><b>Erros 404 por dia </b></pre>

# In[71]:


errors_by_date_sorted_df = (not_found_df
                                .groupBy(F.dayofmonth('time').alias('day'))
                                .count()
                                .sort("day"))

errors_by_date_sorted_pd_df = errors_by_date_sorted_df.toPandas()
errors_by_date_sorted_pd_df


# In[72]:


c = sns.catplot(x='day', y='count',
                data=errors_by_date_sorted_pd_df,
                kind='point', height=5, aspect=1.5)


# In[73]:


(errors_by_date_sorted_df
    .sort("count", ascending=False)
    .show(3))


# <pre><b>Visualizando os erros 404 por hora</b></pre>

# In[74]:


hourly_avg_errors_sorted_df = (not_found_df
                                   .groupBy(F.hour('time')
                                             .alias('hour'))
                                   .count()
                                   .sort('hour'))
hourly_avg_errors_sorted_pd_df = hourly_avg_errors_sorted_df.toPandas()

c = sns.catplot(x='hour', y='count',
                data=hourly_avg_errors_sorted_pd_df,
                kind='bar', height=5, aspect=1.5)


# In[ ]:





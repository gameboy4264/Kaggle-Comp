#子数据集生成

df_country = df.drop('country', axis=1).join(df.country.str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('country'))

df_genr = df.drop('genr', axis=1).join(df.genr.str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('genr'))

for x in range(250):
    if len(df.year[x]) > 4:
        print(x, df.year[x])
df4.year[66] = '1961'
df4.year[205] = '1983'


#时间分析

import matplotlib.pyplot as plt
ax = plt.subplot(111)
ax.set_title('Top250时间数量折线', fontsize=17)
df4.year.value_counts().sort_index().plot(figsize=(14,7))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.set_xlabel('时间', fontsize=17)
ax.set_ylabel('数量', fontsize=17)

fig = plt.figure
ax = plt.subplot(111)
ax.set_title('Top250时间均分折线', fontsize=17)
df4['rating'].groupby(df4['year']).mean().plot(figsize=(14,7))
plt.xticks(fontsize=15, rotation=30)
plt.yticks(fontsize=15)
ax.set_xlabel('时间', fontsize=17)
ax.set_ylabel('平均分', fontsize=17)

#种类分析

fig = plt.figure
ax = plt.subplot(111)
ax.set_title('Top250种类均分排序', fontsize=17)
df_genr['rating'].groupby(df_genr['genr']).mean().sort_values(ascending=False).plot.bar(figsize=(14,7))
plt.xticks(fontsize=15, rotation=60)
plt.yticks(fontsize=15)
ax.set_ylim([8.5, 9.5])
ax.set_xlabel('种类', fontsize=17)
ax.set_ylabel('平均分', fontsize=17)

fig = plt.figure
ax = plt.subplot(111)
ax.set_title('Top250种类数量排序', fontsize=17)
df_genr.genr.value_counts()[1:].plot.bar(figsize=(14,7))
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
ax.set_xlabel('种类', fontsize=17)
ax.set_ylabel('数量', fontsize=17)

#国家分析

fig = plt.figure
ax = plt.subplot(111)
ax.set_title('Top250国家拍摄数量排序', fontsize=17)
df_country2.country.value_counts().head(11).plot.bar(figsize=(14,7))
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
ax.set_xlabel('国家', fontsize=17)
ax.set_ylabel('数量', fontsize=17)

fig = plt.figure
ax = plt.subplot(111)
ax.set_title('Top250拍摄国家均分', fontsize=17)
df_country['rating'].groupby(df_country['country']).mean().sort_values(ascending=False).head(12).plot.bar(figsize=(14,7))
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
ax.set_ylim([8.8, 9.3])
ax.set_xlabel('国家', fontsize=17)
ax.set_ylabel('平均分', fontsize=17)

#导演分析

fig = plt.figure
ax = plt.subplot(111)
ax.set_title('Top250导演数量排序', fontsize=17)
df_director.director.value_counts()[df_director.director.value_counts()>2].plot.bar(figsize=(14,7))
plt.xticks(fontsize=15, rotation=60)
plt.yticks(fontsize=15)
ax.set_xlabel('导演', fontsize=17)
ax.set_ylabel('数量', fontsize=17)
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# Load data
data = pd.read_csv("nobel_prize_data.csv")

# Check basic info
print(data.head())
print(data.tail())
print(data.shape)
data.info()
print(data.describe())

# Check for missing & duplicated values
print(f'NaN? {data.isna().values.any()}')
print(f'duplicates? {data.duplicated().values.any()}')

# Drop missing values
data = data.dropna()

# Convert birth_date to datetime
data['birth_date'] = pd.to_datetime(data['birth_date'])

# Calculate share_percentage from prize_share
separated_values = data['prize_share'].str.split('/', expand=True)
numerator = pd.to_numeric(separated_values[0])
denominator = pd.to_numeric(separated_values[1])
data['share_percentage'] = numerator / denominator

# Donut chart of gender distribution
gender = data.sex.value_counts()
chart = px.pie(
    values=gender.values,
    names=gender.index,
    title="Percentage of men's Nobel prizes vs women's",
    hole=0.5
)
chart.update_traces(textposition='inside', textfont_size=12, textinfo='percent')
chart.show()

# First 3 female winners
print(data[data.sex == 'Female'].sort_values('year').head(3))

# Multiple winners
winners_true = data.duplicated(subset=['full_name'], keep=False)
multiple_winners = data[winners_true]
print(f"There are {multiple_winners.full_name.nunique()} winners who were awarded more than once.")
print(multiple_winners[['year', 'category', 'laureate_type', 'full_name']])

# Unique categories
print(data.category.nunique())

# Bar chart: Number of prizes per category
prizes_per_category = data.category.value_counts()
v_bar = px.bar(
    x=prizes_per_category.index,
    y=prizes_per_category.values,
    color=prizes_per_category.values,
    color_continuous_scale='Aggrnyl',
    title='Number of Prizes Awarded per Category'
)
v_bar.update_layout(xaxis_title='Category', yaxis_title='Prizes', coloraxis_showscale=False)
v_bar.show()

# First 3 Economics winners
print(data[data.category == 'Economics'].sort_values('year').head(3))

# Count men and women by category
category_men_women = data.groupby(['category', 'sex'], as_index=False).agg({'prize': 'count'})
category_men_women.sort_values('prize', ascending=False, inplace=True)

# Bar chart: Gender split per category
bar_split = px.bar(
    x=category_men_women.category,
    y=category_men_women.prize,
    title="Prizes per Category Split by Gender",
    color=category_men_women.sex
)
bar_split.update_layout(xaxis_title="Category", yaxis_title="Number of Prizes")
bar_split.show()

# Count prizes per year
prize_per_year = data.groupby('year').count().prize
moving_average = prize_per_year.rolling(window=5).mean()

plt.figure(figsize=(16, 8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1900, 2021, 5), fontsize=14, rotation=45)

ax = plt.gca()
ax.set_xlim(1900, 2020)
ax.scatter(prize_per_year.index, prize_per_year.values, c='dodgerblue', alpha=0.7, s=100)
ax.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=3)
plt.show()

# Yearly average share
yearly_avg_share = data.groupby('year')['share_percentage'].mean()
share_moving_average = yearly_avg_share.rolling(window=5).mean()

plt.figure(figsize=(16, 8), dpi=200)
plt.title('Prize Share & Total Awards per Year', fontsize=18)
plt.xticks(np.arange(1900, 2021, 5), fontsize=14, rotation=45)

axis1 = plt.gca()
axis2 = axis1.twinx()
axis1.set_xlim(1900, 2020)
axis2.invert_yaxis()

axis1.scatter(prize_per_year.index, prize_per_year.values, c='dodgerblue', alpha=0.7, s=100)
axis1.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=2.5)
axis2.plot(yearly_avg_share.index, share_moving_average.values, c="black", linewidth=2.5)
plt.show()

# Top 10 countries by prizes
top_countries = data.groupby('birth_country_current', as_index=False).agg({'prize': 'count'})
top_countries.sort_values('prize', inplace=True)
top10_countries = top_countries.tail(10)

# Horizontal bar chart
horizontal_bar = px.bar(
    x=top10_countries.prize,
    y=top10_countries.birth_country_current,
    orientation='h',
    title="Top 10 Countries by Number of Prizes",
    color=top10_countries.prize,
    color_continuous_scale='Viridis'
)
horizontal_bar.update_layout(xaxis_title="Number of Prizes", yaxis_title="Country", coloraxis_showscale=False)
horizontal_bar.show()

# Choropleth map
countries_dataframe = data.groupby(['birth_country_current', 'ISO'], as_index=False).agg({'prize': 'count'})
choropleth_map = px.choropleth(
    countries_dataframe,
    hover_name='birth_country_current',
    locations='ISO',
    color='prize',
    color_continuous_scale=px.colors.sequential.matter
)
choropleth_map.update_layout(coloraxis_showscale=True)
choropleth_map.show()

# Bar chart of prizes by category for top 10 countries
country_category = data.groupby(['birth_country_current', 'category'], as_index=False).agg({'prize': 'count'})
merged = pd.merge(country_category, top10_countries, on='birth_country_current')
merged.columns = ['birth_country_current', 'category', 'cat_prize', 'total_prize']
merged.sort_values('total_prize', inplace=True)

country_bar = px.bar(
    x=merged.cat_prize,
    y=merged.birth_country_current,
    title='Top 10 Countries: Prizes by Category',
    color=merged.category,
    orientation='h'
)
country_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Country')
country_bar.show()

# Prizes over time by country
prize_by_year = data.groupby(['birth_country_current', 'year'], as_index=False).count()
prize_by_year = prize_by_year[['year', 'birth_country_current', 'prize']]
prizes_cumulative = prize_by_year.groupby(['birth_country_current', 'year']).sum().groupby(level=0).cumsum()
prizes_cumulative.reset_index(inplace=True)

line_chart = px.line(
    prizes_cumulative,
    x='year',
    y='prize',
    color='birth_country_current',
    hover_name='birth_country_current',
    title="Prizes Over Time by Country"
)
line_chart.update_layout(xaxis_title='Year', yaxis_title='Cumulative Prizes')
line_chart.show()

# Top 10 organizations
top10_organization = data.organization_name.value_counts().head(10)
top10_organization.sort_values(ascending=True, inplace=True)

organisation_bar = px.bar(
    x=top10_organization.values,
    y=top10_organization.index,
    orientation='h',
    color=top10_organization.values,
    color_continuous_scale=px.colors.sequential.haline,
    title='Top 10 Research Institutions'
)
organisation_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Institution', coloraxis_showscale=False)
organisation_bar.show()

# Top 10 organization cities
organisation_city = data.organization_city.value_counts().head(10)
organisation_city.sort_values(ascending=True, inplace=True)

organisation_cities_bar = px.bar(
    x=organisation_city.values,
    y=organisation_city.index,
    orientation='h',
    color=organisation_city.values,
    color_continuous_scale=px.colors.sequential.haline,
    title='Top 10 Research Institution Cities'
)
organisation_cities_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='City', coloraxis_showscale=False)
organisation_cities_bar.show()

# Laureates' birth cities
birth_city = data.organization_birth.value_counts().head(10)
birth_city.sort_values(ascending=True, inplace=True)

birth_cities_bar = px.bar(
    x=birth_city.values,
    y=birth_city.index,
    orientation='h',
    color=birth_city.values,
    color_continuous_scale=px.colors.sequential.Plasma,
    title='Top Birth Cities of Nobel Laureates'
)
birth_cities_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='City', coloraxis_showscale=False)
birth_cities_bar.show()

# Sunburst chart
prepared_dataframe_sunburst = data.groupby(
    ['organization_country', 'organization_city', 'organization_name'], as_index=False
).agg({'prize': 'count'}).sort_values('prize', ascending=False)

sunburst_chart = px.sunburst(
    prepared_dataframe_sunburst,
    path=['organization_country', 'organization_city', 'organization_name'],
    values='prize',
    title='Institutional Distribution of Nobel Prizes'
)
sunburst_chart.update_layout(coloraxis_showscale=False)
sunburst_chart.show()

# Calculate age at award
data['winning_age'] = data['year'] - data.birth_date.dt.year
print(data.nlargest(1, 'winning_age'))  # Oldest
print(data.nsmallest(1, 'winning_age'))  # Youngest

# Age stats
print(data.winning_age.describe())

# Histogram of ages
plt.figure(figsize=(15, 7), dpi=150)
sns.histplot(data, x='winning_age', bins=20)
plt.xlabel('Age at Award')
plt.title('Age Distribution of Nobel Laureates')
plt.show()

# Regression plot: age vs year
plt.figure(figsize=(15, 7), dpi=150)
with sns.axes_style("dark"):
    sns.regplot(data=data, x='year', y='winning_age', lowess=True,
                scatter_kws={'alpha': 0.55}, line_kws={'color': '#FF6C0A'})
plt.show()

# Has the Nobel Prize aged with its winners?
with sns.axes_style('whitegrid'):
    sns.lmplot(data=data, x='year', y='winning_age', row='category', lowess=True, aspect=3,
               scatter_kws={'alpha': 0.55}, line_kws={'color': '#FF6C0A'})
plt.show()

# Age trend by category (hue)
with sns.axes_style("whitegrid"):
    sns.lmplot(data=data, x='year', y='winning_age', hue='category', lowess=True,
               aspect=3, scatter_kws={'alpha': 0.55}, line_kws={'linewidth': 2})
plt.show()
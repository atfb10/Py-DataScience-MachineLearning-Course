'''
Author: Adam Forestier
Date: March 30, 2023
Questions to Answer:
    a.) Is there a conflict of interest for a website that both sells movie tickets and displays review ratings
    b.) Does a website like Fandango artificially display higher review ratings to sell more movie tickets

Notes:
    Fandango has two ratings
        1. Stars = ratings in stars displayed on their website's HTML
        2. Rating = actual true rating numerically shown on the movie's page

Steps: 
    1. Compare Fandango's Star rating vs rating
    2. Compare Fandango's ratings to other website's ratings
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

all_sites = pd.read_csv('all_sites_scores.csv')
fandango = pd.read_csv('fandango_scrape.csv')

# TASK: Explore the DataFrame Properties and Head.
print(fandango.info())
print('--------------------------------------------------------------')
print(fandango.head())
print('--------------------------------------------------------------')


# Task: show relatonship between rating and votes
plt.title('Rating by Vote Count')
sns.scatterplot(data=fandango, x='RATING', y='VOTES', color='Orange', edgecolor='black')
plt.show()

# Task: Calculate correlation between columns
print(fandango.corr())
print('--------------------------------------------------------------')

# Task Create a new column that is able to strip the year from the title strings and set this new column as YEAR
def strip_year(film: str):
    title_split = film.split()
    return title_split[len(title_split) -1].strip('()')
fandango['YEAR'] = np.vectorize(strip_year)(fandango['FILM'])

# Task: How many movies are in the Fandango DataFrame per year
print(fandango['YEAR'].value_counts())
print('--------------------------------------------------------------')

# Task: Visualize the count of movies per year with a plot:
plt.title('Movies by Year')
sns.countplot(data=fandango, x='YEAR')
plt.show()

# TASK: What are the 10 movies with the highest number of votes?
print(fandango.nlargest(n=10,columns='VOTES'))
print('--------------------------------------------------------------')

# TASK: How many movies have zero votes?
print(fandango['VOTES'].value_counts()[0])
print('--------------------------------------------------------------')

# TASK: Create DataFrame of only reviewed films by removing any films that have zero votes.
fandango = fandango[fandango['VOTES'] > 0]


# TASK: Create a KDE plot (or multiple kdeplots) that displays the distribution of ratings that are displayed (STARS) versus what the true rating was from votes (RATING). Clip the KDEs to 0-5.
plt.title('True Rating vs Stars Displayed')
sns.kdeplot(x='RATING', data=fandango, clip=[0, 5], fill=True) 
sns.kdeplot(x='STARS', data=fandango, clip=[0, 5], fill=True) 
plt.show()

# TASK: Let's now actually quantify this discrepancy. Create a new column of the different between STARS displayed versus true RATING. Calculate this difference with STARS-RATING and round these differences to the nearest decimal point.
fandango['STARS_DIFF'] = round(fandango['STARS'] - fandango['RATING'], 1)

# TASK: Create a count plot to display the number of times a certain difference occurs:
plt.title('Count of Times Rating Difference Occurs')
sns.countplot(x='STARS_DIFF', data=fandango)
plt.show()

# TASK: We can see from the plot that one movie was displaying over a 1 star difference than its true rating! What movie had this close to 1 star differential?
print(fandango.loc[fandango['STARS_DIFF'] == 1.0])


# TASK: Explore the all_sites DataFrame columns, info, description.
all_sites_copy = pd.read_csv('all_sites_scores.csv')
print(all_sites.head())
print('--------------------------------------------------------------')
print(all_sites.info())
print('--------------------------------------------------------------')
print(all_sites.describe())
print('--------------------------------------------------------------')


# TASK: Create a scatterplot exploring the relationship between RT Critic reviews and RT User reviews.
plt.title('Rotten Tomatoes Critic vs User Reviews')
sns.scatterplot(x='RottenTomatoes', y='RottenTomatoes_User', data=all_sites)
plt.show()

# TASK: Create a new column based off the difference between critics ratings and users ratings for Rotten Tomatoes. Calculate this with RottenTomatoes-RottenTomatoes_User
all_sites['RottenTomatoes_Diff'] = all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']

# TASK: Calculate the Mean Absolute Difference between RT scores and RT User scores as described above.
absolute_mean_diff = all_sites['RottenTomatoes_Diff'].abs().mean()
print(absolute_mean_diff)
print('--------------------------------------------------------------')

# TASK: Plot the distribution of the differences between RT Critics Score and RT User Score. There should be negative values in this distribution plot. Feel free to use KDE or Histograms to display this distribution.
sns.displot(x='RottenTomatoes_Diff', data=all_sites, kde=True)
plt.title('RT critics Score Minus RT User Score')
plt.show()

# TASK: What are the top 5 movies critics rated higher than critics on average:
print(all_sites[['FILM', 'RottenTomatoes_Diff']].nsmallest(n=5, columns=['RottenTomatoes_Diff']))
print('--------------------------------------------------------------')
print(all_sites[['FILM', 'RottenTomatoes_Diff']].nlargest(n=5, columns=['RottenTomatoes_Diff']))
print('--------------------------------------------------------------')

# TASK: Now create a distribution showing the absolute value difference between Critics and Users on Rotten Tomatoes.
all_sites_abs = all_sites
all_sites_abs['RottenTomatoes_Diff'] = all_sites_abs['RottenTomatoes_Diff'].abs()
sns.displot(x='RottenTomatoes_Diff', data=all_sites_abs, kde=True)
plt.title('Absolute Difference RT critics Score Minus RT User Score')
plt.show()


all_sites = all_sites_copy
all_sites['RottenTomatoes_Diff'] = all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']
# TASK: Display a scatterplot of the Metacritic Rating versus the Metacritic User rating.
sns.scatterplot(x='Metacritic', y='Metacritic_User', data=all_sites)
plt.show()

# TASK: Create a scatterplot for the relationship between vote counts on MetaCritic versus vote counts on IMDB.
sns.scatterplot(x='Metacritic_user_vote_count', y='IMDB_user_vote_count', data=all_sites, alpha=.5)
plt.show()

# TASK: What movie has the highest IMDB user vote count?
print(all_sites.nlargest(n=1, columns=['IMDB_user_vote_count']))
print('--------------------------------------------------------------')

# TASK: What movie has the highest Metacritic User Vote count?
print(all_sites.nlargest(n=1, columns=['Metacritic_user_vote_count']))
print('--------------------------------------------------------------')

# TASK: Combine the Fandango Table with the All Sites table. Not every movie in the Fandango table is in the All Sites table, since some Fandango movies have very little or no reviews. We only want to compare movies that are in both DataFrames, so do an inner merge to merge together both DataFrames based on the FILM columns.
df = pd.merge(fandango, all_sites, how='inner', on='FILM')

# TASK: Create new normalized columns for all ratings so they match up within the 0-5 star range shown on Fandango. There are many ways to do this.
df['RottenTomatoes'] = round(df['RottenTomatoes'] / 20, 1)
df['RottenTomatoes_User'] = round(df['RottenTomatoes_User'] / 20, 1)
df['RottenTomatoes_Diff'] = round(df['RottenTomatoes_Diff'] / 20, 1)
df['Metacritic'] = round(df['Metacritic'] / 20, 1)
df['Metacritic_User'] = round(df['Metacritic_User'] / 2, 1)
df['IMDB'] = round(df['IMDB'] / 2, 1)

# TASK: Now create a norm_scores DataFrame that only contains the normalizes ratings. Include both STARS and RATING from the original Fandango table.
norm_df = df[['FILM', 'STARS', 'RATING', 'RottenTomatoes', 'RottenTomatoes_User', 'Metacritic', 'Metacritic_User', 'IMDB']]
norm_df_copy = norm_df

print(norm_df.head())

# TASK: Create a plot comparing the distributions of normalized ratings across all sites
sns.kdeplot(x='STARS', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='RATING', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='RottenTomatoes', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='RottenTomatoes_User', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='Metacritic', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='Metacritic_User', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='IMDB', data=norm_df, fill=True, clip=[0,5])
plt.legend(labels=['STARS', 'RATING', 'RottenTomatoes', 'RottenTomatoes_User', 'Metacritic', 'Metacritic_User', 'IMDB'], bbox_to_anchor=(.45, .85))
plt.show()

# TASK: Create a KDE plot that compare the distribution of RT critic ratings against the STARS displayed by Fandango.
sns.kdeplot(x='STARS', data=norm_df, fill=True, clip=[0,5])
sns.kdeplot(x='RottenTomatoes', data=norm_df, fill=True, clip=[0,5])
plt.legend(labels=['STARS', 'RottenTomatoes'])
plt.show()

# TASK: Create a clustermap visualization of all normalized scores
sns.clustermap(norm_df.drop('FILM', axis=1), col_cluster=False) # col_clustering = False makes clustering based only on index
plt.show()

# TASK: Clearly Fandango is rating movies much higher than other sites, especially considering that it is then displaying a rounded up version of the rating. Let's examine the top 10 worst movies. Based off the Rotten Tomatoes Critic Ratings, what are the top 10 lowest rated movies? What are the normalized scores across all platforms for these movies
norm_df = norm_df_copy
worst_movies = norm_df.nsmallest(n=10, columns=['RottenTomatoes'])
print(worst_movies)

# FINAL TASK: Visualize the distribution of ratings across all sites for the top 10 worst movies.
sns.kdeplot(x='STARS', data=worst_movies, fill=True, clip=[0,5])
sns.kdeplot(x='RATING', data=worst_movies, fill=True, clip=[0,5])
sns.kdeplot(x='RottenTomatoes', data=worst_movies, fill=True, clip=[0,5])
sns.kdeplot(x='RottenTomatoes_User', data=worst_movies, fill=True, clip=[0,5])
sns.kdeplot(x='Metacritic', data=worst_movies, fill=True, clip=[0,5])
sns.kdeplot(x='Metacritic_User', data=worst_movies, fill=True, clip=[0,5])
sns.kdeplot(x='IMDB', data=worst_movies, fill=True, clip=[0,5])
plt.legend(labels=['STARS', 'RATING', 'RottenTomatoes', 'RottenTomatoes_User', 'Metacritic', 'Metacritic_User', 'IMDB'], bbox_to_anchor=(.85, .9))
plt.show()
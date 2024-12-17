import pandas as pd
import numpy as np
import sys

class PopularityRecommender:
    def __init__(self):
        self.data_loader = MovieDataLoader()
        self.movie_stats = None
        
    def fit(self, ratings_path, movies_path, users_path):
        """
        Train the popularity based recommender
        """
        # Load data
        self.data_loader.load_raw_data(ratings_path, movies_path, users_path)
        
        # Calculate movie statistics
        self.movie_stats = self.data_loader.get_movie_stats()
        
        # Define popularity metric
        # We'll consider a movie popular if it has:
        # 1. More than median number of ratings (ensures enough data)
        # 2. High average rating (quality content)
        median_ratings = self.movie_stats['rating_count'].median()
        
        self.movie_stats['popularity_score'] = (
            self.movie_stats['rating_mean'] * 
            (self.movie_stats['rating_count'] > median_ratings)
        )
        
        # Ignore movies with too few ratings
        self.movie_stats.loc[self.movie_stats['rating_count'] <= median_ratings, 'popularity_score'] = 0
        
    def get_top_n_movies(self, n=10):
        """Get the top n most popular movies"""
        if self.movie_stats is None:
            raise ValueError("Please fit the recommender first using fit()")
            
        top_movies = (self.movie_stats
                     .nlargest(n, 'popularity_score')
                     .copy())
        
        # Format output as required
        recommendations = top_movies.apply(
            lambda x: {
                'MovieID': f"m{int(x['MovieID'])}",
                'Title': x['Title'],
                'Rating': round(x['rating_mean'], 2),
                'Number_of_Ratings': int(x['rating_count'])
            }, axis=1
        ).tolist()
        
        return recommendations

def main():
    # Test the implementation
    recommender = PopularityRecommender()
    
    # Fit the recommender
    recommender.fit(
        'data/ratings.dat',
        'data/movies.dat',
        'data/users.dat'
    )
    
    # Get recommendations
    top_movies = recommender.get_top_n_movies(10)
    
    # Print recommendations in a nice format
    print("\nTop 10 Movie Recommendations:")
    print("-" * 80)
    for i, movie in enumerate(top_movies, 1):
        print(f"{i}. {movie['Title']}")
        print(f"   MovieID: {movie['MovieID']}")
        print(f"   Average Rating: {movie['Rating']}")
        print(f"   Number of Ratings: {movie['Number_of_Ratings']}")
        print()

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import MovieDataLoader
else:
    from src.data_loader import MovieDataLoader
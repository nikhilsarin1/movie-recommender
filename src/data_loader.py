import pandas as pd
import numpy as np

class MovieDataLoader:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.ratings_matrix = None

    def load_raw_data(self, ratings_path, movies_path, users_path):
        """Load the original MovieLens data files"""
        # Load ratings
        self.ratings_df = pd.read_csv(ratings_path, 
                                    sep='::', 
                                    engine='python',
                                    encoding='latin1',
                                    names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        
        # Load movies - using a different approach for movies due to encoding
        with open(movies_path, 'r', encoding='latin1') as f:
            movies_lines = f.readlines()
        
        # Process movies data
        movies_data = []
        for line in movies_lines:
            movieID, title, genres = line.strip().split('::')
            movies_data.append([int(movieID), title, genres])
        
        self.movies_df = pd.DataFrame(movies_data, columns=['MovieID', 'Title', 'Genres'])
        
        # Load users
        self.users_df = pd.read_csv(users_path,
                                  sep='::', 
                                  engine='python',
                                  encoding='latin1',
                                  names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        
        return self.ratings_df, self.movies_df, self.users_df

    def load_rating_matrix(self, matrix_path):
        """Load the pre-processed rating matrix"""
        self.ratings_matrix = pd.read_csv(matrix_path, index_col=0)
        return self.ratings_matrix

    def get_movie_stats(self):
        """Calculate basic statistics for each movie"""
        if self.ratings_df is None:
            raise ValueError("Raw ratings data not loaded yet")
            
        movie_stats = self.ratings_df.groupby('MovieID').agg({
            'Rating': ['count', 'mean', 'std']
        }).reset_index()
        
        movie_stats.columns = ['MovieID', 'rating_count', 'rating_mean', 'rating_std']
        movie_stats = movie_stats.merge(self.movies_df[['MovieID', 'Title']], on='MovieID')
        
        return movie_stats

    def create_rating_matrix(self):
        """Create rating matrix from raw ratings data"""
        if self.ratings_df is None:
            raise ValueError("Raw ratings data not loaded yet")
            
        # Create the matrix
        matrix = self.ratings_df.pivot(
            index='UserID',
            columns='MovieID',
            values='Rating'
        )
        
        # Rename columns to match provided format
        matrix.columns = [f'm{col}' for col in matrix.columns]
        matrix.index = [f'u{idx}' for idx in matrix.index]
        
        self.ratings_matrix = matrix
        return matrix

    def save_rating_matrix(self, save_path):
        """Save rating matrix to CSV"""
        if self.ratings_matrix is not None:
            self.ratings_matrix.to_csv(save_path)
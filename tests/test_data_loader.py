import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MovieDataLoader
import pandas as pd
import numpy as np

def test_data_loader():
    # Initialize loader
    loader = MovieDataLoader()
    
    # 1. Test loading raw files
    print("Testing raw data loading...")
    ratings_df, movies_df, users_df = loader.load_raw_data(
        'data/ratings.dat',
        'data/movies.dat',
        'data/users.dat'
    )
    
    # Verify dimensions and content
    print("\nData dimensions:")
    print(f"Ratings: {ratings_df.shape} (should be around 1M rows × 4 columns)")
    print(f"Movies: {movies_df.shape} (should be around 3900 rows × 3 columns)")
    print(f"Users: {users_df.shape} (should be 6040 rows × 5 columns)")
    
    # 2. Test rating matrix loading
    print("\nTesting rating matrix loading...")
    matrix = loader.load_rating_matrix('data/ratings_matrix.csv')
    print(f"Rating matrix shape: {matrix.shape} (should be 6040 × 3706)")
    
    # 3. Test movie statistics
    print("\nTesting movie statistics...")
    movie_stats = loader.get_movie_stats()
    print("\nTop 5 movies by number of ratings:")
    print(movie_stats.nlargest(5, 'rating_count')[['Title', 'rating_count', 'rating_mean']])
    
    # 4. Test matrix creation from raw data
    print("\nTesting matrix creation from raw data...")
    created_matrix = loader.create_rating_matrix()
    print(f"Created matrix shape: {created_matrix.shape}")
    
    # 5. Verify a specific user's ratings
    print("\nVerifying specific user ratings...")
    user1_ratings = matrix.loc['u1'].dropna()
    print(f"\nUser 1 has rated {len(user1_ratings)} movies")
    print("\nSample of User 1's ratings:")
    print(user1_ratings.head())
    
    return "All tests completed!"

if __name__ == "__main__":
    test_data_loader()
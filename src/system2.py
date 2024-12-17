import numpy as np
import pandas as pd
import os
import sys

class IBCFRecommender:
    def __init__(self):
        self.data_loader = MovieDataLoader()
        self.similarity_matrix = None
        self.ratings_matrix = None
        self.movies_df = None
        self.centered_ratings = None  # Store for verification

    def load_compressed_data(self, filepath):
        import gzip
        print(f"Loading compressed file: {filepath}")
        if filepath.endswith('.csv.gz'):
            with gzip.open(filepath, 'rt') as f:
                return pd.read_csv(f)
        elif filepath.endswith('.npy.gz'):
            with gzip.open(filepath, 'rb') as f:
                return np.load(f)
        
    def verify_first_rows_centering(self, centered_data):
        """Verify the first 10 rows of centered data match expected values"""
        print("\nVerifying First 10 Rows of Centered Matrix:")
        print("-" * 50)
        
        # Convert centered numpy array back to dataframe for easier comparison
        centered_df = pd.DataFrame(
            centered_data,
            index=self.ratings_matrix.index,
            columns=self.ratings_matrix.columns
        )
        
        # Expected values from the image
        expected_values = {
            ('u1', 'm1'): 0.8113208,
            ('u10', 'm1'): 0.8852868,
            ('u100', 'm1'): np.nan,
            ('u1000', 'm1'): 0.8690476,
            ('u1001', 'm1'): 0.3474801,
            ('u999', 'm1003'): -1.1868932,
            ('u999', 'm1004'): -0.1868932,
            ('u996', 'm999'): -0.9358108
        }
        
        print("\nComparing values:")
        for (user, movie), expected in expected_values.items():
            actual = centered_df.loc[user, movie]
            if np.isnan(expected):
                is_match = np.isnan(actual)
            else:
                is_match = np.abs(actual - expected) < 0.0001
            
            print(f"{user}, {movie}:")
            print(f"  Expected: {expected}")
            print(f"  Actual: {actual}")
            print(f"  Match: {is_match}")
            print()
        
    def verify_similarities_step2(self, computed_similarity_matrix):
        """Verify similarity computations against provided CSV file"""
        print("\nStep 2 Verification - Comparing Similarity Matrix:")
        print("-" * 50)
        
        # Load the verification data
        expected_df = pd.read_csv('data/verification/Step2Output-10Rows.csv', index_col=0)
        
        # First display the specific subset for visualization
        specific_ids = ['m1', 'm10', 'm100', 'm1510', 'm260', 'm3212']
        specific_indices = [self.column_to_index[mid] for mid in specific_ids]
        
        specific_df = pd.DataFrame(
            computed_similarity_matrix[np.ix_(specific_indices, specific_indices)],
            index=specific_ids,
            columns=specific_ids
        )
        
        print("\nSpecified Movies Similarity Matrix:")
        print(specific_df.round(7))
        
        # For complete verification, use all movies from the CSV
        movie_ids = expected_df.index.tolist()
        indices = [self.column_to_index[mid] for mid in movie_ids]
        
        computed_df = pd.DataFrame(
            computed_similarity_matrix[np.ix_(indices, indices)],
            index=movie_ids,
            columns=movie_ids
        )
        
        # Compare all values
        print("\nComparing all similarity values:")
        mismatches = []
        total_comparisons = 0
        nan_matches = 0
        
        for i in movie_ids:
            for j in movie_ids:
                total_comparisons += 1
                expected = expected_df.loc[i, j]
                computed = computed_df.loc[i, j]
                
                if pd.isna(expected) and pd.isna(computed):
                    nan_matches += 1
                    continue
                elif pd.isna(expected) != pd.isna(computed):
                    mismatches.append((i, j, expected, computed, "NaN mismatch"))
                elif abs(expected - computed) > 1e-7:
                    mismatches.append((i, j, expected, computed, "Value mismatch"))
        
        print(f"\nTotal comparisons: {total_comparisons}")
        print(f"NaN matches: {nan_matches}")
        print(f"Non-NaN comparisons: {total_comparisons - nan_matches}")
        
        if not mismatches:
            print("All checked values match within tolerance!")
        else:
            print("\nMismatches found:")
            print("MovieID1  MovieID2  Expected      Computed      Type")
            print("-" * 60)
            for m1, m2, exp, comp, type_mismatch in mismatches:
                if pd.isna(exp):
                    print(f"{m1:8} {m2:8}       nan    {comp:12.7f} {type_mismatch}")
                elif pd.isna(comp):
                    print(f"{m1:8} {m2:8} {exp:12.7f}          nan {type_mismatch}")
                else:
                    print(f"{m1:8} {m2:8} {exp:12.7f} {comp:12.7f} {type_mismatch}")
        
        self.similarity_matrix = computed_similarity_matrix
        return len(mismatches) == 0
    
    def verify_similarities_step3(self, filtered_similarity_matrix):
        """Verify the top-30 filtering results"""
        print("\nStep 3 Verification - Comparing Filtered Matrix:")
        print("-" * 50)
        
        # Load the verification data
        expected_df = pd.read_csv('data/verification/Step3Output-10Rows.csv', index_col=0)
        
        # Get first 10 movies for comparison
        movie_ids = ['m1', 'm10', 'm100', 'm1000', 'm1002', 'm1003', 'm1004', 'm1005', 'm1006', 'm1007']
        indices = [self.column_to_index[mid] for mid in movie_ids]
        
        computed_df = pd.DataFrame(
            filtered_similarity_matrix[np.ix_(indices, indices)],
            index=movie_ids,
            columns=movie_ids
        )
        
        # Compare values
        print("\nComparing filtered similarity values:")
        mismatches = []
        nan_matches = 0
        total_comparisons = 0
        
        for i in movie_ids:
            for j in movie_ids:
                expected = expected_df.loc[i, j]
                computed = computed_df.loc[i, j]
                total_comparisons += 1
                
                # Both should be NaN, or both should be values
                if pd.isna(expected) and pd.isna(computed):
                    nan_matches += 1
                    continue
                    
                if pd.isna(expected) != pd.isna(computed):
                    mismatches.append((i, j, expected, computed, "NaN mismatch"))
                elif not pd.isna(expected) and not pd.isna(computed):
                    if abs(expected - computed) > 1e-7:
                        mismatches.append((i, j, expected, computed, "Value mismatch"))
        
        # Print results
        print(f"\nTotal comparisons: {total_comparisons}")
        print(f"NaN matches: {nan_matches}")
        print(f"Non-NaN comparisons: {total_comparisons - nan_matches}")
        
        if mismatches:
            print("\nMismatches found:")
            print("MovieID1  MovieID2  Expected      Computed      Type")
            print("-" * 60)
            for m1, m2, exp, comp, type_mismatch in mismatches:
                if pd.isna(exp):
                    print(f"{m1:8} {m2:8}       nan    {comp:12.7f} {type_mismatch}")
                elif pd.isna(comp):
                    print(f"{m1:8} {m2:8} {exp:12.7f}          nan {type_mismatch}")
                else:
                    print(f"{m1:8} {m2:8} {exp:12.7f} {comp:12.7f} {type_mismatch}")
        else:
            print("\nAll checked values match within tolerance!")
            
        # Print filtered matrix
        print("\nFiltered Similarity Matrix (First 5x5):")
        print(computed_df.iloc[:5, :5].round(7))

        # Check specific known value
        m1000_idx = self.column_to_index['m1000']
        m1021_idx = self.column_to_index['m1021']
        actual_value = filtered_similarity_matrix[m1000_idx, m1021_idx]
        expected_value = 0.9700115756880312
        
        print("\nChecking specific value (m1000, m1021):")
        print(f"Expected: {expected_value:.7f}")
        print(f"Actual:   {actual_value:.7f}")
        print(f"Match:    {abs(actual_value - expected_value) < 1e-7}")
        
        # Given that there might be multiple valid solutions,
        # we should also verify that each movie has ≤ 30 non-NaN values
        non_nan_counts = np.sum(~np.isnan(filtered_similarity_matrix), axis=1)
        max_non_nan = np.max(non_nan_counts)
        print(f"\nMaximum non-NaN values for any movie: {max_non_nan}")
        if max_non_nan > 30:
            print("WARNING: Some movies have more than 30 similarities!")
            
        return len(mismatches) == 0 and max_non_nan <= 30
        
    def fit(self, ratings_matrix_path, movies_path, load_step2=False):
        """Train the IBCF recommender with verification steps"""
        print("Loading data...")
        try:
            self.ratings_matrix = self.load_compressed_data(ratings_matrix_path + '.gz')
        except FileNotFoundError:
            self.ratings_matrix = pd.read_csv(ratings_matrix_path)        
        # Create mappings between movie IDs and column indices
        self.column_to_index = {col: idx for idx, col in enumerate(self.ratings_matrix.columns)}
        self.index_to_column = {idx: col for col, idx in self.column_to_index.items()}
        
        # Convert matrix to numpy and use float32 for better memory usage
        ratings_array = self.ratings_matrix.values.astype(np.float32)
        n_movies = ratings_array.shape[1]  # Define n_movies here since we need it for both paths
        
        print("Data loaded successfully.")
        
        print("\nStep 1: Centering the Rating Matrix")
        # 1. Center the ratings (subtract row means)
        row_means = np.nanmean(ratings_array, axis=1, keepdims=True)
        # Initialize with NaN instead of zeros
        centered_ratings = np.full_like(ratings_array, np.nan)
        for i in range(ratings_array.shape[0]):
            mask = ~np.isnan(ratings_array[i, :])
            if np.any(mask):
                centered_ratings[i, mask] = ratings_array[i, mask] - row_means[i]

        self.centered_ratings = centered_ratings
        
        # Verify centering
        self.verify_first_rows_centering(centered_ratings)
        
        print("\nStep 2: Computing Similarity Matrix")
        if load_step2 and os.path.exists('data/step2_similarity_matrix.npy'):
            print("Loading pre-computed similarity matrix from step 2...")
            similarity_matrix = np.load('data/step2_similarity_matrix.npy')
            # Verify the loaded matrix
            verification_passed = self.verify_similarities_step2(similarity_matrix)
            if not verification_passed:
                print("\nWARNING: Loaded similarity matrix may need adjustment!")
            else:
                print("\nLoaded similarity matrix verification passed!")
        else:
            similarity_matrix = np.full((n_movies, n_movies), np.nan)
            
            for i in range(n_movies):
                for j in range(i+1, n_movies):
                    # Get users who rated both movies (Iij)
                    mask = ~np.isnan(ratings_array[:, i]) & ~np.isnan(ratings_array[:, j])
                    common_users = np.sum(mask)
                    
                    # Only compute if more than 2 common ratings
                    if common_users > 2:
                        # Use original ratings
                        ratings_i = centered_ratings[mask, i]
                        ratings_j = centered_ratings[mask, j]
                        
                        # Compute similarity with corrected formula
                        numerator = np.sum(ratings_i * ratings_j)
                        denom_i = np.sqrt(np.sum(ratings_i**2))
                        denom_j = np.sqrt(np.sum(ratings_j**2))
                        denominator = denom_i * denom_j
                        
                        if denominator > 0:
                            sim = 0.5 + 0.5 * (numerator / denominator)
                            similarity_matrix[i, j] = sim
                            similarity_matrix[j, i] = sim
            
            # Set diagonal to NaN
            np.fill_diagonal(similarity_matrix, np.nan)
            
            # Verify similarities
            verification_passed = self.verify_similarities_step2(similarity_matrix)
            if not verification_passed:
                print("\nWARNING: Similarity matrix computation may need adjustment!")
            else:
                print("\nSimilarity matrix verification passed!")
                
            # Save the step 2 matrix
            np.save('data/step2_similarity_matrix.npy', similarity_matrix)
            print("Step 2 similarity matrix saved")
        
        print("\nStep 3: Applying Top-30 Filtering")
        filtered_similarity_matrix = similarity_matrix.copy()

        for i in range(n_movies):
            similarities = filtered_similarity_matrix[i, :]
            non_nan_indices = np.where(~np.isnan(similarities))[0]
            
            if len(non_nan_indices) > 30:
                # Get similarities for non-NaN indices
                non_nan_sims = similarities[non_nan_indices]
                # Get indices that would sort these similarities in descending order
                sorted_indices = np.argsort(-non_nan_sims)  # Note the minus sign for descending order
                # Keep only top 30
                keep_indices = non_nan_indices[sorted_indices[:30]]  # Changed to get first 30 from sorted
                # Set all to NaN first
                filtered_similarity_matrix[i, :] = np.nan
                # Then set the top 30 values
                filtered_similarity_matrix[i, keep_indices] = similarity_matrix[i, keep_indices]

        self.similarity_matrix = filtered_similarity_matrix

        verification_passed = self.verify_similarities_step3(filtered_similarity_matrix)

        if not verification_passed:
            print("\nWARNING: Top-30 filtering may need adjustment!")
        else:
            print("\nTop-30 filtering verification passed!")

        return filtered_similarity_matrix

    def predict(self, user_vector):
        """Generate predictions for a user vector using IBCF"""
        if self.similarity_matrix is None:
            raise ValueError("Please fit the recommender first using fit()")
            
        predictions = np.zeros(len(user_vector))
        
        # Find which movies have been rated
        rated_indices = np.where(~np.isnan(user_vector) & (user_vector > 0))[0]  # Added non-NaN check
        
        # For each movie we want to predict
        for i in range(len(user_vector)):
            if np.isnan(user_vector[i]) or user_vector[i] == 0:  # Modified condition
                weighted_sum = 0
                sum_similarities = 0
                
                # Look at each rated movie
                for rated_idx in rated_indices:
                    sim = self.similarity_matrix[i, rated_idx]
                    if not np.isnan(sim):  # If there's a valid similarity
                        weighted_sum += sim * user_vector[rated_idx]
                        sum_similarities += sim
                
                # Compute prediction if we have valid similarities
                if sum_similarities > 0:
                    predictions[i] = weighted_sum / sum_similarities
                else:
                    predictions[i] = np.nan
            else:
                predictions[i] = np.nan
        
        return predictions

    def get_top_n_recommendations(self, predictions, n=10):
        """Helper function to format top n recommendations"""
        top_indices = np.argsort(predictions)[-n:][::-1]
        recommendations = []
        for idx in top_indices:
            movie_id = int(self.ratings_matrix.columns[idx][1:])
            recommendations.append({
                'MovieID': f"m{movie_id}",
                'Rating': predictions[idx]
            })
        return recommendations

    def save_similarity_matrix(self, file_path='data/similarity_matrix.npy'):
        """Save the similarity matrix to a file"""
        if self.similarity_matrix is not None:
            np.save(file_path, self.similarity_matrix)
            print(f"Similarity matrix saved to {file_path}")
        else:
            print("No similarity matrix to save")

    def load_similarity_matrix(self, file_path='data/similarity_matrix.npy'):
        """Load the similarity matrix from a file"""
        try:
            self.similarity_matrix = np.load(file_path)
            print(f"Similarity matrix loaded from {file_path}")
            return True
        except FileNotFoundError:
            print(f"No similarity matrix found at {file_path}")
            return False
        
    def check_similarity_matrix(self):
        """Debug the similarity matrix"""
        print("\nSimilarity Matrix Stats:")
        print(f"Shape: {self.similarity_matrix.shape}")
        print(f"Non-NaN values: {np.sum(~np.isnan(self.similarity_matrix))}")
        print(f"Range: {np.nanmin(self.similarity_matrix)} to {np.nanmax(self.similarity_matrix)}")
        
        # Check first few rows
        print("\nFirst 5x5 similarities:")
        print(pd.DataFrame(
            self.similarity_matrix[:5, :5],
            index=self.ratings_matrix.columns[:5],
            columns=self.ratings_matrix.columns[:5]
        ).round(4))

    def myIBCF(self, newuser):
        """
        Input: newuser - a 3706-by-1 vector containing ratings for the 3,706 movies
        Output: List of top 10 movie IDs with their predicted ratings
        """
        # Use existing predict function to get predictions
        predictions = self.predict(newuser)
        
        # Convert to series and sort
        pred_series = pd.Series(predictions, index=self.ratings_matrix.columns)
        sorted_preds = pred_series.sort_values(ascending=False)
        non_nan_preds = sorted_preds.dropna()
        
        # Get top 10 recommendations with ratings
        top_10 = non_nan_preds[:10]
        recommendations = [(movie_id, rating) for movie_id, rating in top_10.items()]
        
        return recommendations

    def test_recommendations(self):
        print("\nTesting Recommendations:")
        print("-" * 50)

        # Test Case 1: User 1181
        print("\nTest Case 1 - User 1181:")
        user1181_vector = self.ratings_matrix.loc['u1181'].values
        
        # Print some of user's ratings
        ratings_series = pd.Series(user1181_vector, index=self.ratings_matrix.columns)
        print("\nSample of user 1181's top ratings:")
        print(ratings_series[ratings_series > 0].sort_values(ascending=False).head())

        recommendations = self.myIBCF(user1181_vector)
        
        print("\nTop 10 recommendations for User 1181:")
        for movie_id, rating in recommendations:
            print(f"{movie_id}: {rating:.4f}")

        # Test Case 2: Hypothetical User
        print("\nTest Case 2 - Hypothetical User (m1613: 5, m1755: 4):")
        hypo_user_vector = np.zeros(len(self.ratings_matrix.columns))
        hypo_user_vector[self.column_to_index['m1613']] = 5
        hypo_user_vector[self.column_to_index['m1755']] = 4
        
        recommendations2 = self.myIBCF(hypo_user_vector)
        
        print("\nTop 10 recommendations for Hypothetical User:")
        for movie_id, rating in recommendations2:
            print(f"{movie_id}: {rating:.4f}")


def main():
    recommender = IBCFRecommender()
    
    recommender.fit('data/Rmat.csv', 'data/movies.dat', load_step2=True)  # Set to False to recompute step 2
    
    recommender.test_recommendations()

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import MovieDataLoader
else:
    from src.data_loader import MovieDataLoader

if __name__ == "__main__":
    print("\nRunning system2.py directly...")
    main()
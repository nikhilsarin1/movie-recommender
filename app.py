import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system2 import IBCFRecommender
from src.system1 import PopularityRecommender
from src.data_loader import MovieDataLoader

# Configure the page
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_recommenders():
    # Load IBCF recommender
    ibcf = IBCFRecommender()
    ibcf.fit(
        os.path.join('data', 'Rmat.csv'),
        os.path.join('data', 'movies.dat'),
        load_step2=True
    )
    
    # Load popularity recommender
    pop = PopularityRecommender()
    pop.fit(
        os.path.join('data', 'ratings.dat'),
        os.path.join('data', 'movies.dat'),
        os.path.join('data', 'users.dat')
    )
    
    return ibcf, pop

@st.cache_data
def load_movie_data():
    loader = MovieDataLoader()
    _, movies_df, _ = loader.load_raw_data(
        os.path.join("data", "ratings.dat"),
        os.path.join("data", "movies.dat"),
        os.path.join("data", "users.dat")
    )
    return movies_df

def get_movie_pool(pop_recommender, n=100):
    """Get a pool of popular movies to show users"""
    return pop_recommender.get_top_n_movies(n)

def display_movie_pair(movies, start_idx):
    """Display a pair of movies side by side with consistent spacing"""
    cols = st.columns(2)
    
    for i, col in enumerate(cols):
        idx = start_idx + i
        if idx < len(movies):
            movie = movies[idx]
            movie_id = movie['MovieID']
            
            with col:
                # Container for consistent spacing
                with st.container():
                    # Create a fixed height container for the title
                    # Increased height to 5rem to accommodate two lines
                    st.markdown(
                        f"""
                        <div style="height: 5.5rem; display: flex; align-items: center;">
                            <h3 style="margin: 0; line-height: 1.3;">{movie['Title']}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Stats in a fixed height container
                    st.write(f"Average Rating: {movie['Rating']:.2f}")
                    st.write(f"Number of Ratings: {movie['Number_of_Ratings']:,}")
                    
                    # Movie poster in fixed height container
                    try:
                        movie_num = int(movie_id[1:])
                        image_path = os.path.join("data", "MovieImages", f"{movie_num}.jpg")
                        if os.path.exists(image_path):
                            st.image(image_path, width=200)
                        else:
                            # Placeholder container for missing poster
                            st.markdown("*(No poster available)*")
                            st.write("")  # Add some spacing
                    except Exception:
                        st.markdown("*(Error loading poster)*")
                        st.write("")  # Add some spacing
                    
                    # Rating slider
                    rating = st.slider(
                        "Rate this movie",
                        min_value=0,
                        max_value=5,
                        value=0,
                        key=f"movie_{movie_id}"
                    )
                    
                    if rating > 0:
                        st.session_state.user_ratings[movie_id] = rating
                    elif movie_id in st.session_state.user_ratings:
                        del st.session_state.user_ratings[movie_id]
                
                # Add consistent spacing between movie pairs
                st.write("---")

def display_recommendation_pair(recommendations, movies_df, start_idx):
    """Display a pair of recommended movies side by side with consistent spacing"""
    cols = st.columns(2)
    
    for i, col in enumerate(cols):
        idx = start_idx + i
        if idx < len(recommendations):
            movie_id, pred_rating = recommendations[idx]
            movie_num = int(movie_id[1:])
            movie_info = movies_df[movies_df['MovieID'] == movie_num].iloc[0]
            
            with col:
                # Container for consistent spacing
                with st.container():
                    # Create a fixed height container for the title - EXACTLY matching the movie pair function
                    st.markdown(
                        f"""
                        <div style="height: 5.5rem; display: flex; align-items: center;">
                            <h3 style="margin: 0; line-height: 1.3;">{movie_info['Title']}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Stats in a fixed height container
                    st.write(f"Predicted Rating: {pred_rating:.2f}")
                    st.write("")  # Add spacing to match the movie pair function's two lines of stats
                    
                    # Movie poster in fixed height container
                    try:
                        image_path = os.path.join("data", "MovieImages", f"{movie_num}.jpg")
                        if os.path.exists(image_path):
                            st.image(image_path, width=200)
                        else:
                            # Placeholder container for missing poster
                            st.markdown("*(No poster available)*")
                            st.write("")  # Add some spacing
                    except Exception:
                        st.markdown("*(Error loading poster)*")
                        st.write("")  # Add some spacing
                    
                    # Add extra spacing to compensate for missing slider
                    st.write("")
                    st.write("")
                
                # Add consistent spacing between movie pairs
                st.write("---")

def main():
    st.title("🎬 Movie Recommender System")
    
    # Load recommenders and data
    ibcf_recommender, pop_recommender = load_recommenders()
    movies_df = load_movie_data()
    
    # Get pool of popular movies
    movie_pool = get_movie_pool(pop_recommender)
    
    # Initialize session state
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'movies_to_show' not in st.session_state:
        st.session_state.movies_to_show = 10
        
    # Create tabs
    tab1, tab2 = st.tabs(["Rate Movies", "Get Recommendations"])
    
    with tab1:
        st.header("Rate These Popular Movies")
        
        # Display movies in aligned pairs
        for i in range(0, st.session_state.movies_to_show, 2):
            if i < len(movie_pool):
                display_movie_pair(movie_pool, i)
        
        # Add "Load More" button with reliable updating
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            # Only show button if there are more movies to load
            if st.session_state.movies_to_show < len(movie_pool):
                if st.button("Load More Movies", use_container_width=True, key="load_more"):
                    st.session_state.movies_to_show += 10
                    st.rerun()
    
    with tab2:
        if st.button("Get Recommendations", type="primary"):
            if len(st.session_state.user_ratings) < 2:
                st.warning("Please rate at least 2 movies to get recommendations.")
            else:
                st.header("Your Personalized Recommendations")
                
                # Create user vector
                user_vector = np.zeros(len(ibcf_recommender.ratings_matrix.columns))
                for movie_id, rating in st.session_state.user_ratings.items():
                    if movie_id in ibcf_recommender.ratings_matrix.columns:
                        idx = ibcf_recommender.column_to_index[movie_id]
                        user_vector[idx] = rating
                
                # Get recommendations
                recommendations = ibcf_recommender.myIBCF(user_vector)
                
                # Display recommendations in aligned pairs
                for i in range(0, len(recommendations), 2):
                    display_recommendation_pair(recommendations, movies_df, i)
        
        # Display current ratings
        if st.session_state.user_ratings:
            st.write("---")
            st.subheader("Your Current Ratings:")
            for movie_id, rating in st.session_state.user_ratings.items():
                movie_num = int(movie_id[1:])
                movie_info = movies_df[movies_df['MovieID'] == movie_num].iloc[0]
                st.write(f"{movie_info['Title']}: {rating} stars")

if __name__ == "__main__":
    main()
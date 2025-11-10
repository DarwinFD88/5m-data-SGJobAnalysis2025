import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Salary Benchmarking Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the job data"""
    try:
        df = pd.read_csv('//wsl.localhost/Ubuntu/home/darwin/NTU PACE Data Science Cohort 3/data/SGJobData.csv')
        
        # Convert job_skills from string to list if needed
        if 'job_skills' in df.columns:
            def parse_skills(x):
                if pd.isna(x):
                    return []
                if isinstance(x, list):
                    return x
                if isinstance(x, str):
                    if x.startswith('['):
                        try:
                            return ast.literal_eval(x)
                        except:
                            return []
                    elif ',' in x:
                        return [s.strip() for s in str(x).split(',') if s.strip() and s.strip() != 'No specific skills mentioned']
                return []
            
            df['job_skills'] = df['job_skills'].apply(parse_skills)
        elif 'job_skills_mapped' in df.columns:
            # If job_skills column doesn't exist, create it from job_skills_mapped
            df['job_skills'] = df['job_skills_mapped'].apply(
                lambda x: [s.strip() for s in str(x).split(',')] 
                if pd.notna(x) and str(x) != 'No specific skills mentioned' and str(x) != 'nan' else []
            )
        else:
            # If neither exists, create empty list
            df['job_skills'] = [[]] * len(df)
        
        # Filter out rows with missing salary data
        df = df[(df['salary_minimum'].notna()) & (df['salary_maximum'].notna())]
        df = df[(df['salary_minimum'] > 0) & (df['salary_maximum'] > 0)]
        
        # Calculate average salary
        df['salary_average'] = (df['salary_minimum'] + df['salary_maximum']) / 2
        
        # Filter for Monthly salary type (most common)
        df = df[df['salary_type'] == 'Monthly']
        
        # Remove outliers from salary figures using IQR method
        if len(df) > 0:
            # Remove outliers from average salary (primary metric)
            Q1 = df['salary_average'].quantile(0.25)
            Q3 = df['salary_average'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds (using 1.5 * IQR rule)
            lower_bound = max(0, Q1 - 1.5 * IQR)  # Ensure lower bound is not negative
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers from average salary
            df = df[(df['salary_average'] >= lower_bound) & (df['salary_average'] <= upper_bound)]
            
            # Additional sanity check: ensure minimum < maximum and both are reasonable
            df = df[df['salary_minimum'] < df['salary_maximum']]
            
            # Remove extreme outliers from min and max (using more lenient bounds)
            # This catches any remaining extreme values that might have passed through
            if len(df) > 0:
                min_lower = df['salary_minimum'].quantile(0.01)  # 1st percentile
                min_upper = df['salary_minimum'].quantile(0.99)  # 99th percentile
                max_lower = df['salary_maximum'].quantile(0.01)  # 1st percentile
                max_upper = df['salary_maximum'].quantile(0.99)  # 99th percentile
                
                df = df[
                    (df['salary_minimum'] >= min_lower) & (df['salary_minimum'] <= min_upper) &
                    (df['salary_maximum'] >= max_lower) & (df['salary_maximum'] <= max_upper)
                ]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def extract_role_from_title(title):
    """Extract a standardized role name from job title"""
    if pd.isna(title):
        return "Unknown"
    
    title_lower = str(title).lower()
    
    # Common role patterns
    role_keywords = {
        'engineer': ['engineer', 'engineering'],
        'developer': ['developer', 'dev'],
        'manager': ['manager', 'management'],
        'analyst': ['analyst', 'analysis'],
        'specialist': ['specialist'],
        'executive': ['executive'],
        'director': ['director'],
        'consultant': ['consultant', 'consulting'],
        'architect': ['architect'],
        'scientist': ['scientist'],
        'designer': ['designer', 'design'],
        'administrator': ['administrator', 'admin'],
        'coordinator': ['coordinator'],
        'assistant': ['assistant'],
        'technician': ['technician'],
        'officer': ['officer'],
        'lead': ['lead'],
        'senior': ['senior'],
        'junior': ['junior'],
    }
    
    # Check for specific roles first
    for role, keywords in role_keywords.items():
        for keyword in keywords:
            if keyword in title_lower:
                return role.capitalize()
    
    return "Other"

def main():
    st.markdown('<h1 class="main-header">💰 Salary Benchmarking Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### HR Consulting Firm - Market Salary Analysis")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("No data available. Please check the data file.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Position Level Filter
    position_levels = ['All'] + sorted([str(x) for x in df['positionLevels'].dropna().unique() if pd.notna(x)])
    selected_position = st.sidebar.selectbox("Position Level", position_levels)
    
    # Experience Filter
    min_exp = int(df['minimumYearsExperience'].min()) if df['minimumYearsExperience'].notna().any() else 0
    max_exp = int(df['minimumYearsExperience'].max()) if df['minimumYearsExperience'].notna().any() else 20
    exp_range = st.sidebar.slider("Minimum Years of Experience", min_exp, max_exp, (min_exp, max_exp))
    
    # Salary Range Filter
    min_salary = int(df['salary_minimum'].min())
    max_salary = int(df['salary_maximum'].max())
    salary_range = st.sidebar.slider("Salary Range (Monthly)", min_salary, max_salary, (min_salary, max_salary), step=500)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_position != 'All':
        filtered_df = filtered_df[filtered_df['positionLevels'] == selected_position]
    
    filtered_df = filtered_df[
        (filtered_df['minimumYearsExperience'] >= exp_range[0]) & 
        (filtered_df['minimumYearsExperience'] <= exp_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['salary_average'] >= salary_range[0]) & 
        (filtered_df['salary_average'] <= salary_range[1])
    ]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Job Postings", f"{len(filtered_df):,}")
    
    with col2:
        avg_salary = filtered_df['salary_average'].mean()
        st.metric("Average Salary", f"${avg_salary:,.0f}")
    
    with col3:
        median_salary = filtered_df['salary_average'].median()
        st.metric("Median Salary", f"${median_salary:,.0f}")
    
    with col4:
        unique_roles = filtered_df['title'].nunique()
        st.metric("Unique Roles", f"{unique_roles:,}")
    
    st.divider()
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 By Job Role", "🛠️ By Skills", "📈 Trends", "🔍 Detailed View"])
    
    with tab1:
        st.header("Salary Benchmarking by Job Role")
        
        # Extract standardized roles
        filtered_df['role_category'] = filtered_df['title'].apply(extract_role_from_title)
        
        # Role-based salary analysis
        role_salary = filtered_df.groupby('role_category').agg({
            'salary_average': ['mean', 'median', 'count', 'std'],
            'salary_minimum': 'mean',
            'salary_maximum': 'mean'
        }).round(0)
        
        role_salary.columns = ['Avg_Salary', 'Median_Salary', 'Count', 'Std_Dev', 'Avg_Min', 'Avg_Max']
        role_salary = role_salary.sort_values('Avg_Salary', ascending=False)
        role_salary = role_salary[role_salary['Count'] >= 10]  # Filter roles with at least 10 postings
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top roles by average salary
            st.subheader("Top Roles by Average Salary")
            top_roles = role_salary.head(15)
            
            fig = px.bar(
                top_roles.reset_index(),
                x='Avg_Salary',
                y='role_category',
                orientation='h',
                labels={'Avg_Salary': 'Average Salary (SGD)', 'role_category': 'Role Category'},
                title="Average Salary by Role",
                color='Avg_Salary',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Role salary distribution
            st.subheader("Salary Distribution by Role")
            selected_roles = st.multiselect(
                "Select roles to compare",
                options=role_salary.index.tolist(),
                default=role_salary.head(5).index.tolist()
            )
            
            if selected_roles:
                role_data = filtered_df[filtered_df['role_category'].isin(selected_roles)]
                fig = px.box(
                    role_data,
                    x='role_category',
                    y='salary_average',
                    labels={'salary_average': 'Average Salary (SGD)', 'role_category': 'Role Category'},
                    title="Salary Distribution Comparison"
                )
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Role salary table
        st.subheader("Detailed Role Salary Statistics")
        display_role_salary = role_salary.copy()
        display_role_salary['Avg_Salary'] = display_role_salary['Avg_Salary'].apply(lambda x: f"${x:,.0f}")
        display_role_salary['Median_Salary'] = display_role_salary['Median_Salary'].apply(lambda x: f"${x:,.0f}")
        display_role_salary['Avg_Min'] = display_role_salary['Avg_Min'].apply(lambda x: f"${x:,.0f}")
        display_role_salary['Avg_Max'] = display_role_salary['Avg_Max'].apply(lambda x: f"${x:,.0f}")
        display_role_salary['Std_Dev'] = display_role_salary['Std_Dev'].apply(lambda x: f"${x:,.0f}")
        display_role_salary = display_role_salary.rename(columns={
            'Count': 'Job Postings',
            'Avg_Salary': 'Avg Salary',
            'Median_Salary': 'Median Salary',
            'Std_Dev': 'Std Deviation',
            'Avg_Min': 'Avg Min Salary',
            'Avg_Max': 'Avg Max Salary'
        })
        st.dataframe(display_role_salary, use_container_width=True)
    
    with tab2:
        st.header("Salary Benchmarking by Skills")
        
        # Flatten skills and calculate salary by skill
        skill_salary_data = []
        
        for idx, row in filtered_df.iterrows():
            skills = row['job_skills'] if isinstance(row['job_skills'], list) else []
            if skills:
                for skill in skills:
                    skill_salary_data.append({
                        'skill': skill,
                        'salary_average': row['salary_average'],
                        'salary_minimum': row['salary_minimum'],
                        'salary_maximum': row['salary_maximum'],
                        'title': row['title']
                    })
        
        if skill_salary_data:
            skill_df = pd.DataFrame(skill_salary_data)
            
            # Skill-based salary analysis
            skill_salary = skill_df.groupby('skill').agg({
                'salary_average': ['mean', 'median', 'count', 'std'],
                'salary_minimum': 'mean',
                'salary_maximum': 'mean'
            }).round(0)
            
            skill_salary.columns = ['Avg_Salary', 'Median_Salary', 'Count', 'Std_Dev', 'Avg_Min', 'Avg_Max']
            skill_salary = skill_salary.sort_values('Avg_Salary', ascending=False)
            skill_salary = skill_salary[skill_salary['Count'] >= 50]  # Filter skills with at least 50 postings
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top skills by average salary
                st.subheader("Top Skills by Average Salary")
                top_skills = skill_salary.head(20)
                
                fig = px.bar(
                    top_skills.reset_index(),
                    x='Avg_Salary',
                    y='skill',
                    orientation='h',
                    labels={'Avg_Salary': 'Average Salary (SGD)', 'skill': 'Skill'},
                    title="Average Salary by Skill",
                    color='Avg_Salary',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Skill salary distribution
                st.subheader("Salary Distribution by Skill")
                selected_skills = st.multiselect(
                    "Select skills to compare",
                    options=skill_salary.index.tolist(),
                    default=skill_salary.head(5).index.tolist(),
                    key="skill_selector"
                )
                
                if selected_skills:
                    skill_data = skill_df[skill_df['skill'].isin(selected_skills)]
                    fig = px.box(
                        skill_data,
                        x='skill',
                        y='salary_average',
                        labels={'salary_average': 'Average Salary (SGD)', 'skill': 'Skill'},
                        title="Salary Distribution Comparison"
                    )
                    fig.update_layout(height=600, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Skill combination analysis
            st.subheader("Top Skill Combinations")
            
            # Find most common skill combinations
            skill_combinations = []
            for idx, row in filtered_df.iterrows():
                skills = row['job_skills'] if isinstance(row['job_skills'], list) else []
                if len(skills) >= 2:
                    skills_sorted = sorted(skills)
                    skill_combinations.append({
                        'combination': ', '.join(skills_sorted),
                        'salary_average': row['salary_average']
                    })
            
            if skill_combinations:
                combo_df = pd.DataFrame(skill_combinations)
                combo_stats = combo_df.groupby('combination').agg({
                    'salary_average': ['mean', 'count']
                }).round(0)
                combo_stats.columns = ['Avg_Salary', 'Count']
                combo_stats = combo_stats[combo_stats['Count'] >= 10].sort_values('Avg_Salary', ascending=False)
                
                top_combos = combo_stats.head(15).reset_index()
                fig = px.bar(
                    top_combos,
                    x='Avg_Salary',
                    y='combination',
                    orientation='h',
                    labels={'Avg_Salary': 'Average Salary (SGD)', 'combination': 'Skill Combination'},
                    title="Top Skill Combinations by Salary",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Skill salary table
            st.subheader("Detailed Skill Salary Statistics")
            display_skill_salary = skill_salary.copy()
            display_skill_salary['Avg_Salary'] = display_skill_salary['Avg_Salary'].apply(lambda x: f"${x:,.0f}")
            display_skill_salary['Median_Salary'] = display_skill_salary['Median_Salary'].apply(lambda x: f"${x:,.0f}")
            display_skill_salary['Avg_Min'] = display_skill_salary['Avg_Min'].apply(lambda x: f"${x:,.0f}")
            display_skill_salary['Avg_Max'] = display_skill_salary['Avg_Max'].apply(lambda x: f"${x:,.0f}")
            display_skill_salary['Std_Dev'] = display_skill_salary['Std_Dev'].apply(lambda x: f"${x:,.0f}")
            display_skill_salary = display_skill_salary.rename(columns={
                'Count': 'Job Postings',
                'Avg_Salary': 'Avg Salary',
                'Median_Salary': 'Median Salary',
                'Std_Dev': 'Std Deviation',
                'Avg_Min': 'Avg Min Salary',
                'Avg_Max': 'Avg Max Salary'
            })
            st.dataframe(display_skill_salary, use_container_width=True)
        else:
            st.warning("No skills data available. Please ensure the job_skills column exists in the dataset.")
    
    with tab3:
        st.header("Salary Trends & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary by experience level
            st.subheader("Salary by Experience Level")
            exp_salary = filtered_df.groupby('minimumYearsExperience').agg({
                'salary_average': ['mean', 'median', 'count']
            }).round(0)
            exp_salary.columns = ['Mean', 'Median', 'Count']
            exp_salary = exp_salary[exp_salary['Count'] >= 10]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=exp_salary.index,
                y=exp_salary['Mean'],
                mode='lines+markers',
                name='Mean Salary',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=exp_salary.index,
                y=exp_salary['Median'],
                mode='lines+markers',
                name='Median Salary',
                line=dict(color='green', width=3)
            ))
            fig.update_layout(
                title="Salary Growth by Years of Experience",
                xaxis_title="Years of Experience",
                yaxis_title="Salary (SGD)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salary distribution histogram
            st.subheader("Overall Salary Distribution")
            fig = px.histogram(
                filtered_df,
                x='salary_average',
                nbins=50,
                labels={'salary_average': 'Average Salary (SGD)', 'count': 'Number of Jobs'},
                title="Salary Distribution Histogram"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Position level analysis
        st.subheader("Salary by Position Level")
        if filtered_df['positionLevels'].notna().any():
            pos_salary = filtered_df.groupby('positionLevels').agg({
                'salary_average': ['mean', 'median', 'count']
            }).round(0)
            pos_salary.columns = ['Mean', 'Median', 'Count']
            pos_salary = pos_salary[pos_salary['Count'] >= 10].sort_values('Mean', ascending=False)
            
            fig = px.bar(
                pos_salary.reset_index(),
                x='positionLevels',
                y='Mean',
                labels={'Mean': 'Average Salary (SGD)', 'positionLevels': 'Position Level'},
                title="Average Salary by Position Level",
                color='Mean',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Detailed Job Postings View")
        
        # Search functionality
        search_term = st.text_input("Search job titles", "")
        
        # Display columns selector
        display_cols = st.multiselect(
            "Select columns to display",
            options=['title', 'salary_minimum', 'salary_maximum', 'salary_average', 
                    'positionLevels', 'minimumYearsExperience', 'job_skills_mapped', 
                    'postedCompany_name', 'employmentTypes'],
            default=['title', 'salary_minimum', 'salary_maximum', 'salary_average', 
                    'positionLevels', 'minimumYearsExperience', 'job_skills_mapped']
        )
        
        # Filter by search term
        display_df = filtered_df.copy()
        if search_term:
            display_df = display_df[display_df['title'].str.contains(search_term, case=False, na=False)]
        
        # Show data
        if display_cols:
            st.dataframe(
                display_df[display_cols].sort_values('salary_average', ascending=False),
                use_container_width=True,
                height=600
            )
        else:
            st.info("Please select at least one column to display.")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any
import json
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings  # if using langchain core
#from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from difflib import get_close_matches
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
import seaborn as sns


# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Smart Excel Data Assistant",
    page_icon="🤖",
    layout="wide"
)

# Configuration
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDINGS_DIR = "embd_cache"
VECTORSTORE_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index")
DATA_HASH_FILE = os.path.join(EMBEDDINGS_DIR, "data_hash.txt")


# Try multiple possible file locations
POSSIBLE_FILE_PATHS = [
    "main.xlsx",
    "dataset.xlsx",
    "dataset/dataset.xlsx", 
    "Database/dataset.xlsx",
    "excel/dataset.xlsx",
    "./dataset.xlsx",
    "dataset/main.xlsx"
]

# Try multiple possible sheet names
POSSIBLE_SHEET_NAMES = ["Data", "Sheet1", "data", "DATABASE", "Main"]

def find_excel_file():
    """Find the Excel file in possible locations"""
    for file_path in POSSIBLE_FILE_PATHS:
        if os.path.exists(file_path):
            return file_path
    return None

def find_sheet_name(file_path):
    """Find the correct sheet name in the Excel file"""
    try:
        excel_file = pd.ExcelFile(file_path)
        available_sheets = excel_file.sheet_names
        
        # Try preferred sheet names first
        for sheet_name in POSSIBLE_SHEET_NAMES:
            if sheet_name in available_sheets:
                return sheet_name, available_sheets
        
        # If no preferred sheet found, return first sheet
        return available_sheets[0], available_sheets
    except Exception as e:
        return None, []

import hashlib
import pickle

def get_data_hash(df):
    """Generate hash of the dataframe to detect changes"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()

def save_vectorstore_metadata(vectorstore, embeddings, data_insights, column_mappings, data_hash):
    """Save vectorstore and related metadata"""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Save FAISS index
    vectorstore.save_local(VECTORSTORE_PATH)
    
    # Save metadata
    metadata = {
        'data_insights': data_insights,
        'column_mappings': column_mappings,
        'data_hash': data_hash
    }
    
    with open(os.path.join(EMBEDDINGS_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save embeddings model info
    with open(os.path.join(EMBEDDINGS_DIR, "embeddings_info.pkl"), 'wb') as f:
        pickle.dump({'model_name': embeddings.model_name}, f)

def load_vectorstore_if_exists(embeddings):
    """Load existing vectorstore if available and data hasn't changed"""
    try:
        if not os.path.exists(VECTORSTORE_PATH):
            return None, None, None, None
        
        # Load metadata
        with open(os.path.join(EMBEDDINGS_DIR, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        # Load vectorstore
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        return (vectorstore, 
                metadata['data_insights'], 
                metadata['column_mappings'], 
                metadata['data_hash'])
    
    except Exception as e:
        st.warning(f"Could not load cached embeddings: {str(e)}")
        return None, None, None, None
    
    
def detect_chart_intent(query: str) -> str:
    query = query.lower()
    if "price" in query and "month" in query:
        return "price_vs_month_line"
    elif "heatmap" in query:
        return "category_heatmap"
    elif "distribution" in query and "category" in query:
        return "category_pie"
    elif "top brand" in query or "brand comparison" in query:
        return "brand_bar"
    elif "trend" in query:
        return "trend_line"
    else:
        return "unknown"

def plot_price_trend(df):
    if 'Month' in df.columns and 'Price' in df.columns:
        monthly_price = df.groupby('Month')['Price'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(data=monthly_price, x='Month', y='Price', marker='o', ax=ax)
        ax.set_title("Average Price Over Months")
        st.pyplot(fig)
    else:
        st.warning("Required columns 'Month' or 'Price' not found.")

def plot_category_heatmap(df):
    if 'Category' in df.columns and 'Retailer' in df.columns:
        pivot = df.pivot_table(index='Category', columns='Retailer', values='Price', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax)
        ax.set_title("Average Price Heatmap by Category & Retailer")
        st.pyplot(fig)
    else:
        st.warning("Required columns not found for heatmap.")

def plot_category_pie(df):
    if 'Category' in df.columns:
        top_cats = df['Category'].value_counts().head(5)
        fig, ax = plt.subplots()
        ax.pie(top_cats, labels=top_cats.index, autopct='%1.1f%%')
        ax.set_title("Top 5 Category Distribution")
        st.pyplot(fig)

def plot_brand_bar(df):
    if 'Brand' in df.columns:
        brand_counts = df['Brand'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=brand_counts.values, y=brand_counts.index, ax=ax)
        ax.set_title("Top 10 Brands by Count")
        st.pyplot(fig)

def plot_trend_generic(df):
    if 'Month' in df.columns and 'Discount Rate' in df.columns:
        trend = df.groupby('Month')['Discount Rate'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(data=trend, x='Month', y='Discount Rate', marker='o', ax=ax)
        ax.set_title("Discount Rate Trend Over Months")
        st.pyplot(fig)

class SmartExcelRAG:
    def __init__(self):
        self.df = None
        self.vectorstore = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.embeddings = None
        self.column_mappings = {}
        self.data_insights = {}
    
    def compare_discount_rate_by_brand(self, campaign_id: str, brands: list):
        if 'Brand' not in self.df.columns or 'Discount Rate' not in self.df.columns:
            return "Required columns missing for this analysis."
        
        campaign_data = self.df[self.df['Issue Id'].astype(str) == str(campaign_id)]
        if campaign_data.empty:
            return f"No data found for campaign ID {campaign_id}."
        
        filtered = campaign_data[campaign_data['Brand'].isin(brands)]
        if filtered.empty:
            return "No matching brands found in the specified campaign."
        summary = filtered.groupby("Brand")["Discount Rate"].agg(["count", "mean", "max", "min"]).reset_index()
        summary.columns = ["Brand", "Entries", "Average Discount %", "Max Discount %", "Min Discount %"]
        return summary.to_markdown(index=False)

    def load_and_process_data(self):
        if (hasattr(self, 'data_insights') and 
            self.data_insights and 
            hasattr(self, 'vectorstore') and 
            self.vectorstore):
            
            excel_file_path = find_excel_file()
            if excel_file_path:
                sheet_name, _ = find_sheet_name(excel_file_path)
                if sheet_name:
                    try:
                        usecols = ['Country', 'Retailer', 'Issue Start Date', 'Issue End Date', 'Brand', 'Category', 'Product', 'Size', 'Price', 'Discounted Price', 'Discount Rate', 'Main Flyer', 'Quarter', 'Month', 'Year', 'Week']
                        self.df = pd.read_excel(excel_file_path, sheet_name=sheet_name, usecols=usecols)
                        self.preprocess_data()
                        self.enhance_data_for_analysis()
                        st.success("✅ Full dataset loaded from existing cache")
                        return True
                    except Exception as e:
                        st.warning(f"Could not load from cache, reinitializing: {e}")
        try:
            excel_file_path = find_excel_file()
            if not excel_file_path:
                st.error("❌ Database file not found. Please ensure you have one of these files:")
                st.write("📁 Looking for files in these locations:")
                for path in POSSIBLE_FILE_PATHS:
                    st.write(f"  • {path}")
                
                st.markdown("---")
                st.subheader("📤 Upload Your Database File")
                uploaded_file = st.file_uploader(
                    "Upload your Excel database file",
                    type=['xlsx', 'xls'],
                    help="Upload your Excel file containing the data"
                    )
                
                if uploaded_file:
                    try:
                        excel_file = pd.ExcelFile(uploaded_file)
                        available_sheets = excel_file.sheet_names
                        st.write(f"📋 Available sheets: {', '.join(available_sheets)}")
                        
                        sheet_name = None
                        for preferred_sheet in POSSIBLE_SHEET_NAMES:
                            if preferred_sheet in available_sheets:
                                sheet_name = preferred_sheet
                                break
                        
                        if not sheet_name:
                            sheet_name = available_sheets[0]
                            
                        self.df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        st.success(f"✅ Data loaded from uploaded file, sheet: {sheet_name}")
                    except Exception as e:
                        st.error(f"❌ Error reading uploaded file: {str(e)}")
                        return False
                else:
                    return False
            else:
                sheet_name, available_sheets = find_sheet_name(excel_file_path)
                if not sheet_name:
                    st.error(f"❌ Could not read Excel file: {excel_file_path}")
                    return False
                
                usecols = ['Country', 'Retailer', 'Issue Start Date', 'Issue End Date', 'Brand', 'Category', 'Product', 'Size', 'Price', 'Discounted Price', 'Discount Rate', 'Main Flyer', 'Quarter', 'Month', 'Year', 'Week']
                self.df = pd.read_excel(excel_file_path, sheet_name=sheet_name, usecols=usecols)
                st.info(f"📁 File: {excel_file_path}")
                st.info(f"📋 Sheet: {sheet_name}")
                if len(available_sheets) > 1:
                    st.info(f"📋 Available sheets: {', '.join(available_sheets)}")
            
            if self.df.empty:
                st.error("❌ No data found in the Excel file")
                return False
            
            self.preprocess_data()
            self.enhance_data_for_analysis()
            self.create_data_insights()
            self.create_column_mappings()
            
            st.success(f"✅ Database loaded: {len(self.df):,} records with {len(self.df.columns)} columns")
            
            with st.expander("📊 Column Information"):
                st.write("**Columns in your database:**")
                for i, col in enumerate(self.df.columns, 1):
                    st.write(f"{i}. {col}")
            
            return True
        except FileNotFoundError:
            st.error(f"❌ Database file not found")
            return False
        except Exception as e:
            st.error(f"❌ Error loading database: {str(e)}")
            st.write("**Debug information:**")
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error details: {str(e)}")
            return False
        
    def quick_restore_from_cache(self):
        try:
            excel_file_path = find_excel_file()
            if not excel_file_path:
                return False
            
            sheet_name, _ = find_sheet_name(excel_file_path)
            if not sheet_name:
                return False
            self.df = pd.read_excel(excel_file_path, sheet_name=sheet_name, nrows=10)  # Just 10 rows for structure
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
                )
            
            cached_vectorstore, cached_insights, cached_mappings, cached_hash = load_vectorstore_if_exists(self.embeddings)
            if cached_vectorstore and cached_insights and cached_mappings:
                self.vectorstore = cached_vectorstore
                self.data_insights = cached_insights
                self.column_mappings = cached_mappings
                
                if self.setup_conversation_chain():
                    return True
            
            return False
        
        except Exception as e:
            st.error(f"Quick restore failed: {e}")
            return False
    
    def preprocess_data(self):
        """Advanced data preprocessing for 22k+ rows"""
        try:
            # Convert date columns
            date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'start' in col.lower() or 'end' in col.lower()]
            for col in date_columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = [col for col in self.df.columns if any(keyword in col.lower() for keyword in 
                             ['price', 'discount', 'rate', 'kg', 'size', 'quantity', 'span', 'amount', 'value'])]
            for col in numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Clean text columns
            text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
            for col in text_columns:
                if col not in date_columns:
                    self.df[col] = self.df[col].astype(str).str.strip()
                    self.df[col] = self.df[col].replace('nan', '')
            
            # Fill missing values strategically
            self.df.fillna({
                col: '' if self.df[col].dtype == 'object' else 0 
                for col in self.df.columns
            }, inplace=True)
            
            # Create additional analytical columns
            self.create_analytical_columns()
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
    
    def create_analytical_columns(self):
        """Create additional columns for better analysis"""
        try:
            # Extract month names and numbers from dates
            date_cols = [col for col in self.df.columns if 'date' in col.lower()]
            for col in date_cols:
                if not self.df[col].isna().all():
                    self.df[f'{col}_month'] = self.df[col].dt.month
                    self.df[f'{col}_month_name'] = self.df[col].dt.strftime('%B')
                    self.df[f'{col}_year'] = self.df[col].dt.year
                    self.df[f'{col}_day'] = self.df[col].dt.day
            
            # Create price per kg columns if not exists
            if 'price' in str(self.df.columns).lower() and 'kg' not in str(self.df.columns).lower():
                price_cols = [col for col in self.df.columns if 'price' in col.lower() and 'kg' not in col.lower()]
                size_cols = [col for col in self.df.columns if 'size' in col.lower() or 'weight' in col.lower()]
                
                if price_cols and size_cols:
                    for price_col in price_cols:
                        for size_col in size_cols:
                            try:
                                self.df[f'{price_col}_per_kg'] = self.df[price_col] / (self.df[size_col] / 1000)
                            except:
                                pass
            
        except Exception as e:
            st.warning(f"Could not create analytical columns: {str(e)}")
    
    def enhance_data_for_analysis(self):
        try:
            date_columns = ['Issue Start Date', 'Issue End Date']
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            if 'Quarter' not in self.df.columns and 'Issue Start Date' in self.df.columns:
                self.df['Quarter'] = self.df['Issue Start Date'].dt.to_period('Q').astype(str)
            if 'Main Flyer' in self.df.columns:
                self.df['Main Flyer'] = self.df['Main Flyer'].astype(str).str.strip()
            if 'Brand' in self.df.columns and 'Category' in self.df.columns:
                self.df['Brand_Category'] = self.df['Brand'].astype(str) + '_' + self.df['Category'].astype(str)
            if 'Price' in self.df.columns and 'Discounted Price' in self.df.columns:
                self.df['Actual_Discount_Amount'] = pd.to_numeric(self.df['Price'], errors='coerce') - pd.to_numeric(self.df['Discounted Price'], errors='coerce')
        except Exception as e:
            st.warning(f"Could not enhance data for analysis: {str(e)}")

    def create_data_insights(self):
        """Create comprehensive data insights for intelligent querying"""
        try:
            # Basic statistics
            self.data_insights = {
                'total_records': len(self.df),
                'date_range': {},
                'unique_counts': {},
                'top_values': {},
                'numerical_stats': {}
            }
            
            # Date range analysis
            date_cols = [col for col in self.df.columns if 'date' in col.lower()]
            for col in date_cols:
                if not self.df[col].isna().all():
                    self.data_insights['date_range'][col] = {
                        'min': self.df[col].min(),
                        'max': self.df[col].max()
                    }
            
            # Unique value counts for categorical columns
            categorical_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                if unique_count < 1000:  # Only store if manageable
                    self.data_insights['unique_counts'][col] = unique_count
                    self.data_insights['top_values'][col] = self.df[col].value_counts().head(10).to_dict()
            
            # Numerical statistics
            numeric_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
            for col in numeric_cols:
                self.data_insights['numerical_stats'][col] = {
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std()
                }
            
        except Exception as e:
            st.warning(f"Could not create data insights: {str(e)}")
    
    def create_column_mappings(self):
        try:
            patterns = {
                'brand': ['Brand', 'brand', 'K Brand'],
                'product': ['Product', 'Single Sku', 'sku', 'name'],
                'price': ['Price', 'Discounted Price', 'cost', 'amount'],
                'price_per_kg': ['Discounted Price Per (kg/lt/mt)', 'Price Per Kg', 'price/kg'],
                'discount': ['Discount Rate', 'discount', 'offer', 'promotion'],
                'date': ['Issue Start Date', 'Issue End Date', 'date', 'time'],
                'country': ['Country', 'region', 'location'],
                'retailer': ['Retailer', 'store', 'shop', 'outlet', 'account'],
                'category': ['Category', 'K Category', 'Sub Category', 'type', 'class'],
                'size': ['Size', 'Size Packed', 'weight', 'volume', 'quantity'],
                'month': ['Month', 'Month (MMM)', 'period'],
                'quarter': ['Quarter', 'Q1', 'Q2', 'Q3', 'Q4'],
                'flyer': ['Main Flyer', 'flyer', 'leaflet']
                }
            
            for concept, keywords in patterns.items():
                matching_cols = []
                for col in self.df.columns:
                    if any(keyword.lower() in col.lower() for keyword in keywords):
                        matching_cols.append(col)
                        self.column_mappings[concept] = matching_cols
        except Exception as e:
            st.warning(f"Could not create column mappings: {str(e)}")
    
    def create_enhanced_documents(self):
        documents = []
        summary_content = self.create_comprehensive_summary()
        documents.append(Document(
            page_content=summary_content,
            metadata={'type': 'summary', 'importance': 'high'}
            ))
        
        analytical_docs = self.create_analytical_documents()
        documents.extend(analytical_docs)
        
        sample_docs = self.create_strategic_samples()
        documents.extend(sample_docs)
        return documents
    
    def create_comprehensive_summary(self):
        summary_parts = []
        summary_parts.append(f"DATASET: {self.data_insights['total_records']} records covering retail flyer data")
        for col, date_info in self.data_insights['date_range'].items():
            summary_parts.append(f"{col}: {date_info['min']} to {date_info['max']}")
            
        if 'Brand' in self.df.columns:
            brand_stats = self.df.groupby('Brand').agg({
                'Price': ['count', 'mean', 'min', 'max'],
                'Discount Rate': 'mean',
                'Main Flyer': lambda x: (x == 'Yes').sum()
            }).round(2)
            
            for brand in brand_stats.index[:10]:  # Top 10 brands
                brand_info = brand_stats.loc[brand]
                summary_parts.append(
                    f"BRAND_{brand}: {brand_info[('Price', 'count')]} records, "
                    f"avg_price_{brand_info[('Price', 'mean')]}, "
                    f"flyer_appearances_{brand_info[('Main Flyer', '<lambda>')]}"
                    )
                
        if 'Category' in self.df.columns:
            category_counts = self.df['Category'].value_counts().head(10)
            category_str = ' | '.join([f"{cat}({count})" for cat, count in category_counts.items()])
            summary_parts.append(f"TOP_CATEGORIES: {category_str}")
        if 'Country' in self.df.columns:
            country_counts = self.df['Country'].value_counts().head(5)
            country_str = ' | '.join([f"{country}({count})" for country, count in country_counts.items()])
            summary_parts.append(f"COUNTRIES: {country_str}")
            
        return " || ".join(summary_parts)
    
    def create_analytical_documents(self):
        
        documents = []
        try:
            if 'Brand' in self.df.columns and 'Price' in self.df.columns:
                brand_comparison = self.df.groupby('Brand').agg({
                    'Price': ['count', 'mean', 'std', 'min', 'max'],
                    'Discount Rate': ['mean', 'std'],
                    'Main Flyer': lambda x: (x == 'Yes').sum(),
                    'Country': 'nunique',
                    'Retailer': 'nunique'
                }).round(2)
                
                comparison_content = "BRAND_ANALYSIS: "
                for brand in brand_comparison.index:
                    stats = brand_comparison.loc[brand]
                    comparison_content += f"{brand}_stats: records={stats[('Price', 'count')]}, avg_price={stats[('Price', 'mean')]}, flyer_count={stats[('Main Flyer', '<lambda>')]}, countries={stats[('Country', 'nunique')]} || "
                    
                documents.append(Document(
                    page_content=comparison_content,
                    metadata={'type': 'brand_analysis'}
                    ))
                
            if 'Month' in self.df.columns:
                monthly_trends = self.df.groupby('Month').agg({
                    'Price': 'mean',
                    'Discount Rate': 'mean',
                    'Brand': 'nunique'
                }).round(2)
                
                trend_content = "MONTHLY_TRENDS: "
                for month in monthly_trends.index:
                    stats = monthly_trends.loc[month]
                    trend_content += f"{month}: avg_price={stats['Price']}, avg_discount={stats['Discount Rate']}, brands={stats['Brand']} || "
                    
                documents.append(Document(
                    page_content=trend_content,
                    metadata={'type': 'monthly_trends'}
                    ))
                
            if 'Category' in self.df.columns and 'Brand' in self.df.columns:
                cat_brand_matrix = self.df.groupby(['Category', 'Brand']).agg({
                    'Price': ['count', 'mean'],
                    'Main Flyer': lambda x: (x == 'Yes').sum()
                }).round(2)
                
                matrix_content = "CATEGORY_BRAND_MATRIX: "
                for (category, brand), stats in cat_brand_matrix.iterrows():
                    matrix_content += f"{category}_{brand}: count={stats[('Price', 'count')]}, avg_price={stats[('Price', 'mean')]}, flyers={stats[('Main Flyer', '<lambda>')]} || "
                    
                documents.append(Document(
                    page_content=matrix_content,
                    metadata={'type': 'category_brand_analysis'}
                    ))
        except Exception as e:
            print(f"Warning: Could not create some analytical documents: {e}")
        return documents
    
    def create_strategic_samples(self):
        documents = []
        try:
            if 'Price' in self.df.columns:
                high_price_sample = self.df.nlargest(100, 'Price')
                high_price_content = self.create_sample_content(high_price_sample, "HIGH_PRICE_RECORDS")
                documents.append(Document(
                    page_content=high_price_content,
                    metadata={'type': 'high_price_sample'}
                    ))
                
            if 'Brand' in self.df.columns:
                top_brands = self.df['Brand'].value_counts().head(5).index
                for brand in top_brands:
                    brand_sample = self.df[self.df['Brand'] == brand].sample(min(50, len(self.df[self.df['Brand'] == brand])))
                    brand_content = self.create_sample_content(brand_sample, f"BRAND_{brand}_SAMPLE")
                    documents.append(Document(
                        page_content=brand_content,
                        metadata={'type': 'brand_sample', 'brand': brand}
                        ))
            
            if 'Issue Start Date' in self.df.columns:
                recent_sample = self.df.nlargest(100, 'Issue Start Date')
                recent_content = self.create_sample_content(recent_sample, "RECENT_RECORDS")
                documents.append(Document(
                page_content=recent_content,
                metadata={'type': 'recent_sample'}
                ))
        except Exception as e:
            print(f"Warning: Could not create sample documents: {e}")
            
        return documents
    
    def create_sample_content(self, sample_df, prefix):
        content_parts = [prefix]
        for idx, row in sample_df.iterrows():
            row_parts = []
            for col in ['Brand', 'Product', 'Category', 'Price', 'Discounted Price', 'Country', 'Month', 'Main Flyer']:
                if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                    row_parts.append(f"{col}:{row[col]}")
            
            if row_parts:
                content_parts.append("RECORD: " + " | ".join(row_parts))
        return " || ".join(content_parts)

    def create_summary_document(self):
        """Create comprehensive summary document"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"DATABASE OVERVIEW: {self.data_insights['total_records']} total records")
        
        # Date ranges
        for col, date_info in self.data_insights['date_range'].items():
            summary_parts.append(f"{col}: {date_info['min']} to {date_info['max']}")
        
        # Key categories
        for col, top_values in self.data_insights['top_values'].items():
            if len(top_values) > 0:
                values_str = ', '.join([f"{k}({v})" for k, v in list(top_values.items())[:5]])
                summary_parts.append(f"{col} TOP VALUES: {values_str}")
        
        # Numerical ranges
        for col, stats in self.data_insights['numerical_stats'].items():
            summary_parts.append(f"{col} RANGE: {stats['min']:.2f} to {stats['max']:.2f}, AVG: {stats['mean']:.2f}")
        
        return " | ".join(summary_parts)
    
    def create_categorical_documents(self):
        """Create documents for categorical analysis"""
        documents = []
        
        # Brand analysis
        if 'brand' in self.column_mappings:
            for brand_col in self.column_mappings['brand']:
                brand_analysis = self.df.groupby(brand_col).agg({
                    col: ['count', 'mean', 'min', 'max'] for col in self.df.columns 
                    if self.df[col].dtype in ['int64', 'float64']
                }).round(2)
                
                if not brand_analysis.empty:
                    content = f"BRAND ANALYSIS for {brand_col}: " + str(brand_analysis.to_dict())
                    documents.append(Document(
                        page_content=content,
                        metadata={'type': 'brand_analysis', 'column': brand_col}
                    ))
        
        return documents
    
    def create_batch_document(self, batch_df, start_idx):
        """Create document for a batch of records"""
        content_parts = []
        
        for idx, row in batch_df.iterrows():
            row_parts = [f"RECORD_{idx}"]
            
            # Include all columns but prioritize important ones
            important_cols = []
            for concept, cols in self.column_mappings.items():
                important_cols.extend(cols)
            
            # Add important columns first
            for col in important_cols:
                if col in row.index:
                    value = str(row[col]).strip()
                    if value and value != '0' and value != 'nan':
                        row_parts.append(f"{col}:{value}")
            
            # Add remaining columns
            for col in batch_df.columns:
                if col not in important_cols and col in row.index:
                    value = str(row[col]).strip()
                    if value and value != '0' and value != 'nan':
                        row_parts.append(f"{col}:{value}")
            
            content_parts.append(" | ".join(row_parts))
        
        return Document(
            page_content=" || ".join(content_parts),
            metadata={'type': 'batch', 'start_idx': start_idx, 'end_idx': start_idx + len(batch_df)}
        )
    
    def setup_vectorstore(self):
        try:
            current_hash = get_data_hash(self.df)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
                )
            
            with st.spinner("📥 Checking for cached embeddings..."):
                cached_vectorstore, cached_insights, cached_mappings, cached_hash = load_vectorstore_if_exists(self.embeddings)
                
            if (cached_vectorstore is not None and 
                cached_hash == current_hash and 
                cached_insights is not None and 
                cached_mappings is not None):
                
                st.success("✅ Using cached embeddings - faster startup!")
                self.vectorstore = cached_vectorstore
                self.data_insights = cached_insights
                self.column_mappings = cached_mappings
                return True
            
            with st.spinner(f"🔄 Processing {len(self.df):,} records (this will take a moment)..."):
                st.info("💡 This is a one-time process. Future sessions will be much faster!")
                documents = self.create_enhanced_documents()
                if not documents:
                    st.error("No documents created")
                    return False
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,  # Larger chunks for better context
                    chunk_overlap=300,
                    length_function=len,
                    separators=[" || ", " | ", "\n", " ", ""]
                )
                
                split_docs = text_splitter.split_documents(documents)
                
                self.vectorstore = FAISS.from_documents(
                    split_docs, 
                    self.embeddings,
                    distance_strategy="COSINE"
                    )
                save_vectorstore_metadata(
                    self.vectorstore, 
                    self.embeddings, 
                    self.data_insights, 
                    self.column_mappings, 
                    current_hash
                    )
                
                st.success(f"✅ Processed {len(self.df):,} records with {len(split_docs)} optimized document chunks")
                return True
            
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
            return False
    
    def setup_conversation_chain(self):
        """Setup optimized conversation chain"""
        try:
            # Initialize Groq LLM with better parameters
            llm = ChatGoogleGenerativeAI(   
                google_api_key=GEMINI_API_KEY,
                model="gemini-2.5-flash",  #gemini-pro
                temperature=0.1,
                max_output_tokens=4000
            )
            
            # Enhanced prompt template
            prompt_template = """
            You are an expert data analyst specializing in retail flyer and promotional data analysis.
            You have access to a comprehensive database with {total_records} records containing detailed flyer participation data.
            DATASET STRUCTURE:
            - Columns: Country, Retailer, Issue Id, Issue Start Date, Issue End Date, Issue Span, Week, Month, Quarter, Category, Brand, Product, Single Sku, Size, Price, Discounted Price, Discount Rate, Main Flyer (Yes/No), etc.
            - Main focus: Flyer participation analysis, brand comparisons, promotional effectiveness
            - Key entities: Brands (LAY'S, PARADISE, PRINGLES, NESTLE, etc.), Countries (Bahrain, UAE, KSA, etc.), Retailers, Categories (Chips, Kids Cereals, etc.)
            - Time periods: Q1-Q4 2022, monthly breakdowns 
            - Flyerdata: Main Flyer participation (Yes/No), promotional periods, discount rates
            
            ANALYTICAL CAPABILITIES:
            1. Flyer Participation Share: Calculate percentage of flyer appearances for brands
            2. Brand Comparisons: Compare performance metrics between brands
            3. Time-based Analysis: Quarter-over-quarter, month-over-month trends
            4. Category Analysis: Performance within specific product categories
            5. Geographic Analysis: Country/retailer-specific insights
            6. Promotional Effectiveness: Discount rates, pricing strategies
            
            RESPONSE REQUIREMENTS:
            - Provide specific numbers and percentages
            - Include time periods and category filters used
            - Show calculation methodology when relevant
            - Use structured format with clear headings
            - Include comparative analysis when multiple brands mentioned
            
            ANALYSIS GUIDELINES:
            1. For date queries: Use exact date matching and ranges
            2. For pricing queries: Consider price, discounted price, price/kg
            3. For brand comparisons: Show detailed comparisons with metrics
            4. For geographic queries: Filter by country/region accurately
            5. For frequency analysis: Provide month-over-month trends
            6. Always show relevant columns: dates, prices, SKU names, brands
            
            RESPONSE FORMAT:
            - Start with direct answer
            - Include specific data points and numbers
            - Show relevant columns as requested
            - Provide insights and patterns
            - Use tables or structured format when appropriate
            
            Your output must:
            - Start with a 1-line conclusion
            - Use tables for comparative data
            - Only include data from valid rows (skip missing or zeroed values)
            
            Context from database: {context}
            
            Previous conversation: {chat_history}
            
            User Query: {question}
            
            Detailed Analysis:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"],
                partial_variables={"total_records": self.data_insights['total_records']}
            )
            
            # Create conversation chain with better retrieval
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}  # Retrieve more documents for better context
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                verbose=False
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up conversation chain: {str(e)}")
            return False
    
    def enhanced_query_processing(self, question: str):
        try:
            if self.is_analytical_query(question):
                direct_answer = self.process_analytical_query(question)
                if direct_answer:
                    return direct_answer, []
                
                enhanced_question = self.add_query_context(question)
                if not self.conversation_chain:
                    return "❌ Error: Conversation chain not initialized.", []
                response = self.conversation_chain({
                    "question": enhanced_question
                    })
                
                return response["answer"], response.get("source_documents", [])
            if not self.conversation_chain:
                return "❌ Error: Conversation chain not initialized.", []
            response = self.conversation_chain({
                "question": question
                })
            return response["answer"], response.get("source_documents", [])
        
        except Exception as e:
            return f"Error processing query: {str(e)}", []
    

    def add_query_context(self, question: str):
        """Add intelligent context to improve query understanding"""
        context_parts = []
        
        # Add column information based on query keywords
        question_lower = question.lower()
        
        if any(brand in question_lower for brand in ['pringles', 'lays', 'brand']):
            brand_cols = self.column_mappings.get('brand', [])
            context_parts.append(f"Brand columns available: {', '.join(brand_cols)}")
        
        if any(date_word in question_lower for date_word in ['date', 'started', 'between', 'month']):
            date_cols = [col for col in self.df.columns if 'date' in col.lower()]
            context_parts.append(f"Date columns available: {', '.join(date_cols)}")
        
        if any(price_word in question_lower for price_word in ['price', 'cost', 'kg']):
            price_cols = [col for col in self.df.columns if 'price' in col.lower()]
            context_parts.append(f"Price columns available: {', '.join(price_cols)}")
        
        if any(geo_word in question_lower for geo_word in ['uae', 'ksa', 'country', 'region']):
            geo_cols = self.column_mappings.get('location', [])
            context_parts.append(f"Location columns available: {', '.join(geo_cols)}")
        
        # Add context to question
        if context_parts:
            enhanced_question = f"{question}\n\nContext: {' | '.join(context_parts)}"
        else:
            enhanced_question = question
        
        return enhanced_question
    
    def is_analytical_query(self, question: str) -> bool:
        keywords = [
        "average", "compare", "most", "highest", "share", "distribution",
        "how many", "top", "lowest", "trend", "total", "frequency", "count", "appearances"
        ]
        return any(word in question.lower() for word in keywords)
    
    def perform_brand_comparison(self, question: str):
        try:
            brands = self.extract_brands_from_query(question)
            if len(brands) < 2:
                return "Please specify at least two brands to compare."
            
            comparison_df = self.df.copy()
            
            if 'Product' in comparison_df.columns:
                if 'corn flakes' in question.lower():
                    comparison_df = comparison_df[comparison_df['Product'].str.contains('corn flakes', case=False, na=False)]
            
            size_col = None
            for col in self.df.columns:
                if 'size' in col.lower() and self.df[col].dtype in ['int64', 'float64']:
                    size_col = col
                    break
            if not size_col:
                return "Size column not found in the dataset."
            
            results = []
            for brand in brands:
                brand_df = comparison_df[comparison_df['Brand'].str.contains(brand, case=False, na=False)]
                if not brand_df.empty:
                    avg_size = brand_df[size_col].mean()
                    max_size = brand_df[size_col].max()
                    min_size = brand_df[size_col].min()
                    results.append((brand, avg_size, min_size, max_size, len(brand_df)))
                    
            if not results:
                return "No records found for the specified brands."
            
            response = "**Brand Size Comparison**\n\n"
            for brand, avg_size, min_size, max_size, count in results:
                response += f"**{brand}**\n"
                response += f"- Average Size: {avg_size:.2f} g\n"
                response += f"- Min Size: {min_size:.2f} g\n"
                response += f"- Max Size: {max_size:.2f} g\n"
                response += f"- Records Considered: {count}\n\n"
                
            results.sort(key=lambda x: x[1], reverse=True)
            response += "**📊 Ranking by Average Size:**\n"
            for i, (brand, avg_size, *_rest) in enumerate(results, 1):
                response += f"{i}. {brand} — {avg_size:.2f} g\n"
            
            return response
        except Exception as e:
            return f"Error comparing brands: {str(e)}"


    def process_analytical_query(self, question: str):
        try:
            question_lower = question.lower()
            if 'show all' in question_lower or 'leaflets' in question_lower:
                return self.show_leaflets_with_filters(question)
            elif 'price/kg' in question_lower or 'price per kg' in question_lower:
                return self.analyze_price_per_kg(question)
            elif 'brand wise frequency' in question_lower or 'month on month' in question_lower:
                return self.analyze_brand_frequency(question)
            elif 'participation share' in question_lower or 'share of flier' in question_lower:
                return self.calculate_flyer_participation_share(question)
            elif any(word in question_lower for word in ['vs', 'versus', 'compare']):
                brands = self.extract_brands_from_query(question)
                if len(brands) >= 2:
                    campaign_id = self.extract_campaign_id(question)
                    if campaign_id:
                        return self.compare_discount_rate_by_brand(campaign_id, brands)
                    else:
                        filtered = self.df[self.df['Brand'].isin(brands)]
                        if 'Discount Rate' in filtered.columns and filtered['Discount Rate'].notna().any():
                            summary = filtered.groupby("Brand")["Discount Rate"].agg(["count", "mean", "max", "min"]).reset_index()
                            summary.columns = ["Brand", "Entries", "Average Discount %", "Max Discount %", "Min Discount %"]
                            return summary.to_markdown(index=False)
                        else:
                            return self.perform_brand_comparison(question)
                else:
                    return "Please specify at least two brands to compare."

            return None
        
        except Exception as e:
            return f"Error in analytical processing: {str(e)}"
        
    def extract_campaign_id(self, question: str):
        match = re.search(r'\bcampaign\s*(\d+)', question.lower())
        return match.group(1) if match else None

        
    def show_leaflets_with_filters(self, question: str):
        try:
            brands = self.extract_brands_from_query(question)
            country = self.extract_country_from_query(question)
            date_range = self.extract_date_range_from_query(question)
            filtered_df = self.df.copy()
            
            if brands:
                brand_filter = filtered_df['Brand'].str.contains('|'.join(brands), case=False, na=False)
                filtered_df = filtered_df[brand_filter]
            if country:
                filtered_df = filtered_df[filtered_df['Country'].str.contains(country, case=False, na=False)]
            if date_range:
                filtered_df = self.apply_date_range_filter(filtered_df, date_range)
            
            display_columns = ['Issue Start Date', 'Issue End Date', 'Product', 'Price', 'Discounted Price', 'Brand', 'Country', 'Retailer']
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            
            if len(filtered_df) > 0:
                result_df = filtered_df[available_columns].copy()
                
                for date_col in ['Issue Start Date', 'Issue End Date']:
                    if date_col in result_df.columns:
                        result_df[date_col] = pd.to_datetime(result_df[date_col], errors='coerce').dt.strftime('%d-%m-%Y')
                response = f"**Leaflets Found: {len(result_df)} records**\n\n"
                
                if brands:
                    response += f"**Brands:** {', '.join(brands)}\n"
                if country:
                    response += f"**Country:** {country}\n"
                if date_range:
                    response += f"**Date Range:** {date_range}\n"
                
                response += "\n**Results:**\n\n"
                
                for idx, row in result_df.head(20).iterrows():
                    response += f"**{idx+1}.** {row.get('Brand', 'N/A')} - {row.get('Product', 'N/A')}\n"
                    response += f"   Start: {row.get('Issue Start Date', 'N/A')} | End: {row.get('Issue End Date', 'N/A')}\n"
                    response += f"   Price: {row.get('Price', 'N/A')} | Discounted: {row.get('Discounted Price', 'N/A')}\n"
                    response += f"   Retailer: {row.get('Retailer', 'N/A')}\n\n"
                    
                if len(result_df) > 20:
                    response += f"*... and {len(result_df) - 20} more records*"
                    
                return response
            else:
                return "No leaflets found matching your criteria."
            
        except Exception as e:
            return f"Error showing leaflets: {str(e)}"
    
    def analyze_price_per_kg(self, question: str):
        try:
            brands = self.extract_brands_from_query(question)
            threshold = self.extract_numeric_threshold(question)
            price_per_kg_col = None
            for col in self.df.columns:
                if 'price' in col.lower() and 'kg' in col.lower():
                    price_per_kg_col = col
                    break
                
                if not price_per_kg_col:
                    if 'Discounted Price Per (kg/lt/mt)' in self.df.columns:
                        price_per_kg_col = 'Discounted Price Per (kg/lt/mt)'
                    elif 'Price' in self.df.columns and 'Size' in self.df.columns:
                        self.df['Calculated_Price_Per_Kg'] = pd.to_numeric(self.df['Price'], errors='coerce') / (pd.to_numeric(self.df['Size'], errors='coerce') / 1000)
                        price_per_kg_col = 'Calculated_Price_Per_Kg'
                if not price_per_kg_col:
                    return "Price per kg data not available in the dataset."
                
                filtered_df = self.df.copy()
                
                if brands:
                    brand_filter = filtered_df['Brand'].str.contains('|'.join(brands), case=False, na=False)
                    filtered_df = filtered_df[brand_filter]
                
                if threshold:
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[price_per_kg_col], errors='coerce') > threshold]
                    
                if len(filtered_df) > 0:
                    brand_analysis = {}
                    
                    for brand in brands:
                        brand_data = filtered_df[filtered_df['Brand'].str.contains(brand, case=False, na=False)]
                        if len(brand_data) > 0:
                            brand_analysis[brand] = {
                                'count': len(brand_data),
                                'avg_price_per_kg': brand_data[price_per_kg_col].mean(),
                                'max_price_per_kg': brand_data[price_per_kg_col].max(),
                                'min_price_per_kg': brand_data[price_per_kg_col].min()
                                }
                            
                    response = f"**Price/Kg Analysis (>{threshold} threshold)**\n\n"
                    
                    for brand, stats in brand_analysis.items():
                        response += f"**{brand}:**\n"
                        response += f"- Records: {stats['count']}\n"
                        response += f"- Average Price/Kg: {stats['avg_price_per_kg']:.2f}\n"
                        response += f"- Max Price/Kg: {stats['max_price_per_kg']:.2f}\n"
                        response += f"- Min Price/Kg: {stats['min_price_per_kg']:.2f}\n\n"
                    
                    return response
                else:
                    return f"No records found with price/kg > {threshold} for the specified brands."
        except Exception as e:
            return f"Error analyzing price per kg: {str(e)}"
        
    def analyze_brand_frequency(self, question: str):
        try:
            country = self.extract_country_from_query(question)
            portfolio = 'Kids' if 'kids' in question.lower() else None
            filtered_df = self.df.copy()
            
            if country:
                filtered_df = filtered_df[filtered_df['Country'].str.contains(country, case=False, na=False)]
            if portfolio:
                portfolio_filter = (
                    filtered_df['Category'].str.contains('Kids', case=False, na=False) |
                    filtered_df['K Category'].str.contains('Kids', case=False, na=False) |
                    filtered_df['Sub Category'].str.contains('Kids', case=False, na=False)
                    )
                
                filtered_df = filtered_df[portfolio_filter]
            
            if 'Month' in filtered_df.columns and 'Brand' in filtered_df.columns:
                frequency_analysis = filtered_df.groupby(['Brand', 'Month']).size().reset_index(name='Frequency')
                pivot_table = frequency_analysis.pivot(index='Brand', columns='Month', values='Frequency').fillna(0)
                response = f"**Brand-wise Frequency Analysis - {portfolio or 'All'} Portfolio**\n\n"
                
                if country:
                    response += f"**Country:** {country}\n"
                response += "**Month-on-Month Frequency:**\n\n"
                
                for brand in pivot_table.index:
                    response += f"**{brand}:**\n"
                    for month in pivot_table.columns:
                        freq = pivot_table.loc[brand, month]
                        if freq > 0:
                            response += f"  {month}: {int(freq)} leaflets\n"
                    response += "\n"
                
                if 'top 5 accounts' in question.lower():
                    top_accounts = filtered_df['Retailer'].value_counts().head(5)
                    response += "**Top 5 Accounts:**\n"
                    for i, (account, count) in enumerate(top_accounts.items(), 1):
                        response += f"{i}. {account}: {count} leaflets\n"
                        
                return response
            else:
                return "Month or Brand columns not found in the dataset."
        except Exception as e:
            return f"Error analyzing brand frequency: {str(e)}"
    
    def extract_country_from_query(self, question: str):
        countries = ['UAE', 'KSA', 'Saudi Arabia', 'Bahrain', 'Kuwait', 'Qatar', 'Oman']
        question_upper = question.upper()
        
        for country in countries:
            if country.upper() in question_upper:
                return country
            
            return None
    
    def extract_date_range_from_query(self, question: str):
        import re
        between_match = re.search(r'between\s+(\d+)[a-z]*\s+and\s+(\d+)[a-z]*', question.lower())
        
        if between_match:
            start_day = between_match.group(1)
            end_day = between_match.group(2)
            return f"between {start_day} and {end_day}"
        
        return None
    
    def apply_date_range_filter(self, df, date_range):
        try:
            if 'between' in date_range:
                import re
                days = re.findall(r'\d+', date_range)
                if len(days) >= 2:
                    start_day = int(days[0])
                    end_day = int(days[1])
                    
                    if 'Issue Start Date' in df.columns:
                        df['start_day'] = pd.to_datetime(df['Issue Start Date'], errors='coerce').dt.day
                        df = df[(df['start_day'] >= start_day) | (df['start_day'] <= end_day)]
                        df = df.drop('start_day', axis=1)
            return df
        except Exception as e:
            return df
    def extract_numeric_threshold(self, question: str):
        import re
        
        more_than_match = re.search(r'more than\s+(\d+(?:\.\d+)?)', question.lower())
        if more_than_match:
            return float(more_than_match.group(1))
        
        lower_than_match = re.search(r'lower than\s+(\d+(?:\.\d+)?)', question.lower())
        if lower_than_match:
            return float(lower_than_match.group(1))
        
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        if numbers:
            return float(numbers[0])
        
        return None
        
    def calculate_flyer_participation_share(self, question: str):
        try:
            brands = self.extract_brands_from_query(question)
            time_period = self.extract_time_period_from_query(question)
            category = self.extract_category_from_query(question)
            if not brands:
                return "Please specify the brands you want to compare (e.g., Lay's vs Paradise)"
            
            filtered_df = self.df.copy()
            if time_period:
                filtered_df = self.apply_time_filter(filtered_df, time_period)
            if category:
                filtered_df = filtered_df[filtered_df['Category'].str.contains(category, case=False, na=False)]
            results = []
            total_records = len(filtered_df)
            
            for brand in brands:
                brand_data = filtered_df[filtered_df['Brand'].str.contains(brand, case=False, na=False)]
                
                if len(brand_data) > 0:
                    flyer_appearances = len(brand_data[brand_data['Main Flyer'] == 'Yes'])
                    total_brand_records = len(brand_data)
                    participation_rate = (flyer_appearances / total_brand_records) * 100 if total_brand_records > 0 else 0
                    market_share = (total_brand_records / total_records) * 100 if total_records > 0 else 0
                    
                    results.append({
                    'Brand': brand,
                    'Total Records': total_brand_records,
                    'Flyer Appearances': flyer_appearances,
                    'Participation Rate': f"{participation_rate:.1f}%",
                    'Market Share': f"{market_share:.1f}%"
                    })
            if results:
                response = f"**Flyer Participation Analysis**\n\n"
                
                if time_period:
                    response += f"**Time Period:** {time_period}\n"
                    if category:
                        response += f"**Category:** {category}\n"
                    response += f"**Total Records Analyzed:** {total_records:,}\n\n"
                    
                    for result in results:
                        response += f"**{result['Brand']}:**\n"
                        response += f"- Total Records: {result['Total Records']:,}\n"
                        response += f"- Flyer Appearances: {result['Flyer Appearances']:,}\n"
                        response += f"- Participation Rate: {result['Participation Rate']}\n"
                        response += f"- Market Share: {result['Market Share']}\n\n"
                    
                    if len(results) > 1:
                        response += "**Comparison:**\n"
                        for i, result in enumerate(results):
                            response += f"{i+1}. {result['Brand']}: {result['Participation Rate']} participation rate\n"
                    return response
                else:
                    return "No data found for the specified brands and criteria."
        except Exception as e:
            return f"Error calculating flyer participation: {str(e)}"
    
    def extract_brands_from_query(self, question: str):
        if 'Brand' not in self.df.columns:
            return []
        
        unique_brands = [str(b).strip().upper() for b in self.df['Brand'].dropna().unique()]
        question_upper = question.upper()
        found_brands = set()
        
        for brand in unique_brands:
            if brand in question_upper:
                found_brands.add(brand)
                
        brand_variations = {
            "LAY'S": ["LAYS", "LAY'S", "LAY"],
            "PARADISE": ["PARADISE"],
            "PRINGLES": ["PRINGLES"],
            "NESTLE": ["NESTLÉ", "NESTLE", "NESTL"],
            "KELLOGG'S": ["KELLOGGS", "KELLOGG'S", "KELLOG"],
            }
        
        for standard_brand, aliases in brand_variations.items():
            for alias in aliases:
                if alias in question_upper:
                    close_match = get_close_matches(standard_brand, unique_brands, n=1, cutoff=0.7)
                    if close_match:
                        found_brands.add(close_match[0])
                    elif standard_brand in unique_brands:
                        found_brands.add(standard_brand)
                    break
        
        return list(found_brands)  # ✅ moved here

    
    def extract_time_period_from_query(self, question: str):
        import re
        quarter_match = re.search(r'Q[1-4]\s*20\d{2}', question, re.IGNORECASE)
        
        if quarter_match:
            return quarter_match.group()
        
        months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
        
        for month in months:
            if month in question.lower():
                year_match = re.search(r'20\d{2}', question)
                if year_match:
                    return f"{month.capitalize()} {year_match.group()}"
        return None
    
    def extract_category_from_query(self, question: str):
        unique_categories = self.df['Category'].unique() if 'Category' in self.df.columns else []
        question_lower = question.lower()
        
        for category in unique_categories:
            if category and str(category).lower() in question_lower:
                return str(category)
            
        return None
    
    def apply_time_filter(self, df, time_period):
        try:
            if time_period.startswith('Q'):
                return df[df['Quarter'] == time_period] if 'Quarter' in df.columns else df
            elif any(month in time_period.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                return df[df['Month'].str.contains(time_period, case=False, na=False)] if 'Month' in df.columns else df
            
            return df
        
        except Exception as e:
            return df

def validate_cache_integrity():
        try:
            if not os.path.exists(EMBEDDINGS_DIR):
                return False
            required_files = ["metadata.pkl", "embeddings_info.pkl"]
            faiss_files = ["index.faiss", "index.pkl"]
            
            for file in required_files:
                if not os.path.exists(os.path.join(EMBEDDINGS_DIR, file)):
                    return False
            
            faiss_path = os.path.join(EMBEDDINGS_DIR, "faiss_index")
            for file in faiss_files:
                if not os.path.exists(os.path.join(faiss_path, file)):
                    return False
            
            return True
        except Exception:
            return False
            
def main():
    st.title("🤖 Smart Excel Data Assistant")
    st.markdown("*Intelligent query system for comprehensive data analysis*")
    
    # Initialize the system with persistent check
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SmartExcelRAG()
        st.session_state.system_ready = False
        st.session_state.chat_history = []
        st.session_state.data_loaded = False
        st.session_state.embeddings_loaded = False

    # Quick check if system was already ready but session restarted
    if not st.session_state.system_ready and os.path.exists(EMBEDDINGS_DIR):
        # Try to quickly restore from cache without full reinitialization
        try:
            with st.spinner("🔄 Restoring from cache..."):
                if st.session_state.rag_system.quick_restore_from_cache():
                    st.session_state.system_ready = True
        except Exception as e:
            st.warning(f"Cache restore failed: {e}. Will reinitialize...")
    
    # Check if system is ready, if not try to initialize
    if not st.session_state.system_ready:
        
        # Show current working directory for debugging
        with st.expander("🔍 Debug Information"):
            st.write(f"**Current working directory:** {os.getcwd()}")
            st.write("**Files in current directory:**")
            files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
            if files:
                for file in files:
                    st.write(f"  📄 {file}")
            else:
                st.write("  No Excel files found in current directory")
        
        # Try to initialize
        with st.spinner("🚀 Initializing Smart Data Assistant..."):
            if st.session_state.rag_system.load_and_process_data():
                # Only proceed if data is loaded
                if st.session_state.rag_system.df is not None:
                    if (st.session_state.rag_system.setup_vectorstore() and 
                        st.session_state.rag_system.setup_conversation_chain()):
                        st.session_state.system_ready = True
                        st.success("✅ System ready! Ask me anything about your data.")
                    else:
                        st.error("❌ Failed to initialize the vector store or conversation chain")
                else:
                    st.warning("⚠️ Please upload your database file above to continue")
    
    # Main interface
    if st.session_state.system_ready:
        
        # Chat interface
        st.subheader("💬 Ask Your Questions")
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Show Data Summary"):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "Give me a comprehensive summary of this database"
                })
        with col2:
            if st.button("📊 Top Brands Analysis"):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "What are the top brands in this database and their key metrics?"
                })
        with col3:
            if st.button("📅 Date Range Analysis"):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "What is the date range coverage and key time periods in this data?"
                })
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Process any queued messages
        # Process only one unprocessed user message
        for idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user" and not message.get("processed", False):    
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Analyzing your query..."):
                        response, sources = st.session_state.rag_system.enhanced_query_processing(message["content"])
                        st.write(response)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                            })
                        st.session_state.chat_history[idx]["processed"] = True
                        
                        if sources:
                            with st.expander("📋 Source Information"):
                                for i, source in enumerate(sources[:3]):
                                    st.write(f"**Source {i+1}:** {source.page_content[:200]}...")
                                    if source.metadata:
                                        st.write(f"*Metadata: {source.metadata}*")
                                    st.divider()
                    break  # Only process one message

            
            
        # Chat input
        user_query = st.chat_input("Ask me anything about your data...")
        
        if user_query:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.rerun()  # Rerun to process the message
        
        # Clear chat option
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.rag_system.memory.clear()
            st.rerun()
    
    else:
        st.info("📁 Please ensure your Excel database file is available or upload it above to start using the assistant.")

if __name__ == "__main__":
    main()

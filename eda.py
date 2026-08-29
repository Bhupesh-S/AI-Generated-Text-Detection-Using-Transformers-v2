"""
Exploratory Data Analysis (EDA) for AI-Generated Text Detection
Final Year Project: Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text Using Ensemble Learning

This script performs comprehensive EDA on the final dataset and generates detailed reports
suitable for IEEE/Scopus research papers and Final Year Project reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import textstat
from pathlib import Path
import logging
import warnings
from tqdm import tqdm
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/eda/eda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class EDAAnalyzer:
    """
    A comprehensive EDA analyzer for AI-generated text detection dataset.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the EDA analyzer.
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.images_dir = Path('reports/eda/images')
        self.tables_dir = Path('reports/eda/tables')
        self.report_dir = Path('reports/eda')
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("EDA Analyzer initialized")
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        try:
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully with {len(self.df)} rows")
            return self.df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def dataset_summary(self) -> dict:
        """
        Generate comprehensive dataset summary statistics.
        
        Returns:
            dict: Dictionary containing dataset summary statistics
        """
        logger.info("Generating dataset summary")
        
        summary = {
            'Number of Rows': len(self.df),
            'Number of Columns': len(self.df.columns),
            'Dataset Shape': self.df.shape,
            'Column Names': list(self.df.columns),
            'Data Types': self.df.dtypes.to_dict(),
            'Missing Values': self.df.isnull().sum().to_dict(),
            'Duplicate Values': self.df.duplicated().sum(),
            'Memory Usage (MB)': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Save summary to CSV
        summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
        summary_df.to_csv(self.tables_dir / 'dataset_summary.csv')
        logger.info("Dataset summary saved to CSV")
        
        return summary
    
    def class_distribution(self) -> dict:
        """
        Analyze class distribution in the dataset.
        
        Returns:
            dict: Dictionary containing class distribution statistics
        """
        logger.info("Analyzing class distribution")
        
        label_counts = self.df['label'].value_counts()
        label_percentages = self.df['label'].value_counts(normalize=True) * 100
        
        distribution = {
            'Human Written (0)': {
                'Count': label_counts.get(0, 0),
                'Percentage': label_percentages.get(0, 0)
            },
            'AI Generated (1)': {
                'Count': label_counts.get(1, 0),
                'Percentage': label_percentages.get(1, 0)
            }
        }
        
        # Save to CSV
        dist_df = pd.DataFrame.from_dict(distribution, orient='index')
        dist_df.to_csv(self.tables_dir / 'class_distribution.csv')
        
        # Create visualizations
        self._plot_class_distribution(label_counts, label_percentages)
        
        logger.info("Class distribution analysis completed")
        return distribution
    
    def _plot_class_distribution(self, counts: pd.Series, percentages: pd.Series):
        """
        Create class distribution visualizations.
        
        Args:
            counts (pd.Series): Label counts
            percentages (pd.Series): Label percentages
        """
        # Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ['Human Written (0)', 'AI Generated (1)']
        values = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Class Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution Bar Chart', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'class_distribution_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pie Chart
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#2E86AB', '#A23B72']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(
            [counts.get(0, 0), counts.get(1, 0)],
            labels=labels,
            colors=colors,
            explode=explode,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2}
        )
        
        ax.set_title('Class Distribution Pie Chart', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'class_distribution_pie.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Class distribution visualizations saved")
    
    def text_statistics(self) -> dict:
        """
        Calculate comprehensive text statistics.
        
        Returns:
            dict: Dictionary containing text statistics
        """
        logger.info("Calculating text statistics")
        
        # Calculate text lengths
        self.df['text_length'] = self.df['text'].astype(str).apply(len)
        self.df['word_count'] = self.df['text'].astype(str).apply(lambda x: len(x.split()))
        self.df['sentence_count'] = self.df['text'].astype(str).apply(lambda x: len(nltk.sent_tokenize(x)))
        self.df['avg_word_length'] = self.df['text'].astype(str).apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        stats = {
            'Text Length': {
                'Mean': self.df['text_length'].mean(),
                'Median': self.df['text_length'].median(),
                'Std': self.df['text_length'].std(),
                'Min': self.df['text_length'].min(),
                'Max': self.df['text_length'].max(),
                '25th Percentile': self.df['text_length'].quantile(0.25),
                '50th Percentile': self.df['text_length'].quantile(0.50),
                '75th Percentile': self.df['text_length'].quantile(0.75),
                '90th Percentile': self.df['text_length'].quantile(0.90),
                '95th Percentile': self.df['text_length'].quantile(0.95)
            },
            'Word Count': {
                'Mean': self.df['word_count'].mean(),
                'Median': self.df['word_count'].median(),
                'Std': self.df['word_count'].std(),
                'Min': self.df['word_count'].min(),
                'Max': self.df['word_count'].max(),
                '25th Percentile': self.df['word_count'].quantile(0.25),
                '50th Percentile': self.df['word_count'].quantile(0.50),
                '75th Percentile': self.df['word_count'].quantile(0.75),
                '90th Percentile': self.df['word_count'].quantile(0.90),
                '95th Percentile': self.df['word_count'].quantile(0.95)
            },
            'Sentence Count': {
                'Mean': self.df['sentence_count'].mean(),
                'Median': self.df['sentence_count'].median(),
                'Std': self.df['sentence_count'].std(),
                'Min': self.df['sentence_count'].min(),
                'Max': self.df['sentence_count'].max()
            },
            'Average Word Length': {
                'Mean': self.df['avg_word_length'].mean(),
                'Median': self.df['avg_word_length'].median(),
                'Std': self.df['avg_word_length'].std()
            }
        }
        
        # Save to CSV
        stats_df = pd.DataFrame.from_dict({(k, v): stats[k][v] 
                                          for k in stats.keys() 
                                          for v in stats[k].keys()}, 
                                         orient='index', 
                                         columns=['Value'])
        stats_df.to_csv(self.tables_dir / 'text_statistics.csv')
        
        logger.info("Text statistics calculated and saved")
        return stats
    
    def plot_histograms(self):
        """
        Create comprehensive histogram visualizations for text features.
        """
        logger.info("Creating histogram visualizations")
        
        # Text Length Distribution Histogram
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.df['text_length'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Text Length Distribution Histogram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'text_length_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Box Plot of Text Length
        fig, ax = plt.subplots(figsize=(10, 6))
        boxprops = dict(linestyle='-', linewidth=2, color='#2E86AB')
        medianprops = dict(linestyle='-', linewidth=2, color='red')
        whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')
        capprops = dict(linestyle='-', linewidth=1.5, color='black')
        
        bp = ax.boxplot(self.df['text_length'], 
                       vert=True,
                       patch_artist=True,
                       boxprops=boxprops,
                       medianprops=medianprops,
                       whiskerprops=whiskerprops,
                       capprops=capprops)
        
        ax.set_ylabel('Text Length (characters)', fontsize=12, fontweight='bold')
        ax.set_title('Box Plot of Text Length', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'text_length_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Word Count Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.df['word_count'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Word Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Character Count Distribution (by class)
        fig, ax = plt.subplots(figsize=(12, 6))
        human_lengths = self.df[self.df['label'] == 0]['text_length']
        ai_lengths = self.df[self.df['label'] == 1]['text_length']
        
        ax.hist(human_lengths, bins=50, alpha=0.6, label='Human Written', color='#2E86AB', edgecolor='black')
        ax.hist(ai_lengths, bins=50, alpha=0.6, label='AI Generated', color='#A23B72', edgecolor='black')
        ax.set_xlabel('Character Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Character Count Distribution by Class', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'character_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sentence Count Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.df['sentence_count'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sentence Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Sentence Count Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'sentence_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Average Word Length Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.df['avg_word_length'], bins=50, color='#C73E1D', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Average Word Length', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Average Word Length Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'avg_word_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Histogram visualizations completed")
    
    def generate_wordcloud(self):
        """
        Generate word clouds for Human and AI generated text.
        """
        logger.info("Generating word clouds")
        
        # Sample text by class to avoid memory issues
        sample_size = 10000
        
        human_sample = self.df[self.df["label"] == 0].sample(
            n=min(sample_size, len(self.df[self.df["label"] == 0])),
            random_state=42
        )
        
        ai_sample = self.df[self.df["label"] == 1].sample(
            n=min(sample_size, len(self.df[self.df["label"] == 1])),
            random_state=42
        )
        
        human_text = " ".join(human_sample["text"].astype(str))
        ai_text = " ".join(ai_sample["text"].astype(str))
        
        # Generate word cloud for Human text
        wordcloud_human = WordCloud(
            width=1600, height=800,
            background_color='white',
            colormap='Blues',
            max_words=200,
            relative_scaling=0.5,
            collocations=False
        ).generate(human_text)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud_human, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud - Human Written Text', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'wordcloud_human.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate word cloud for AI text
        wordcloud_ai = WordCloud(
            width=1600, height=800,
            background_color='white',
            colormap='Reds',
            max_words=200,
            relative_scaling=0.5,
            collocations=False
        ).generate(ai_text)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud_ai, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud - AI Generated Text', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'wordcloud_ai.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Top 30 Most Frequent Words
        self._plot_top_words(human_text, ai_text)
        
        logger.info("Word clouds generated successfully")
    
    def _plot_top_words(self, human_text: str, ai_text: str):
        """
        Plot top 30 most frequent words for each class.
        
        Args:
            human_text (str): Combined human text
            ai_text (str): Combined AI text
        """
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and count words for Human
        human_words = [word.lower() for word in word_tokenize(human_text) 
                      if word.isalpha() and word.lower() not in stop_words]
        human_word_counts = Counter(human_words).most_common(30)
        
        # Tokenize and count words for AI
        ai_words = [word.lower() for word in word_tokenize(ai_text) 
                   if word.isalpha() and word.lower() not in stop_words]
        ai_word_counts = Counter(ai_words).most_common(30)
        
        # Plot Top 30 Words - Human
        fig, ax = plt.subplots(figsize=(14, 8))
        words, counts = zip(*human_word_counts)
        y_pos = np.arange(len(words))
        
        ax.barh(y_pos, counts, color='#2E86AB', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Top 30 Most Frequent Words - Human Written Text', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'top_words_human.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Top 30 Words - AI
        fig, ax = plt.subplots(figsize=(14, 8))
        words, counts = zip(*ai_word_counts)
        y_pos = np.arange(len(words))
        
        ax.barh(y_pos, counts, color='#A23B72', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Top 30 Most Frequent Words - AI Generated Text', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'top_words_ai.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save to CSV
        human_df = pd.DataFrame(human_word_counts, columns=['Word', 'Frequency'])
        human_df.to_csv(self.tables_dir / 'top_words_human.csv', index=False)
        
        ai_df = pd.DataFrame(ai_word_counts, columns=['Word', 'Frequency'])
        ai_df.to_csv(self.tables_dir / 'top_words_ai.csv', index=False)
    
    def readability_metrics(self) -> dict:
        """
        Calculate readability metrics for the dataset.
        
        Returns:
            dict: Dictionary containing readability metrics
        """
        logger.info("Calculating readability metrics")
        
        # Sample a subset for readability calculation (due to computational cost)
        sample_size = min(10000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        metrics = {
            'Flesch Reading Ease': [],
            'Flesch-Kincaid Grade': [],
            'Gunning Fog Index': [],
            'SMOG Index': [],
            'Automated Readability Index': [],
            'Coleman-Liau Index': []
        }
        
        logger.info(f"Calculating readability for {sample_size} samples...")
        for text in tqdm(sample_df['text'].astype(str), desc="Readability Metrics"):
            try:
                metrics['Flesch Reading Ease'].append(textstat.flesch_reading_ease(text))
                metrics['Flesch-Kincaid Grade'].append(textstat.flesch_kincaid_grade(text))
                metrics['Gunning Fog Index'].append(textstat.gunning_fog(text))
                metrics['SMOG Index'].append(textstat.smog_index(text))
                metrics['Automated Readability Index'].append(textstat.automated_readability_index(text))
                metrics['Coleman-Liau Index'].append(textstat.coleman_liau_index(text))
            except:
                for key in metrics:
                    metrics[key].append(0)
        
        # Calculate statistics
        readability_stats = {}
        for metric, values in metrics.items():
            readability_stats[metric] = {
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            }
        
        # Save to CSV
        readability_df = pd.DataFrame.from_dict(
            {(k, v): readability_stats[k][v] 
             for k in readability_stats.keys() 
             for v in readability_stats[k].keys()},
            orient='index',
            columns=['Value']
        )
        readability_df.to_csv(self.tables_dir / 'readability_metrics.csv')
        
        # Create visualization
        self._plot_readability_metrics(readability_stats)
        
        logger.info("Readability metrics calculated and saved")
        return readability_stats
    
    def _plot_readability_metrics(self, stats: dict):
        """
        Create visualization for readability metrics.
        
        Args:
            stats (dict): Readability statistics
        """
        metrics = list(stats.keys())
        means = [stats[m]['Mean'] for m in metrics]
        stds = [stats[m]['Std'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x_pos = np.arange(len(metrics))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color='#2E86AB', alpha=0.8, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Readability Metrics Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'readability_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def lexical_diversity(self) -> dict:
        """
        Calculate lexical diversity metrics.
        
        Returns:
            dict: Dictionary containing lexical diversity metrics
        """
        logger.info("Calculating lexical diversity metrics")
        
        # Sample for computational efficiency
        sample_size = min(10000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        diversity_metrics = {
            'Vocabulary Size': [],
            'Unique Words': [],
            'Type-Token Ratio (TTR)': [],
            'Lexical Diversity': []
        }
        
        logger.info(f"Calculating lexical diversity for {sample_size} samples...")
        for text in tqdm(sample_df['text'].astype(str), desc="Lexical Diversity"):
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha()]
            
            if len(words) > 0:
                unique_words = len(set(words))
                total_words = len(words)
                ttr = unique_words / total_words if total_words > 0 else 0
                
                diversity_metrics['Vocabulary Size'].append(len(set(words)))
                diversity_metrics['Unique Words'].append(unique_words)
                diversity_metrics['Type-Token Ratio (TTR)'].append(ttr)
                diversity_metrics['Lexical Diversity'].append(unique_words / np.sqrt(total_words) if total_words > 0 else 0)
            else:
                for key in diversity_metrics:
                    diversity_metrics[key].append(0)
        
        # Calculate statistics
        diversity_stats = {}
        for metric, values in diversity_metrics.items():
            diversity_stats[metric] = {
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            }
        
        # Save to CSV
        diversity_df = pd.DataFrame.from_dict(
            {(k, v): diversity_stats[k][v] 
             for k in diversity_stats.keys() 
             for v in diversity_stats[k].keys()},
            orient='index',
            columns=['Value']
        )
        diversity_df.to_csv(self.tables_dir / 'lexical_diversity.csv')
        
        # Create visualization
        self._plot_lexical_diversity(diversity_stats)
        
        logger.info("Lexical diversity metrics calculated and saved")
        return diversity_stats
    
    def _plot_lexical_diversity(self, stats: dict):
        """
        Create visualization for lexical diversity metrics.
        
        Args:
            stats (dict): Lexical diversity statistics
        """
        metrics = list(stats.keys())
        means = [stats[m]['Mean'] for m in metrics]
        stds = [stats[m]['Std'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(metrics))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color='#A23B72', alpha=0.8, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Lexical Diversity Metrics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'lexical_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def nlp_statistics(self) -> dict:
        """
        Calculate comprehensive NLP statistics.
        
        Returns:
            dict: Dictionary containing NLP statistics
        """
        logger.info("Calculating NLP statistics")
        
        # Sample for computational efficiency
        sample_size = min(10000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        nlp_stats = {
            'Token Count': [],
            'Unique Words': [],
            'Vocabulary Richness': [],
            'Type Token Ratio (TTR)': [],
            'Average Sentence Length': [],
            'Stopword Percentage': [],
            'Punctuation Percentage': [],
            'Uppercase Ratio': [],
            'Digit Ratio': []
        }
        
        stop_words = set(stopwords.words('english'))
        
        logger.info(f"Calculating NLP statistics for {sample_size} samples...")
        for text in tqdm(sample_df['text'].astype(str), desc="NLP Statistics"):
            tokens = word_tokenize(text)
            words = [w for w in tokens if w.isalpha()]
            
            if len(tokens) > 0:
                # Token count
                nlp_stats['Token Count'].append(len(tokens))
                
                # Unique words
                unique_words = len(set([w.lower() for w in words]))
                nlp_stats['Unique Words'].append(unique_words)
                
                # Vocabulary richness
                nlp_stats['Vocabulary Richness'].append(unique_words / len(words) if len(words) > 0 else 0)
                
                # TTR
                nlp_stats['Type Token Ratio (TTR)'].append(unique_words / len(tokens) if len(tokens) > 0 else 0)
                
                # Average sentence length
                sentences = sent_tokenize(text)
                avg_sent_len = len(words) / len(sentences) if len(sentences) > 0 else 0
                nlp_stats['Average Sentence Length'].append(avg_sent_len)
                
                # Stopword percentage
                stopwords_count = sum(1 for w in words if w.lower() in stop_words)
                nlp_stats['Stopword Percentage'].append(stopwords_count / len(words) if len(words) > 0 else 0)
                
                # Punctuation percentage
                punct_count = sum(1 for c in text if c in '.,!?;:"\'-()[]{}')
                nlp_stats['Punctuation Percentage'].append(punct_count / len(text) if len(text) > 0 else 0)
                
                # Uppercase ratio
                uppercase_count = sum(1 for c in text if c.isupper())
                nlp_stats['Uppercase Ratio'].append(uppercase_count / len(text) if len(text) > 0 else 0)
                
                # Digit ratio
                digit_count = sum(1 for c in text if c.isdigit())
                nlp_stats['Digit Ratio'].append(digit_count / len(text) if len(text) > 0 else 0)
            else:
                for key in nlp_stats:
                    nlp_stats[key].append(0)
        
        # Calculate statistics
        nlp_summary = {}
        for metric, values in nlp_stats.items():
            nlp_summary[metric] = {
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            }
        
        # Save to CSV
        nlp_df = pd.DataFrame.from_dict(
            {(k, v): nlp_summary[k][v] 
             for k in nlp_summary.keys() 
             for v in nlp_summary[k].keys()},
            orient='index',
            columns=['Value']
        )
        nlp_df.to_csv(self.tables_dir / 'nlp_statistics.csv')
        
        # Create visualization
        self._plot_nlp_statistics(nlp_summary)
        
        logger.info("NLP statistics calculated and saved")
        return nlp_summary
    
    def _plot_nlp_statistics(self, stats: dict):
        """
        Create visualization for NLP statistics.
        
        Args:
            stats (dict): NLP statistics
        """
        metrics = list(stats.keys())
        means = [stats[m]['Mean'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x_pos = np.arange(len(metrics))
        
        bars = ax.bar(x_pos, means, color='#F18F01', alpha=0.8, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
        ax.set_title('NLP Statistics Overview', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'nlp_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_human_ai(self) -> dict:
        """
        Compare Human vs AI writing characteristics.
        
        Returns:
            dict: Dictionary containing comparison metrics
        """
        logger.info("Comparing Human vs AI writing characteristics")
        
        # Sample for computational efficiency
        sample_size = min(5000, len(self.df[self.df['label'] == 0]), len(self.df[self.df['label'] == 1]))
        human_sample = self.df[self.df['label'] == 0].sample(n=sample_size, random_state=42)
        ai_sample = self.df[self.df['label'] == 1].sample(n=sample_size, random_state=42)
        
        comparison = {
            'Human': {},
            'AI': {}
        }
        
        # Calculate metrics for each class
        for label, sample in [('Human', human_sample), ('AI', ai_sample)]:
            comparison[label]['Average Sentence Length'] = sample['text'].astype(str).apply(
                lambda x: len(word_tokenize(x)) / max(1, len(sent_tokenize(x)))
            ).mean()
            
            comparison[label]['Average Word Length'] = sample['avg_word_length'].mean()
            comparison[label]['Vocabulary Diversity'] = sample['text'].astype(str).apply(
                lambda x: len(set(word_tokenize(x.lower()))) / max(1, len(word_tokenize(x)))
            ).mean()
            
            # Readability
            readability_scores = []
            for text in tqdm(sample['text'].astype(str), desc=f"Readability ({label})", leave=False):
                try:
                    readability_scores.append(textstat.flesch_reading_ease(text))
                except:
                    readability_scores.append(0)
            comparison[label]['Readability (Flesch)'] = np.mean(readability_scores)
        
        # Save to CSV
        comparison_df = pd.DataFrame.from_dict(comparison, orient='index')
        comparison_df.to_csv(self.tables_dir / 'human_ai_comparison.csv')
        
        # Create visualization
        self._plot_human_ai_comparison(comparison)
        
        logger.info("Human vs AI comparison completed")
        return comparison
    
    def _plot_human_ai_comparison(self, comparison: dict):
        """
        Create visualization for Human vs AI comparison.
        
        Args:
            comparison (dict): Comparison metrics
        """
        metrics = list(comparison['Human'].keys())
        human_values = [comparison['Human'][m] for m in metrics]
        ai_values = [comparison['AI'][m] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, human_values, width, label='Human Written', 
                      color='#2E86AB', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, ai_values, width, label='AI Generated', 
                      color='#A23B72', alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Human vs AI Writing Characteristics Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'human_ai_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, summary: dict, distribution: dict, stats: dict, 
                   readability: dict, diversity: dict, nlp_stats: dict, 
                   comparison: dict):
        """
        Generate comprehensive markdown and PDF reports.
        
        Args:
            summary (dict): Dataset summary
            distribution (dict): Class distribution
            stats (dict): Text statistics
            readability (dict): Readability metrics
            diversity (dict): Lexical diversity
            nlp_stats (dict): NLP statistics
            comparison (dict): Human vs AI comparison
        """
        logger.info("Generating comprehensive EDA report")
        
        report_content = f"""# Exploratory Data Analysis Report
## Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text Using Ensemble Learning

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Dataset Overview

### 1.1 Basic Information

- **Number of Rows:** {summary['Number of Rows']:,}
- **Number of Columns:** {summary['Number of Columns']}
- **Dataset Shape:** {summary['Dataset Shape']}
- **Memory Usage:** {summary['Memory Usage (MB)']:.2f} MB

### 1.2 Column Information

**Column Names:** {', '.join(summary['Column Names'])}

**Data Types:**
"""
        for col, dtype in summary['Data Types'].items():
            report_content += f"- {col}: {dtype}\n"
        
        report_content += f"""
### 1.3 Data Quality

**Missing Values:**
"""
        for col, count in summary['Missing Values'].items():
            report_content += f"- {col}: {count}\n"
        
        report_content += f"""
**Duplicate Values:** {summary['Duplicate Values']:,}

---

## 2. Class Distribution

"""
        for label, data in distribution.items():
            report_content += f"- **{label}:** {data['Count']:,} samples ({data['Percentage']:.2f}%)\n"
        
        report_content += f"""
![Class Distribution Bar Chart](images/class_distribution_bar.png)

![Class Distribution Pie Chart](images/class_distribution_pie.png)

---

## 3. Text Statistics

### 3.1 Text Length Statistics

| Metric | Value |
|--------|-------|
| Mean | {stats['Text Length']['Mean']:.2f} |
| Median | {stats['Text Length']['Median']:.2f} |
| Standard Deviation | {stats['Text Length']['Std']:.2f} |
| Minimum | {stats['Text Length']['Min']:.2f} |
| Maximum | {stats['Text Length']['Max']:.2f} |
| 25th Percentile | {stats['Text Length']['25th Percentile']:.2f} |
| 50th Percentile | {stats['Text Length']['50th Percentile']:.2f} |
| 75th Percentile | {stats['Text Length']['75th Percentile']:.2f} |
| 90th Percentile | {stats['Text Length']['90th Percentile']:.2f} |
| 95th Percentile | {stats['Text Length']['95th Percentile']:.2f} |

### 3.2 Word Count Statistics

| Metric | Value |
|--------|-------|
| Mean | {stats['Word Count']['Mean']:.2f} |
| Median | {stats['Word Count']['Median']:.2f} |
| Standard Deviation | {stats['Word Count']['Std']:.2f} |
| Minimum | {stats['Word Count']['Min']:.2f} |
| Maximum | {stats['Word Count']['Max']:.2f} |
| 25th Percentile | {stats['Word Count']['25th Percentile']:.2f} |
| 50th Percentile | {stats['Word Count']['50th Percentile']:.2f} |
| 75th Percentile | {stats['Word Count']['75th Percentile']:.2f} |
| 90th Percentile | {stats['Word Count']['90th Percentile']:.2f} |
| 95th Percentile | {stats['Word Count']['95th Percentile']:.2f} |

### 3.3 Sentence Count Statistics

| Metric | Value |
|--------|-------|
| Mean | {stats['Sentence Count']['Mean']:.2f} |
| Median | {stats['Sentence Count']['Median']:.2f} |
| Standard Deviation | {stats['Sentence Count']['Std']:.2f} |
| Minimum | {stats['Sentence Count']['Min']:.2f} |
| Maximum | {stats['Sentence Count']['Max']:.2f} |

### 3.4 Average Word Length Statistics

| Metric | Value |
|--------|-------|
| Mean | {stats['Average Word Length']['Mean']:.2f} |
| Median | {stats['Average Word Length']['Median']:.2f} |
| Standard Deviation | {stats['Average Word Length']['Std']:.2f} |

![Text Length Distribution Histogram](images/text_length_histogram.png)

![Box Plot of Text Length](images/text_length_boxplot.png)

![Word Count Distribution](images/word_count_distribution.png)

![Character Count Distribution](images/character_count_distribution.png)

![Sentence Count Distribution](images/sentence_count_distribution.png)

![Average Word Length Distribution](images/avg_word_length_distribution.png)

---

## 4. Word Frequency Analysis

### 4.1 Word Clouds

![Word Cloud - Human Written Text](images/wordcloud_human.png)

![Word Cloud - AI Generated Text](images/wordcloud_ai.png)

### 4.2 Top 30 Most Frequent Words

![Top 30 Words - Human](images/top_words_human.png)

![Top 30 Words - AI](images/top_words_ai.png)

---

## 5. Readability Metrics

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
"""
        for metric, values in readability.items():
            report_content += f"| {metric} | {values['Mean']:.2f} | {values['Median']:.2f} | {values['Std']:.2f} | {values['Min']:.2f} | {values['Max']:.2f} |\n"
        
        report_content += f"""
![Readability Metrics](images/readability_metrics.png)

---

## 6. Lexical Diversity

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
"""
        for metric, values in diversity.items():
            report_content += f"| {metric} | {values['Mean']:.2f} | {values['Median']:.2f} | {values['Std']:.2f} | {values['Min']:.2f} | {values['Max']:.2f} |\n"
        
        report_content += f"""
![Lexical Diversity Metrics](images/lexical_diversity.png)

---

## 7. NLP Statistics

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
"""
        for metric, values in nlp_stats.items():
            report_content += f"| {metric} | {values['Mean']:.4f} | {values['Median']:.4f} | {values['Std']:.4f} | {values['Min']:.4f} | {values['Max']:.4f} |\n"
        
        report_content += f"""
![NLP Statistics](images/nlp_statistics.png)

---

## 8. Human vs AI Writing Comparison

| Characteristic | Human Written | AI Generated |
|----------------|---------------|---------------|
"""
        for metric in comparison['Human'].keys():
            report_content += f"| {metric} | {comparison['Human'][metric]:.4f} | {comparison['AI'][metric]:.4f} |\n"
        
        report_content += f"""
![Human vs AI Comparison](images/human_ai_comparison.png)

---

## 9. Key Observations

### 9.1 Dataset Characteristics

1. **Dataset Size:** The dataset contains approximately {summary['Number of Rows']:,} samples, providing a substantial corpus for training AI text detection models.
2. **Class Balance:** The class distribution shows {'balanced' if abs(distribution['Human Written (0)']['Percentage'] - distribution['AI Generated (1)']['Percentage']) < 10 else 'imbalanced'} classes with {distribution['Human Written (0)']['Percentage']:.1f}% human-written and {distribution['AI Generated (1)']['Percentage']:.1f}% AI-generated text.
3. **Text Length Variation:** Text lengths exhibit significant variation (std: {stats['Text Length']['Std']:.2f}), indicating diverse document types and writing styles in the dataset.

### 9.2 Writing Style Differences

1. **Sentence Length:** {'Human-written texts show' if comparison['Human']['Average Sentence Length'] > comparison['AI']['Average Sentence Length'] else 'AI-generated texts show'} longer average sentences ({comparison['Human']['Average Sentence Length']:.2f} vs {comparison['AI']['Average Sentence Length']:.2f}).
2. **Vocabulary Diversity:** {'Human writing demonstrates' if comparison['Human']['Vocabulary Diversity'] > comparison['AI']['Vocabulary Diversity'] else 'AI writing demonstrates'} higher vocabulary diversity ({comparison['Human']['Vocabulary Diversity']:.4f} vs {comparison['AI']['Vocabulary Diversity']:.4f}).
3. **Readability:** {'Human-written texts are' if comparison['Human']['Readability (Flesch)'] > comparison['AI']['Readability (Flesch)'] else 'AI-generated texts are'} more readable according to the Flesch Reading Ease score ({comparison['Human']['Readability (Flesch)']:.2f} vs {comparison['AI']['Readability (Flesch)']:.2f}).

### 9.3 Lexical Features

1. **Average Word Length:** The average word length is {stats['Average Word Length']['Mean']:.2f} characters, indicating typical academic writing style.
2. **Lexical Diversity:** The Type-Token Ratio (TTR) averages {diversity['Type-Token Ratio (TTR)']['Mean']:.4f}, suggesting {'rich' if diversity['Type-Token Ratio (TTR)']['Mean'] > 0.5 else 'moderate'} vocabulary usage across the dataset.
3. **Stopword Usage:** The average stopword percentage is {nlp_stats['Stopword Percentage']['Mean']:.2%}, consistent with natural language patterns.

---

## 10. Research Insights

### 10.1 Implications for AI Detection

1. **Distinctive Patterns:** The analysis reveals measurable differences between human and AI writing styles, particularly in sentence structure and vocabulary diversity.
2. **Feature Engineering:** The identified metrics (sentence length, vocabulary diversity, readability scores) can serve as effective features for machine learning models.
3. **Model Training:** The balanced class distribution and substantial dataset size provide ideal conditions for training robust detection models.

### 10.2 Potential Challenges

1. **Overlap in Writing Styles:** Some metrics show significant overlap between classes, suggesting that single-feature approaches may be insufficient.
2. **Dataset Variability:** The high standard deviation in text lengths indicates diverse document types, requiring adaptive preprocessing strategies.
3. **Computational Resources:** Processing {summary['Number of Rows']:,} samples requires efficient implementation and adequate computational resources.

### 10.3 Recommendations

1. **Multi-Feature Approach:** Combine multiple linguistic features (lexical, syntactic, semantic) for improved detection accuracy.
2. **Ensemble Methods:** Utilize ensemble learning techniques to leverage diverse feature representations.
3. **Explainability:** Implement explainable AI techniques to provide interpretable results for academic validation.

---

## 11. Conclusion

This comprehensive EDA provides valuable insights into the characteristics of human-written and AI-generated academic text. The dataset of {summary['Number of Rows']:,} samples exhibits balanced class distribution and diverse writing styles, making it suitable for training sophisticated AI text detection models. Key findings include:

- Measurable differences in sentence structure and vocabulary diversity between classes
- Distinct readability patterns that can aid in classification
- Rich linguistic features suitable for feature engineering
- Adequate dataset size for training robust models

These insights will guide the development of the Hybrid Explainable Transformer Framework, ensuring effective detection of AI-generated academic text while maintaining interpretability for research validation.

---

## Appendix

### Generated Files

**Tables:**
- `tables/dataset_summary.csv`
- `tables/class_distribution.csv`
- `tables/text_statistics.csv`
- `tables/readability_metrics.csv`
- `tables/lexical_diversity.csv`
- `tables/nlp_statistics.csv`
- `tables/human_ai_comparison.csv`
- `tables/top_words_human.csv`
- `tables/top_words_ai.csv`

**Images:**
- `images/class_distribution_bar.png`
- `images/class_distribution_pie.png`
- `images/text_length_histogram.png`
- `images/text_length_boxplot.png`
- `images/word_count_distribution.png`
- `images/character_count_distribution.png`
- `images/sentence_count_distribution.png`
- `images/avg_word_length_distribution.png`
- `images/wordcloud_human.png`
- `images/wordcloud_ai.png`
- `images/top_words_human.png`
- `images/top_words_ai.png`
- `images/readability_metrics.png`
- `images/lexical_diversity.png`
- `images/nlp_statistics.png`
- `images/human_ai_comparison.png`

---

*This report was automatically generated by the EDA Analysis Pipeline for the Final Year Project.*
"""
        
        # Save markdown report
        report_path = self.report_dir / 'EDA_Report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report saved to {report_path}")
        
        # Generate PDF report (optional - may fail on Windows without GTK libraries)
        try:
            import markdown
            import weasyprint
            
            # Convert markdown to HTML
            html_content = markdown.markdown(report_content)
            
            # Add basic CSS styling
            html_with_style = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                    h1 {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; }}
                    h2 {{ color: #A23B72; border-bottom: 1px solid #A23B72; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #2E86AB; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                    code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Convert to PDF
            pdf_path = self.report_dir / 'EDA_Report.pdf'
            weasyprint.HTML(string=html_with_style, base_url=str(self.report_dir)).write_pdf(pdf_path)
            logger.info(f"PDF report saved to {pdf_path}")
            
        except ImportError:
            logger.warning("PDF generation requires 'markdown' and 'weasyprint' packages. Install with: pip install markdown weasyprint")
            logger.info("Skipping PDF generation. Markdown report is available.")
        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            logger.info("This is common on Windows. The markdown report is available and can be converted to PDF using other tools.")
        
        logger.info("Report generation completed")


def main():
    """
    Main function to run the complete EDA pipeline.
    """
    logger.info("=" * 80)
    logger.info("Starting EDA Analysis Pipeline")
    logger.info("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = EDAAnalyzer('datasets/merged/final_dataset.csv')
        
        # Load dataset
        analyzer.load_dataset()
        
        # Run EDA components
        logger.info("Running EDA components...")
        
        summary = analyzer.dataset_summary()
        distribution = analyzer.class_distribution()
        stats = analyzer.text_statistics()
        analyzer.plot_histograms()
        analyzer.generate_wordcloud()
        readability = analyzer.readability_metrics()
        diversity = analyzer.lexical_diversity()
        nlp_stats = analyzer.nlp_statistics()
        comparison = analyzer.compare_human_ai()
        
        # Generate report
        analyzer.save_report(summary, distribution, stats, readability, 
                            diversity, nlp_stats, comparison)
        
        logger.info("=" * 80)
        logger.info("EDA Analysis Pipeline Completed Successfully")
        logger.info("=" * 80)
        logger.info(f"Reports saved to: {analyzer.report_dir}")
        logger.info(f"Images saved to: {analyzer.images_dir}")
        logger.info(f"Tables saved to: {analyzer.tables_dir}")
        
    except Exception as e:
        logger.error(f"Error in EDA pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

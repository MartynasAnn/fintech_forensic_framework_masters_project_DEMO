# data_adaptation_module.py

import pandas as pd
import numpy as np
import networkx as nx
import sqlite3
import json
import hashlib
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure logging with audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_adaptation_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataAdaptationModule:
    """
    Converts financial transaction data into a graph structure
    with automatic feature selection based on scientific methods.
    
    Key Features:
    - Automated feature weighting using multiple methods
    - Ground truth documentation and data limitations tracking
    - Chain-of-custody and audit logging
    - Reproducible processing with fixed random seeds
    - Feature importance analysis with Random Forest and Mutual Information
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, 
                 n_core_features: int = 5,
                 variance_threshold: float = 0.01,
                 db_path: str = 'fintech_metadata.db',
                 feature_weighting_method: str = 'auto',
                 random_seed: int = RANDOM_SEED):
        """
        Initialize the Data Adaptation Module.
        
        Args:
            n_core_features: Number of most important features to use for graph
            variance_threshold: Minimum variance for feature selection
            db_path: Path to SQLite database for metadata storage
            feature_weighting_method: Method for feature weighting 
                                     ('auto', 'random_forest', 'mutual_info', 'hybrid')
            random_seed: Random seed for reproducibility
        """
        self.n_core_features = n_core_features
        self.variance_threshold = variance_threshold
        self.db_path = db_path
        self.feature_weighting_method = feature_weighting_method
        self.random_seed = random_seed
        
        self.graph = None
        self.feature_importance = {}
        self.selected_features = []
        self.feature_weights = {}
        self.scaler = StandardScaler()
        
        # Ground truth and data limitations tracking
        self.ground_truth_metadata = {
            'source': 'Unknown',
            'labeling_method': 'Not specified',
            'limitations': [],
            'data_quality_issues': [],
            'license': 'Not specified',
            'ethical_considerations': []
        }
        
        # Chain of custody tracking
        self.custody_chain = []
        self._log_custody_event('Module Initialized', {
            'version': self.VERSION,
            'n_core_features': n_core_features,
            'random_seed': random_seed,
            'feature_weighting_method': feature_weighting_method
        })
        
        logger.info(f"DataAdaptationModule v{self.VERSION} initialized with seed={random_seed}")
    
    def _log_custody_event(self, event: str, metadata: Dict) -> None:
        """
        Log chain-of-custody event with cryptographic hash.
        
        Args:
            event: Description of the event
            metadata: Event metadata
        """
        timestamp = datetime.utcnow().isoformat()
        event_data = {
            'timestamp': timestamp,
            'event': event,
            'metadata': metadata
        }
        
        # Create cryptographic hash of event
        event_str = json.dumps(event_data, sort_keys=True)
        event_hash = hashlib.sha256(event_str.encode()).hexdigest()
        event_data['hash'] = event_hash
        
        self.custody_chain.append(event_data)
        logger.info(f"Custody Event: {event} | Hash: {event_hash[:16]}...")
    
    def set_ground_truth_metadata(self, 
                                  source: str,
                                  labeling_method: str,
                                  limitations: List[str],
                                  license: str = 'Not specified',
                                  ethical_notes: List[str] = None) -> None:
        """
        Document ground truth origin and data limitations.
        
        Args:
            source: Data source description
            labeling_method: How ground truth labels were created
            limitations: List of known data limitations
            license: Data license information
            ethical_notes: Ethical considerations
        """
        self.ground_truth_metadata = {
            'source': source,
            'labeling_method': labeling_method,
            'limitations': limitations,
            'license': license,
            'ethical_considerations': ethical_notes or []
        }
        
        self._log_custody_event('Ground Truth Metadata Set', self.ground_truth_metadata)
        logger.info(f"Ground truth metadata documented: {source}")
    
    def load_data(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """
        Load transaction data from DataFrame, CSV, Excel, or Parquet.

        Args:
            data: Either a DataFrame or path to data file

        Returns:
            DataFrame with loaded data
        """
        try:
            # Check if data is already a DataFrame
            if isinstance(data, pd.DataFrame):
                logger.info(f"Data provided as DataFrame ({len(data)} rows)")
                df = data.copy()

                # Calculate data hash for integrity verification
                data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
            
                self._log_custody_event('Data Loaded', {
                    'source': 'DataFrame (in-memory)',
                    'num_rows': len(df),
                    'num_columns': len(df.columns),
                    'data_hash': data_hash
                })

                logger.info(f"Loaded {len(df)} records from DataFrame")
                return df

            # Handle string file path
            elif isinstance(data, str):
                filepath = data

                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filepath)
                elif filepath.endswith('.parquet'):
                    df = pd.read_parquet(filepath)
                else:
                    raise ValueError("Unsupported file format. Use CSV, Excel, or Parquet.")

                # Calculate data hash for integrity verification
                data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

                self._log_custody_event('Data Loaded', {
                    'filepath': filepath,
                    'num_rows': len(df),
                    'num_columns': len(df.columns),
                    'data_hash': data_hash
                })

                logger.info(f"Loaded {len(df)} records from {filepath}")
                return df

            else:
                raise TypeError(f"Expected DataFrame or file path string, got {type(data)}")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transaction data with quality checks.
        
        Preprocessing steps:
        - Handle missing values
        - Remove duplicates
        - Encode categorical variables
        - Document data quality issues
        - Standardize column names
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        df = df.copy()
        
        initial_rows = len(df)
        quality_issues = []
        
        # Standardize column names (handle different naming conventions)
        column_mapping = {}
        
        # Map source/sender columns
        for col in ['source', 'from', 'sending_address', 'from_account']:
            if col in df.columns and 'sender_id' not in df.columns:
                column_mapping[col] = 'sender_id'
                break
        
        # Map target/receiver columns
        for col in ['target', 'to', 'receiving_address', 'to_account']:
            if col in df.columns and 'receiver_id' not in df.columns:
                column_mapping[col] = 'receiver_id'
                break
        
        # Apply column renaming
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"  ├─ Standardized column names: {column_mapping}")
        
        # Identify key transaction columns
        required_cols = ['sender_id', 'receiver_id', 'amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            quality_issues.append(f"Missing required columns: {missing_cols}")
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Handle missing values in required columns
        existing_required = [col for col in required_cols if col in df.columns]
        if existing_required:
            df = df.dropna(subset=existing_required)
        
        if len(df) < initial_rows:
            rows_dropped = initial_rows - len(df)
            quality_issues.append(f"Dropped {rows_dropped} rows due to missing values")
        
        # Remove duplicate transactions
        if 'transaction_id' in df.columns:
            duplicates = df.duplicated(subset=['transaction_id']).sum()
            if duplicates > 0:
                df = df.drop_duplicates(subset=['transaction_id'])
                quality_issues.append(f"Removed {duplicates} duplicate transactions")
                logger.warning(f"Removed {duplicates} duplicates")
        
        # Fill other missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna('UNKNOWN', inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Document quality issues
        self.ground_truth_metadata['data_quality_issues'] = quality_issues
        
        self._log_custody_event('Data Preprocessed', {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'quality_issues': quality_issues,
            'column_standardization': column_mapping
        })
        
        logger.info(f"Preprocessing complete: {len(df)} valid transactions")
        return df

    
    def compute_automated_feature_weights(self,
                                         df: pd.DataFrame,
                                         features: List[str],
                                         target_col: Optional[str] = None) -> Dict[str, float]:
        """
        Compute automated feature weights using multiple methods.
        
        Methods:
        1. Random Forest feature importance
        2. Mutual Information scores
        3. Hybrid approach combining both
        
        Args:
            df: DataFrame with features
            features: List of feature names
            target_col: Optional ground truth column
            
        Returns:
            Dictionary mapping feature names to weights
        """
        logger.info(f"Computing automated feature weights using method: {self.feature_weighting_method}")
        
        # Prepare feature matrix
        X = df[features].copy()
        
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Create target variable
        if target_col and target_col in df.columns:
            y = df[target_col]
        else:
            # Use amount-based proxy for anomaly detection
            y = (df['amount'] > df['amount'].quantile(0.95)).astype(int) if 'amount' in df.columns else np.random.randint(0, 2, len(df))
        
        weights = {}
        
        if self.feature_weighting_method in ['auto', 'random_forest', 'hybrid']:
            # Method 1: Random Forest Feature Importance
            logger.info("  ├─ Computing Random Forest importance...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=-1)
            rf_model.fit(X, y)
            rf_importance = dict(zip(features, rf_model.feature_importances_))
            
            if self.feature_weighting_method == 'random_forest':
                weights = rf_importance
        
        if self.feature_weighting_method in ['auto', 'mutual_info', 'hybrid']:
            # Method 2: Mutual Information
            logger.info("  ├─ Computing Mutual Information scores...")
            mi_scores = mutual_info_classif(X, y, random_state=self.random_seed)
            mi_importance = dict(zip(features, mi_scores))
            
            if self.feature_weighting_method == 'mutual_info':
                weights = mi_importance
        
        if self.feature_weighting_method in ['auto', 'hybrid']:
            # Method 3: Hybrid - Average of RF and MI (normalized)
            logger.info("  ├─ Computing Hybrid weights...")
            rf_norm = {k: v / (max(rf_importance.values()) + 1e-10) for k, v in rf_importance.items()}
            mi_norm = {k: v / (max(mi_importance.values()) + 1e-10) for k, v in mi_importance.items()}
            
            weights = {
                k: (rf_norm[k] + mi_norm[k]) / 2
                for k in features
            }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        self.feature_weights = weights
        
        self._log_custody_event('Feature Weights Computed', {
            'method': self.feature_weighting_method,
            'top_features': sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        })
        
        logger.info(f"  └─ ✅ Computed weights for {len(weights)} features")
        
        return weights
    
    def select_valuable_features(self, 
                                  df: pd.DataFrame,
                                  target_col: str = None) -> Tuple[List[str], Dict]:
        """
        Automatic feature selection with automated weighting.
        
        Methods:
        1. Variance Threshold (removes low-variance features)
        2. Mutual Information (for feature importance)
        3. Correlation Analysis (removes redundant features)
        4. Automated weighting (Random Forest + Mutual Information)
        
        Args:
            df: Input DataFrame
            target_col: Optional ground truth column
            
        Returns:
            Tuple of (selected_features, feature_scores)
        """
        logger.info("Starting automatic feature selection with automated weighting...")
        
        # Separate core transaction columns (always keep these)
        core_transaction_cols = []
        for col in ['sender_id', 'receiver_id', 'amount', 'timestamp', 'transaction_id']:
            if col in df.columns:
                core_transaction_cols.append(col)
        
        # Get numerical and categorical features (excluding core columns)
        feature_cols = [col for col in df.columns if col not in core_transaction_cols]
        
        if not feature_cols:
            return core_transaction_cols, {}
        
        # Prepare feature matrix
        X = df[feature_cols].copy()
        
        # Encode categorical variables
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Method 1: Variance Threshold
        logger.info("  ├─ Applying Variance Threshold...")
        selector = VarianceThreshold(threshold=self.variance_threshold)
        try:
            X_var = selector.fit_transform(X)
            var_features = X.columns[selector.get_support()].tolist()
        except:
            var_features = X.columns.tolist()
        
        # Method 2: Mutual Information
        logger.info("  ├─ Computing Mutual Information scores...")
        if target_col and target_col in df.columns:
            y = df[target_col]
        else:
            y = (df['amount'] > df['amount'].quantile(0.95)).astype(int) if 'amount' in df.columns else np.random.randint(0, 2, len(df))
        
        try:
            mi_scores = mutual_info_classif(X[var_features], y, random_state=self.random_seed)
            mi_dict = dict(zip(var_features, mi_scores))
        except:
            mi_dict = {col: 1.0 for col in var_features}
        
        # Method 3: Correlation Analysis
        logger.info("  ├─ Analyzing feature correlations...")
        corr_matrix = X[var_features].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper_triangle.columns 
                   if any(upper_triangle[col] > 0.95)]
        
        final_features = [f for f in var_features if f not in to_drop]
        
        # Method 4: Compute automated weights
        logger.info("  ├─ Computing automated feature weights...")
        feature_weights = self.compute_automated_feature_weights(df, final_features, target_col)
        
        # Rank by weights and select top N
        sorted_features = sorted(
            final_features, 
            key=lambda x: feature_weights.get(x, 0), 
            reverse=True
        )
        
        selected_features = sorted_features[:self.n_core_features]
        
        # Always include core transaction columns
        selected_features = core_transaction_cols + selected_features
        
        # Prepare feature importance scores
        feature_scores = {
            'mutual_information': mi_dict,
            'automated_weights': feature_weights,
            'selected': selected_features,
            'dropped_correlated': to_drop,
            'weighting_method': self.feature_weighting_method
        }
        
        logger.info(f"  └─ ✅ Selected {len(selected_features)} features")
        logger.info(f"     Core features: {core_transaction_cols}")
        logger.info(f"     Additional features: {selected_features[len(core_transaction_cols):]}")
        logger.info(f"     Top weighted features: {sorted_features[:3]}")
        
        self.selected_features = selected_features
        self.feature_importance = feature_scores
        
        self._log_custody_event('Features Selected', {
            'num_selected': len(selected_features),
            'weighting_method': self.feature_weighting_method,
            'top_features': selected_features[:10]
        })
        
        return selected_features, feature_scores
    
    def store_metadata_to_sql(self, 
                              df: pd.DataFrame, 
                              selected_features: List[str]) -> None:
        """
        Store non-selected features, metadata, and audit trail in SQL database.
        
        Stores:
        - Transaction metadata
        - Feature importance scores
        - Feature weights
        - Ground truth metadata
        - Chain of custody log
        
        Args:
            df: Input DataFrame
            selected_features: List of selected features
        """
        logger.info("Storing metadata to SQL database...")
        
        unused_features = [col for col in df.columns if col not in selected_features]
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Store transaction metadata
            if unused_features:
                metadata_df = df[['transaction_id'] + unused_features] if 'transaction_id' in df.columns else df[unused_features]
                metadata_df.to_sql('transaction_metadata', conn, if_exists='replace', index=False)
                logger.info(f"  ├─ Stored {len(unused_features)} unused features")
            
            # Store feature importance scores
            if self.feature_importance and 'mutual_information' in self.feature_importance:
                importance_df = pd.DataFrame([
                    {'feature': k, 'mi_score': v}
                    for k, v in self.feature_importance['mutual_information'].items()
                ])
                importance_df.to_sql('feature_importance', conn, if_exists='replace', index=False)
            
            # Store automated feature weights
            if self.feature_weights:
                weights_df = pd.DataFrame([
                    {'feature': k, 'weight': v, 'method': self.feature_weighting_method}
                    for k, v in self.feature_weights.items()
                ])
                weights_df.to_sql('feature_weights', conn, if_exists='replace', index=False)
                logger.info(f"  ├─ Stored automated weights for {len(self.feature_weights)} features")
            
            # Store ground truth metadata (convert lists to JSON strings)
            gt_metadata_serializable = {
                'source': self.ground_truth_metadata['source'],
                'labeling_method': self.ground_truth_metadata['labeling_method'],
                'limitations': json.dumps(self.ground_truth_metadata['limitations']),  # ← Convert list to JSON
                'data_quality_issues': json.dumps(self.ground_truth_metadata['data_quality_issues']),  # ← Convert list to JSON
                'license': self.ground_truth_metadata['license'],
                'ethical_considerations': json.dumps(self.ground_truth_metadata['ethical_considerations'])  # ← Convert list to JSON
            }
            gt_df = pd.DataFrame([gt_metadata_serializable])
            gt_df.to_sql('ground_truth_metadata', conn, if_exists='replace', index=False)
            logger.info("  ├─ Stored ground truth metadata")
            
            # Store chain of custody (convert nested dicts/lists to JSON)
            custody_serializable = []
            for event in self.custody_chain:
                custody_serializable.append({
                    'timestamp': event['timestamp'],
                    'event': event['event'],
                    'metadata': json.dumps(event['metadata']),  # ← Convert dict to JSON
                    'hash': event['hash']
                })
            custody_df = pd.DataFrame(custody_serializable)
            custody_df.to_sql('custody_chain', conn, if_exists='replace', index=False)
            logger.info(f"  ├─ Stored {len(self.custody_chain)} custody events")
            
            conn.close()
            logger.info("  └─ ✅ All metadata stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing metadata: {e}")
            raise

    
    def construct_graph(self, 
                       df: pd.DataFrame, 
                       selected_features: List[str]) -> nx.Graph:
        """
        Construct a NetworkX graph from selected features.
        
        Graph structure:
        - Nodes: Users (sender_id, receiver_id)
        - Edges: Transactions with selected features as attributes
        - Edge weights: Based on automated feature weighting
        
        Args:
            df: Input DataFrame
            selected_features: List of selected features
            
        Returns:
            NetworkX DiGraph
        """
        logger.info("Constructing transaction graph with weighted features...")
        
        G = nx.DiGraph()
        
        sender_col = 'sender_id' if 'sender_id' in df.columns else df.columns[0]
        receiver_col = 'receiver_id' if 'receiver_id' in df.columns else df.columns[1]
        
        for idx, row in df.iterrows():
            sender = str(row[sender_col])
            receiver = str(row[receiver_col])
            
            # Add nodes
            if not G.has_node(sender):
                G.add_node(sender, node_type='user', user_id=sender)
            if not G.has_node(receiver):
                G.add_node(receiver, node_type='user', user_id=receiver)
            
            # Prepare edge attributes with automated weights
            edge_attrs = {}
            weighted_score = 0.0
            
            for feature in selected_features:
                if feature in row and feature not in [sender_col, receiver_col]:
                    feature_value = row[feature]
                    edge_attrs[feature] = feature_value
                    
                    # Add weighted contribution
                    if feature in self.feature_weights:
                        weight = self.feature_weights[feature]
                        # Normalize feature value if numeric
                        if isinstance(feature_value, (int, float)):
                            weighted_score += weight * float(feature_value)
            
            edge_attrs['weighted_importance_score'] = weighted_score
            
            # Add or update edge
            if G.has_edge(sender, receiver):
                G[sender][receiver]['transaction_count'] = G[sender][receiver].get('transaction_count', 1) + 1
                if 'amount' in edge_attrs:
                    G[sender][receiver]['total_amount'] = G[sender][receiver].get('total_amount', 0) + edge_attrs['amount']
                G[sender][receiver]['weighted_importance_score'] = G[sender][receiver].get('weighted_importance_score', 0) + weighted_score
            else:
                edge_attrs['transaction_count'] = 1
                if 'amount' in edge_attrs:
                    edge_attrs['total_amount'] = edge_attrs['amount']
                G.add_edge(sender, receiver, **edge_attrs)
        
        logger.info(f"  └─ ✅ Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        self.graph = G
        
        self._log_custody_event('Graph Constructed', {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'weighted_features': True
        })
        
        return G
    
    def add_graph_features(self, G: nx.Graph) -> nx.Graph:
        """
        Add graph-based features to nodes for anomaly detection.
        
        Features computed:
        - Degree centrality
        - Betweenness centrality
        - PageRank
        - Clustering coefficient
        
        Args:
            G: Input graph
            
        Returns:
            Graph with added node features
        """
        logger.info("Computing graph-based features...")
        
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            pagerank = nx.pagerank(G)
            clustering = nx.clustering(G.to_undirected())
            
            for node in G.nodes():
                G.nodes[node]['degree_centrality'] = degree_centrality[node]
                G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
                G.nodes[node]['pagerank'] = pagerank[node]
                G.nodes[node]['clustering_coefficient'] = clustering[node]
            
            logger.info("  └─ ✅ Added: degree, betweenness, pagerank, clustering")
            
            self._log_custody_event('Graph Features Added', {
                'features': ['degree_centrality', 'betweenness_centrality', 'pagerank', 'clustering_coefficient']
            })
            
        except Exception as e:
            logger.error(f"Error computing graph features: {e}")
            raise
        
        return G
    
    def generate_reproducibility_report(self) -> Dict:
        """
        Generate a comprehensive reproducibility report.
        
        Returns:
            Dictionary with reproducibility information
        """
        report = {
            'module_version': self.VERSION,
            'random_seed': self.random_seed,
            'configuration': {
                'n_core_features': self.n_core_features,
                'variance_threshold': self.variance_threshold,
                'feature_weighting_method': self.feature_weighting_method
            },
            'dependencies': {
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'networkx': nx.__version__,
                'sklearn': '1.0+'  # Add actual version
            },
            'ground_truth_metadata': self.ground_truth_metadata,
            'custody_chain_length': len(self.custody_chain),
            'feature_selection': {
                'selected_features': self.selected_features,
                'automated_weights': self.feature_weights
            }
        }
        
        return report
    
    def process(self, data: Union[pd.DataFrame, str], ground_truth_info: Dict = None) -> Tuple[nx.Graph, Dict]:

        """
        Main processing pipeline with full reproducibility and audit tracking.
        
        Pipeline:
        1. Load and preprocess data
        2. Document ground truth (if provided)
        3. Select valuable features with automated weighting
        4. Store metadata to SQL
        5. Construct graph with weighted features
        6. Add graph features
        7. Generate reproducibility report
        
        Args:
            filepath: Path to data file
            ground_truth_info: Optional ground truth metadata
            
        Returns:
            Tuple of (graph, metadata)
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 FINTECH FORENSICS - DATA ADAPTATION MODULE v" + self.VERSION)
        logger.info("="*60 + "\n")
        
        # Document ground truth if provided
        if ground_truth_info:
            self.set_ground_truth_metadata(**ground_truth_info)
        
        # Step 1: Load data
        logger.info("📁 Loading transaction data...")
        df = self.load_data(data)
        logger.info(f"  └─ ✅ Loaded {len(df)} transactions with {len(df.columns)} columns\n")
        
        # Step 2: Preprocess
        logger.info("🔧 Preprocessing data...")
        df = self.preprocess_data(df)
        logger.info(f"  └─ ✅ Preprocessed {len(df)} valid transactions\n")
        
        # Step 3: Feature selection with automated weighting
        selected_features, feature_scores = self.select_valuable_features(df)
        logger.info(f"  └─ ✅ Selected {len(selected_features)} features\n")
        
        # Step 4: Store metadata
        self.store_metadata_to_sql(df, selected_features)
        logger.info("  └─ ✅ Metadata stored successfully\n")
        
        # Step 5: Construct graph
        graph = self.construct_graph(df, selected_features)
        logger.info(f"  └─ ✅ Graph created with {graph.number_of_nodes()} nodes\n")
        
        # Step 6: Add graph features
        graph = self.add_graph_features(graph)
        logger.info("  └─ ✅ Graph features added\n")
        
        # Step 7: Generate reproducibility report
        reproducibility_report = self.generate_reproducibility_report()
        
        # Prepare metadata
        metadata = {
            'num_transactions': len(df),
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'selected_features': selected_features,
            'feature_importance': feature_scores,
            'feature_weights': self.feature_weights,
            'graph_density': nx.density(graph),
            'ground_truth_metadata': self.ground_truth_metadata,
            'reproducibility_report': reproducibility_report,
            'custody_chain': self.custody_chain
        }
        
        logger.info("="*60)
        logger.info("✅ DATA ADAPTATION COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Summary:")
        logger.info(f"• Nodes (Users): {metadata['num_nodes']}")
        logger.info(f"• Edges (Transactions): {metadata['num_edges']}")
        logger.info(f"• Graph Density: {metadata['graph_density']:.4f}")
        logger.info(f"• Selected Features: {len(selected_features)}")
        logger.info(f"• Feature Weighting Method: {self.feature_weighting_method}")
        logger.info("="*60 + "\n")
        
        self._log_custody_event('Processing Complete', {
            'num_transactions': len(df),
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        })
        
        return graph, metadata
    
    def export_configuration(self, filepath: str = 'config_export.json') -> None:
        """
        Export module configuration for reproducibility.
        
        Args:
            filepath: Path to save configuration file
        """
        config = {
            'module_version': self.VERSION,
            'random_seed': self.random_seed,
            'n_core_features': self.n_core_features,
            'variance_threshold': self.variance_threshold,
            'feature_weighting_method': self.feature_weighting_method,
            'db_path': self.db_path,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration exported to {filepath}")


# Example usage and helper functions
def create_sample_data():
    """Create sample financial transaction data for testing."""
    np.random.seed(RANDOM_SEED)
    
    n_transactions = 1000
    n_users = 100
    
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
        'sender_id': np.random.randint(1000, 1000 + n_users, n_transactions),
        'receiver_id': np.random.randint(1000, 1000 + n_users, n_transactions),
        'amount': np.random.lognormal(4, 1.5, n_transactions),
        'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='H'),
        'transaction_type': np.random.choice(['TRANSFER', 'PAYMENT', 'WITHDRAWAL'], n_transactions),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_transactions, p=[0.7, 0.2, 0.1]),
        'status': np.random.choice(['COMPLETED', 'PENDING', 'FAILED'], n_transactions, p=[0.9, 0.07, 0.03]),
        'device_id': np.random.randint(5000, 5100, n_transactions),
        'ip_address': [f'192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}' for _ in range(n_transactions)],
        'merchant_category': np.random.choice(['RETAIL', 'FOOD', 'TRAVEL', 'ENTERTAINMENT', 'OTHER'], n_transactions),
        'risk_score': np.random.uniform(0, 1, n_transactions),
        'account_age_days': np.random.randint(1, 365*3, n_transactions),
        'previous_transactions': np.random.poisson(10, n_transactions),
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_transactions, 50, replace=False)
    df.loc[anomaly_indices, 'amount'] = df.loc[anomaly_indices, 'amount'] * 10
    df.loc[anomaly_indices, 'risk_score'] = np.random.uniform(0.8, 1.0, 50)
    
    return df


if __name__ == "__main__":
    # Create sample data
    print("📝 Creating sample transaction data...")
    sample_df = create_sample_data()
    sample_df.to_csv('sample_transactions.csv', index=False)
    print(f"  └─ ✅ Created sample_transactions.csv with {len(sample_df)} transactions\n")
    
    # Define ground truth metadata (example)
    ground_truth_info = {
        'source': 'Synthetic financial transaction dataset',
        'labeling_method': 'Rule-based: Transactions with amounts > 95th percentile marked as high-risk',
        'limitations': [
            'Synthetic data may not capture real-world transaction patterns',
            'Limited temporal dependencies',
            'Simplified fraud patterns'
        ],
        'license': 'MIT License - For research and educational purposes only',
        'ethical_notes': [
            'No real customer data used',
            'Privacy-preserving synthetic generation',
            'Not for production use without validation'
        ]
    }
    
    # Initialize module with automated feature weighting
    adapter = DataAdaptationModule(
        n_core_features=5,
        variance_threshold=0.01,
        db_path='fintech_metadata.db',
        feature_weighting_method='hybrid',  # Use hybrid approach
        random_seed=RANDOM_SEED
    )
    
    # Process data with ground truth documentation
    graph, metadata = adapter.process(
        'sample_transactions.csv',
        ground_truth_info=ground_truth_info
    )
    
    # Save graph for next module
    nx.write_gpickle(graph, 'transaction_graph.gpickle')
    print("💾 Graph saved to 'transaction_graph.gpickle'")
    
    # Export configuration for reproducibility
    adapter.export_configuration('data_adaptation_config.json')
    
    # Save metadata
    with open('data_adaptation_metadata.json', 'w') as f:
        # Make metadata JSON-serializable
        json_metadata = {
            'num_transactions': metadata['num_transactions'],
            'num_nodes': metadata['num_nodes'],
            'num_edges': metadata['num_edges'],
            'selected_features': metadata['selected_features'],
            'graph_density': metadata['graph_density'],
            'feature_weighting_method': adapter.feature_weighting_method,
            'ground_truth_metadata': metadata['ground_truth_metadata'],
            'module_version': adapter.VERSION
        }
        json.dump(json_metadata, f, indent=2)
    print("💾 Metadata saved to 'data_adaptation_metadata.json'")
    
    # Display feature weights summary
    print("\n" + "="*60)
    print("📊 AUTOMATED FEATURE WEIGHTS SUMMARY")
    print("="*60)
    print(f"Method: {adapter.feature_weighting_method}")
    print("\nTop 5 Features by Weight:")
    sorted_weights = sorted(adapter.feature_weights.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, weight) in enumerate(sorted_weights[:5], 1):
        print(f"  {i}. {feature}: {weight:.4f}")
    print("="*60)
    
    # Display reproducibility information
    print("\n" + "="*60)
    print("🔬 REPRODUCIBILITY INFORMATION")
    print("="*60)
    print(f"Module Version: {adapter.VERSION}")
    print(f"Random Seed: {adapter.random_seed}")
    print(f"Configuration File: data_adaptation_config.json")
    print(f"Audit Log: data_adaptation_audit.log")
    print(f"Metadata Database: {adapter.db_path}")
    print(f"Chain of Custody Events: {len(adapter.custody_chain)}")
    print("="*60 + "\n")
    
    print("✅ Data Adaptation Module completed successfully!")
    print("   Next step: Run anomaly_detection_module.py")


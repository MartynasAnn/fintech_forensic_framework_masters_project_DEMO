# main.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import sys
import traceback
import time
import hashlib

# Import all modules
try:
    from data_adaptation_module import DataAdaptationModule
    from anomaly_detection_module import AnomalyDetectionModule
    from gui_module import (
        AccessControl, GraphVisualizer, ExplainabilityEngine, 
        TemporalAnalysis, ReportGenerator
    )
except ImportError as e:
    st.error(f"⚠️ Error importing modules: {e}")
    st.info("Please ensure all module files are in the same directory.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fintech_forensics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="FinTech Forensics - Complete Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class PipelineOrchestrator:
    """
    Orchestrates the complete pipeline from file upload to visualization.
    """
    
    def __init__(self):
        self.steps = {
            'file_upload': {'status': 'pending', 'data': None},
            'data_adaptation': {'status': 'pending', 'data': None},
            'anomaly_detection': {'status': 'pending', 'data': None},
            'visualization': {'status': 'pending', 'data': None}
        }
        
    def run_file_upload(self, uploaded_file) -> bool:
        """Step 1: File Upload and Validation"""
        try:
            logger.info("=" * 60)
            logger.info("STEP 1: FILE UPLOAD AND VALIDATION")
            logger.info("=" * 60)
            
            # Read file based on extension
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Validate data
            required_cols = ['source', 'target', 'amount', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info(f"Required columns: {', '.join(required_cols)}")
                st.info(f"Found columns: {', '.join(df.columns.tolist())}")
                return False
            
            # Store data
            self.steps['file_upload']['data'] = df
            self.steps['file_upload']['status'] = 'completed'
            
            logger.info(f"✅ File uploaded successfully: {len(df)} transactions")
            return True
            
        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            st.error(f"Error reading file: {e}")
            return False
    
    def run_data_adaptation(self) -> bool:
        """Step 2: Data Adaptation - Transform to Graph"""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: DATA ADAPTATION MODULE")
            logger.info("=" * 60)
            
            df = self.steps['file_upload']['data']
            
            # Initialize data adaptation module
            adapter = DataAdaptationModule(
                n_core_features=5,
                feature_weighting_method='hybrid',
                random_seed=42
            )
            
            # ✅ FIXED: Unpack the tuple properly
            processed_graph, adaptation_metadata = adapter.process(df)
            
            # Store results
            self.steps['data_adaptation']['data'] = {
                'graph': processed_graph,
                'metadata': adaptation_metadata,
                'statistics': {
                    'num_nodes': processed_graph.number_of_nodes(),
                    'num_edges': processed_graph.number_of_edges(),
                    'num_transactions': len(df),
                    'selected_features': adaptation_metadata.get('selected_features', []),
                    'feature_weights': adaptation_metadata.get('feature_weights', {})
                }
            }
            self.steps['data_adaptation']['status'] = 'completed'
            
            logger.info(f"✅ Data adaptation completed: {processed_graph.number_of_nodes()} nodes, {processed_graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data adaptation failed: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Data adaptation error: {e}")
            return False
    
    def run_anomaly_detection(self) -> bool:
        """Step 3: Anomaly Detection"""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 3: ANOMALY DETECTION MODULE")
            logger.info("=" * 60)
            
            graph = self.steps['data_adaptation']['data']['graph']
            
            # Get parameters from session state
            lof_neighbors = st.session_state.get('lof_neighbors', 20)
            lof_contamination = st.session_state.get('lof_contamination', 0.1)
            gae_hidden_dim = st.session_state.get('gae_hidden_dim', 32)
            gae_embedding_dim = st.session_state.get('gae_embedding_dim', 16)
            gae_epochs = st.session_state.get('gae_epochs', 100)
            anomaly_threshold = st.session_state.get('anomaly_threshold', 0.7)
            
            # Initialize anomaly detection module
            detector = AnomalyDetectionModule(
                lof_neighbors=lof_neighbors,
                lof_contamination=lof_contamination,
                gae_hidden_dim=gae_hidden_dim,
                gae_embedding_dim=gae_embedding_dim,
                gae_epochs=gae_epochs,
                anomaly_threshold=anomaly_threshold,
                fusion_weights=(0.5, 0.5),
                random_seed=42
            )
            
            # Run detection (without ground truth in real scenario)
            marked_graph, results = detector.process(
                graph,
                y_true=None,  # No ground truth in production
                run_ablation=False,  # Skip ablation for speed
                run_baselines=False,  # Skip baselines for speed
                compute_shap=False  # Skip SHAP for speed
            )
            
            # Store results
            self.steps['anomaly_detection']['data'] = {
                'graph': marked_graph,
                'results': results,
                'statistics': results['statistics']
            }
            self.steps['anomaly_detection']['status'] = 'completed'
            
            num_anomalies = results['statistics']['num_anomalies']
            logger.info(f"✅ Anomaly detection completed: {num_anomalies} anomalies detected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Anomaly detection error: {e}")
            return False
    
    def get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        if status == 'completed':
            return '✅'
        elif status == 'running':
            return '⏳'
        elif status == 'failed':
            return '❌'
        else:
            return '⏸️'



def login_page():
    """Display login page."""
    st.markdown("<div class='main-header'>🔐 FinTech Forensics - Login</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### Secure Access Portal
        Please enter your credentials to access the anomaly detection pipeline.
        """)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submit:
                user = AccessControl.authenticate(username, password)
                if user:
                    st.session_state['user'] = user
                    st.session_state['login_time'] = datetime.now()
                    st.success(f"✅ Welcome, {user['name']}!")
                    AccessControl.log_access(user, 'LOGIN', 'Pipeline', {'status': 'success'})
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        st.markdown("---")
        
        with st.expander("📋 Demo Accounts"):
            st.markdown("""
            **Available Demo Accounts:**
            
            1. **Administrator**
               - Username: `admin`
               - Password: `admin123`
               - Full access to all features
            
            2. **Financial Analyst**
               - Username: `analyst`
               - Password: `analyst123`
               - Analysis and export access
            
            3. **Compliance Viewer**
               - Username: `viewer`
               - Password: `viewer123`
               - Read-only access
            """)

def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Intelligently detect which columns map to required fields.
    Returns suggested mappings.
    """
    required_fields = {
        'source': ['source', 'from', 'sender', 'from_account', 'origin', 'src', 'payer', 'from_id', 'source_id'],
        'target': ['target', 'to', 'receiver', 'recipient', 'to_account', 'destination', 'dst', 'payee', 'to_id', 'target_id'],
        'amount': ['amount', 'value', 'sum', 'total', 'transaction_amount', 'payment', 'price', 'volume'],
        'timestamp': ['timestamp', 'date', 'time', 'datetime', 'transaction_date', 'created_at', 'date_time', 'trans_date']
    }
    
    detected = {}
    columns_lower = [col.lower() for col in df.columns]
    
    for field, keywords in required_fields.items():
        for keyword in keywords:
            if keyword in columns_lower:
                # Get original column name (with proper case)
                original_col = df.columns[columns_lower.index(keyword)]
                detected[field] = original_col
                break
    
    return detected


def column_mapping_interface(df: pd.DataFrame) -> Dict[str, str]:
    """
    Interactive interface for mapping user columns to required fields.
    """
    st.markdown("### 🗺️ Column Mapping")
    st.info("Map your data columns to the required fields for analysis.")
    
    # Auto-detect suggestions
    suggestions = detect_column_mapping(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required Fields:**")
        st.markdown("""
        - **Source**: Sender/origin account
        - **Target**: Receiver/destination account  
        - **Amount**: Transaction value
        - **Timestamp**: Date/time of transaction
        """)
    
    with col2:
        st.markdown("**Your Columns:**")
        st.code(", ".join(df.columns.tolist()))
    
    st.markdown("---")
    
    # Mapping interface
    mapping = {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🔵 Source Account**")
        default_source = suggestions.get('source', df.columns[0] if len(df.columns) > 0 else None)
        mapping['source'] = st.selectbox(
            "Source Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_source) if default_source in df.columns else 0,
            key='map_source'
        )
        if mapping['source']:
            st.caption(f"Sample: {df[mapping['source']].iloc[0]}")
    
    with col2:
        st.markdown("**🟢 Target Account**")
        default_target = suggestions.get('target', df.columns[1] if len(df.columns) > 1 else None)
        mapping['target'] = st.selectbox(
            "Target Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_target) if default_target in df.columns else 0,
            key='map_target'
        )
        if mapping['target']:
            st.caption(f"Sample: {df[mapping['target']].iloc[0]}")
    
    with col3:
        st.markdown("**💰 Amount**")
        default_amount = suggestions.get('amount', df.columns[2] if len(df.columns) > 2 else None)
        mapping['amount'] = st.selectbox(
            "Amount Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_amount) if default_amount in df.columns else 0,
            key='map_amount'
        )
        if mapping['amount']:
            st.caption(f"Sample: {df[mapping['amount']].iloc[0]}")
    
    with col4:
        st.markdown("**⏰ Timestamp**")
        default_timestamp = suggestions.get('timestamp', df.columns[3] if len(df.columns) > 3 else None)
        mapping['timestamp'] = st.selectbox(
            "Timestamp Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_timestamp) if default_timestamp in df.columns else 0,
            key='map_timestamp'
        )
        if mapping['timestamp']:
            st.caption(f"Sample: {df[mapping['timestamp']].iloc[0]}")
    
    # Optional columns
    st.markdown("---")
    st.markdown("#### 📋 Optional Fields (Advanced)")
    
    with st.expander("Map Additional Columns"):
        optional_cols = [col for col in df.columns if col not in mapping.values()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if optional_cols:
                mapping['transaction_id'] = st.selectbox(
                    "Transaction ID (Optional)",
                    options=['None'] + optional_cols,
                    key='map_tx_id'
                )
                if mapping['transaction_id'] == 'None':
                    mapping['transaction_id'] = None
        
        with col2:
            if optional_cols:
                mapping['category'] = st.selectbox(
                    "Category/Type (Optional)",
                    options=['None'] + optional_cols,
                    key='map_category'
                )
                if mapping['category'] == 'None':
                    mapping['category'] = None
    
    # Validation
    st.markdown("---")
    required_fields = ['source', 'target', 'amount', 'timestamp']
    missing = [f for f in required_fields if not mapping.get(f)]
    
    if missing:
        st.error(f"❌ Please map all required fields: {', '.join(missing)}")
        return None
    
    # Check for duplicates
    mapped_cols = [v for k, v in mapping.items() if v and k in required_fields]
    if len(mapped_cols) != len(set(mapped_cols)):
        st.error("❌ Each column can only be mapped once!")
        return None
    
    st.success("✅ All required fields mapped successfully!")
    
    # Preview mapped data
    st.markdown("---")
    st.markdown("#### 👀 Preview Mapped Data")
    
    preview_df = pd.DataFrame({
        'Source': df[mapping['source']].head(3),
        'Target': df[mapping['target']].head(3),
        'Amount': df[mapping['amount']].head(3),
        'Timestamp': df[mapping['timestamp']].head(3)
    })
    st.dataframe(preview_df, use_container_width=True)
    
    return mapping

def file_upload_page(orchestrator: PipelineOrchestrator):
    """File upload interface with intelligent column mapping."""
    st.markdown("<div class='step-header'>📁 Step 1: Data Upload</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your transaction data file in any format. The system will help you map your columns.
    
    **Supported formats:** CSV, Excel (XLSX/XLS), JSON
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload transaction data in any format"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 File selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
        
        try:
            # Read file
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"❌ Unsupported file format: {file_extension}")
                return
            
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Preview raw data
            with st.expander("👀 Preview Raw Data"):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Columns: {', '.join(df.columns.tolist())}")
            
            st.markdown("---")
            
            # Column mapping interface
            column_mapping = column_mapping_interface(df)
            
            if column_mapping:
                # Store mapping in session state
                st.session_state['column_mapping'] = column_mapping
                st.session_state['raw_dataframe'] = df
                
                # Process button
                if st.button("🚀 Confirm and Process", type="primary", use_container_width=True):
                    with st.spinner("Processing file with column mapping..."):
                        # Apply column mapping to create standardized DataFrame
                        mapped_df = pd.DataFrame({
                            'source': df[column_mapping['source']],
                            'target': df[column_mapping['target']],
                            'amount': df[column_mapping['amount']],
                            'timestamp': df[column_mapping['timestamp']]
                        })
                        
                        # Add optional columns if mapped
                        if column_mapping.get('transaction_id'):
                            mapped_df['transaction_id'] = df[column_mapping['transaction_id']]
                        else:
                            # Generate transaction IDs
                            mapped_df['transaction_id'] = [f"TX_{i:06d}" for i in range(len(mapped_df))]
                        
                        if column_mapping.get('category'):
                            mapped_df['category'] = df[column_mapping['category']]
                        
                        # Store in orchestrator
                        orchestrator.steps['file_upload']['data'] = mapped_df
                        orchestrator.steps['file_upload']['status'] = 'completed'
                        
                        st.success("✅ File uploaded and mapped successfully!")
                        st.session_state['pipeline_step'] = 'data_adaptation'
                        
                        logger.info(f"✅ File uploaded with mapping: {column_mapping}")
                        time.sleep(1)
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            logger.error(f"File upload error: {e}")
            logger.error(traceback.format_exc())

def data_adaptation_page(orchestrator: PipelineOrchestrator):
    """Data adaptation interface."""
    st.markdown("<div class='step-header'>🔄 Step 2: Data Adaptation</div>", unsafe_allow_html=True)
    
    df = orchestrator.steps['file_upload']['data']
    
    st.markdown(f"""
    Transform transaction data into a graph structure for anomaly detection.
    
    **Data Summary:**
    - Total transactions: {len(df):,}
    - Unique accounts: {len(set(df['source'].unique()) | set(df['target'].unique())):,}
    - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}
    """)
    
    # Configuration options
    with st.expander("⚙️ Advanced Configuration"):
        st.markdown("**Feature Engineering Options:**")
        
        col1, col2 = st.columns(2)
        with col1:
            compute_centrality = st.checkbox("Compute Centrality Metrics", value=True)
            compute_pagerank = st.checkbox("Compute PageRank", value=True)
        with col2:
            compute_clustering = st.checkbox("Compute Clustering Coefficient", value=True)
            compute_communities = st.checkbox("Detect Communities", value=True)
        
        st.session_state['compute_centrality'] = compute_centrality
        st.session_state['compute_pagerank'] = compute_pagerank
        st.session_state['compute_clustering'] = compute_clustering
        st.session_state['compute_communities'] = compute_communities
    
    # Process button
    if st.button("🔄 Transform to Graph", type="primary", use_container_width=True):
        with st.spinner("Transforming data to graph structure..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing data adaptation module...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            status_text.text("Building transaction graph...")
            progress_bar.progress(40)
            
            if orchestrator.run_data_adaptation():
                progress_bar.progress(100)
                status_text.text("Graph construction complete!")
                
                # Show results
                stats = orchestrator.steps['data_adaptation']['data']['statistics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nodes (Accounts)", f"{stats['num_nodes']:,}")
                with col2:
                    st.metric("Edges (Transaction Paths)", f"{stats['num_edges']:,}")
                with col3:
                    st.metric("Original Transactions", f"{stats['num_transactions']:,}")
                
                st.success("✅ Data adaptation completed successfully!")
                st.session_state['pipeline_step'] = 'anomaly_detection'
                time.sleep(2)
                st.rerun()


def anomaly_detection_page(orchestrator: PipelineOrchestrator):
    """Anomaly detection configuration and execution."""
    st.markdown("<div class='step-header'>🔍 Step 3: Anomaly Detection</div>", unsafe_allow_html=True)
    
    graph_data = orchestrator.steps['data_adaptation']['data']
    graph = graph_data['graph']
    
    st.markdown(f"""
    Configure and run anomaly detection algorithms on the transaction graph.
    
    **Graph Summary:**
    - Nodes: {graph.number_of_nodes():,}
    - Edges: {graph.number_of_edges():,}
    - Average Degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}
    """)
    
    # Configuration
    with st.expander("⚙️ Detection Parameters", expanded=True):
        st.markdown("### LOF (Local Outlier Factor) Parameters")
        col1, col2 = st.columns(2)
        with col1:
            lof_neighbors = st.slider("Number of Neighbors", 5, 50, 20)
            st.session_state['lof_neighbors'] = lof_neighbors
        with col2:
            lof_contamination = st.slider("Contamination Rate", 0.01, 0.3, 0.1, 0.01)
            st.session_state['lof_contamination'] = lof_contamination
        
        st.markdown("### GAE (Graph Autoencoder) Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            gae_hidden_dim = st.selectbox("Hidden Dimension", [16, 32, 64], index=1)
            st.session_state['gae_hidden_dim'] = gae_hidden_dim
        with col2:
            gae_embedding_dim = st.selectbox("Embedding Dimension", [8, 16, 32], index=1)
            st.session_state['gae_embedding_dim'] = gae_embedding_dim
        with col3:
            gae_epochs = st.slider("Training Epochs", 50, 200, 100, 10)
            st.session_state['gae_epochs'] = gae_epochs
        
        st.markdown("### Detection Threshold")
        anomaly_threshold = st.slider("Anomaly Score Threshold", 0.0, 1.0, 0.7, 0.05)
        st.session_state['anomaly_threshold'] = anomaly_threshold
        
        st.info(f"💡 Current settings will flag accounts with anomaly score > {anomaly_threshold:.2f}")
    
    # Run detection button
    if st.button("🔍 Run Anomaly Detection", type="primary", use_container_width=True):
        with st.spinner("Running anomaly detection algorithms..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing detection module...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            status_text.text("Running LOF detection...")
            progress_bar.progress(30)
            time.sleep(0.5)
            
            status_text.text("Training Graph Autoencoder...")
            progress_bar.progress(50)
            
            if orchestrator.run_anomaly_detection():
                progress_bar.progress(100)
                status_text.text("Anomaly detection complete!")
                
                # Show results
                stats = orchestrator.steps['anomaly_detection']['data']['statistics']
                
                st.markdown("---")
                st.markdown("### 🎯 Detection Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Accounts", f"{stats['total_nodes']:,}")
                with col2:
                    st.metric("Anomalies Detected", f"{stats['num_anomalies']:,}", 
                             delta=f"{stats['anomaly_percentage']:.1f}%")
                with col3:
                    st.metric("LOF Only", f"{stats['detection_breakdown']['lof_only']:,}")
                with col4:
                    st.metric("GAE Only", f"{stats['detection_breakdown']['gae_only']:,}")
                
                st.success("✅ Anomaly detection completed successfully!")
                st.session_state['pipeline_step'] = 'visualization'
                time.sleep(2)
                st.rerun()


def visualization_page(orchestrator: PipelineOrchestrator):
    """Results visualization and analysis."""
    user = st.session_state.get('user')
    
    st.markdown("<div class='step-header'>📊 Step 4: Results & Visualization</div>", unsafe_allow_html=True)
    
    detection_data = orchestrator.steps['anomaly_detection']['data']
    graph = detection_data['graph']
    stats = detection_data['statistics']
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔍 Graph View", 
        "🎯 Investigations", 
        "📈 Explainability",
        "📄 Reports"
    ])
    
    with tab1:
        dashboard_tab(graph, stats, user)
    
    with tab2:
        graph_visualization_tab(graph, user)
    
    with tab3:
        investigation_tab(graph, user)
    
    with tab4:
        explainability_tab(graph, user)
    
    with tab5:
        reports_tab(graph, stats, user)


def dashboard_tab(graph: nx.Graph, stats: Dict, user: Dict):
    """Dashboard tab content."""
    st.subheader("📊 Detection Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts", f"{stats['total_nodes']:,}")
    with col2:
        st.metric("Anomalies", f"{stats['num_anomalies']:,}", 
                 delta=f"{stats['anomaly_percentage']:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Both Methods", f"{stats['detection_breakdown']['both']:,}")
    with col4:
        avg_score = stats.get('avg_fused_score', 0)
        st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Detection Method Breakdown")
        import plotly.express as px
        
        breakdown = stats['detection_breakdown']
        fig = px.pie(
            values=[breakdown['lof_only'], breakdown['gae_only'], breakdown['both']],
            names=['LOF Only', 'GAE Only', 'Both Methods'],
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Anomaly Score Distribution")
        
        anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
        scores = [graph.nodes[n].get('anomaly_score', 0) for n in anomaly_nodes]
        
        if scores:
            fig = px.histogram(x=scores, nbins=30, 
                             labels={'x': 'Anomaly Score', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)

def graph_visualization_tab(graph: nx.Graph, user: Dict):
    """Graph visualization tab."""
    st.subheader("🔍 Interactive Graph Visualization")
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    with col1:
        highlight_anomalies = st.checkbox("Highlight Anomalies", value=True)
    with col2:
        show_labels = st.checkbox("Show Labels", value=False)
    with col3:
        layout = st.selectbox("Layout", ['spring', 'circular', 'kamada_kawai'])
    
    # Sample if too large
    display_graph = graph
    if graph.number_of_nodes() > 1000:
        st.warning(f"⚠️ Graph has {graph.number_of_nodes()} nodes. Sampling 1000 nodes for visualization...")
        
        anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
        normal_nodes = [n for n, d in graph.nodes(data=True) if not d.get('is_anomaly', False)]
        
        sample_normal = np.random.choice(normal_nodes, min(1000 - len(anomaly_nodes), len(normal_nodes)), replace=False)
        sampled_nodes = list(anomaly_nodes) + list(sample_normal)
        display_graph = graph.subgraph(sampled_nodes).copy()
    
    # Generate visualization
    with st.spinner("Generating graph visualization..."):
        fig = GraphVisualizer.create_interactive_graph(
            display_graph,
            highlight_anomalies=highlight_anomalies,
            show_labels=show_labels,
            layout=layout
        )
        st.plotly_chart(fig, use_container_width=True)
    
    AccessControl.log_access(user, 'VIEW', 'Graph Visualization')


def investigation_tab(graph: nx.Graph, user: Dict):
    """Investigation tab content."""
    st.subheader("🎯 Node Investigation")
    
    anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_node = st.selectbox("Select Account to Investigate", sorted(anomaly_nodes))
    with col2:
        radius = st.slider("Subgraph Radius", 1, 5, 2)
    
    if selected_node:
        node_data = graph.nodes[selected_node]
        
        # Node info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Anomaly Score", f"{node_data.get('anomaly_score', 0):.3f}")
        with col2:
            st.metric("LOF Score", f"{node_data.get('lof_score', 0):.3f}")
        with col3:
            st.metric("GAE Score", f"{node_data.get('gae_score', 0):.3f}")
        with col4:
            st.metric("Degree", graph.degree(selected_node))
        
        st.info(f"🔍 **Detected by:** {node_data.get('detected_by', 'N/A')}")
        
        # Subgraph
        st.markdown("---")
        st.markdown(f"### 🕸️ Local Network ({radius}-hop)")
        
        with st.spinner("Extracting subgraph..."):
            fig = GraphVisualizer.create_subgraph_visualization(graph, selected_node, radius)
            st.plotly_chart(fig, use_container_width=True)
        
        AccessControl.log_access(user, 'INVESTIGATE', f'Node:{selected_node}')


def explainability_tab(graph: nx.Graph, user: Dict):
    """Explainability tab content."""
    st.subheader("📈 Explainability & Feature Analysis")
    
    anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
    
    if not anomaly_nodes:
        st.warning("No anomalies detected")
        return
    
    selected_node = st.selectbox("Select Anomalous Account", sorted(anomaly_nodes))
    
    if selected_node:
        with st.spinner("Generating explanation..."):
            explanation = ExplainabilityEngine.generate_node_explanation(graph, selected_node)
        
        # Status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anomaly Score", f"{explanation['anomaly_score']:.3f}")
        with col2:
            st.metric("Methods", ', '.join(explanation['detection_methods']))
        with col3:
            risk = "🔴 HIGH" if explanation['anomaly_score'] > 0.8 else "🟡 MEDIUM" if explanation['anomaly_score'] > 0.6 else "🟢 LOW"
            st.markdown(f"**Risk Level:** {risk}")
        
        # Raw explanation
        st.markdown("---")
        st.markdown("### 📝 Automated Explanation")
        st.info(explanation['raw_explanation'])
        
        # Feature contributions
        st.markdown("---")
        st.markdown("### 📊 Feature Contributions")
        
        fig = ExplainabilityEngine.create_explanation_chart(explanation)
        st.plotly_chart(fig, use_container_width=True)
        
        # Behavioral patterns
        if explanation['behavioral_patterns']:
            st.markdown("---")
            st.markdown("### 🎯 Behavioral Patterns")
            for pattern in explanation['behavioral_patterns']:
                st.markdown(f"• {pattern}")
        
        # Risk factors
        if explanation['risk_factors']:
            st.markdown("---")
            st.markdown("### ⚠️ Risk Factors")
            for factor in explanation['risk_factors']:
                st.markdown(f"• {factor}")
        
        AccessControl.log_access(user, 'ANALYZE', f'Explainability:{selected_node}')


def reports_tab(graph: nx.Graph, stats: Dict, user: Dict):
    """Reports tab content."""
    st.subheader("📄 Report Generation & Export")
    
    st.markdown("""
    Generate comprehensive investigation reports and export data for further analysis.
    """)
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", [
            "Executive Summary",
            "Detailed Analysis",
            "Technical Report"
        ])
    
    with col2:
        export_format = st.selectbox("Export Format", [
            "Text (.txt)",
            "CSV (.csv)",
            "JSON (.json)"
        ])
    
    include_top_n = st.slider("Include Top N Anomalies", 5, 50, 10)
    
    # Generate report
    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            # Generate text report
            report_text = ReportGenerator.generate_text_report(graph, {}, stats)
            
            st.success("✅ Report generated successfully!")
            
            # Preview
            st.markdown("---")
            st.markdown("### 📄 Report Preview")
            st.text_area("Report Content", report_text, height=400)
            
            # Download buttons
            st.markdown("---")
            st.markdown("### 💾 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download Text Report",
                    data=report_text,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Generate CSV
                df_export = ReportGenerator.export_to_csv(graph, 'temp.csv')
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Data",
                    data=csv,
                    file_name=f"anomaly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Generate JSON
                json_data = {
                    'report_id': hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:16],
                    'generated_at': datetime.utcnow().isoformat(),
                    'generated_by': user['username'],
                    'statistics': stats
                }
                st.download_button(
                    label="📥 Download JSON Metadata",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"anomaly_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            AccessControl.log_access(user, 'EXPORT', f'Report Generated: {report_type}')


def pipeline_progress_sidebar(orchestrator: PipelineOrchestrator):
    """Display pipeline progress in sidebar."""
    with st.sidebar:
        st.markdown("### 🔄 Pipeline Progress")
        
        steps_info = [
            ("📁 File Upload", 'file_upload'),
            ("🔄 Data Adaptation", 'data_adaptation'),
            ("🔍 Anomaly Detection", 'anomaly_detection'),
            ("📊 Visualization", 'visualization')
        ]
        
        for step_name, step_key in steps_info:
            status = orchestrator.steps[step_key]['status']
            emoji = orchestrator.get_status_emoji(status)
            st.markdown(f"{emoji} {step_name}")
        
        st.markdown("---")


def main():
    """Main application entry point."""
    
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    
    if 'pipeline_step' not in st.session_state:
        st.session_state['pipeline_step'] = 'file_upload'
    
    if 'orchestrator' not in st.session_state:
        st.session_state['orchestrator'] = PipelineOrchestrator()
    
    orchestrator = st.session_state['orchestrator']
    
    # Check authentication
    user = st.session_state.get('user')
    
    if not user:
        login_page()
        return
    
    # Display header
    st.markdown("<div class='main-header'>🔍 FinTech Forensics - Anomaly Detection Pipeline</div>", unsafe_allow_html=True)
    
    # User info and logout
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Welcome, {user['name']}** ({user['role']})")
    with col2:
        login_duration = datetime.now() - st.session_state.get('login_time', datetime.now())
        st.markdown(f"⏱️ Session: {int(login_duration.total_seconds() / 60)} min")
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            AccessControl.log_access(user, 'LOGOUT', 'Pipeline')
            st.session_state['user'] = None
            st.session_state['orchestrator'] = PipelineOrchestrator()
            st.session_state['pipeline_step'] = 'file_upload'
            st.rerun()
    
    st.markdown("---")
    
    # Show pipeline progress in sidebar
    pipeline_progress_sidebar(orchestrator)
    
    # Navigation buttons in sidebar
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        if st.button("📁 1. File Upload", use_container_width=True, 
                    disabled=(orchestrator.steps['file_upload']['status'] == 'completed')):
            st.session_state['pipeline_step'] = 'file_upload'
            st.rerun()
        
        if st.button("🔄 2. Data Adaptation", use_container_width=True,
                    disabled=(orchestrator.steps['file_upload']['status'] != 'completed')):
            st.session_state['pipeline_step'] = 'data_adaptation'
            st.rerun()
        
        if st.button("🔍 3. Anomaly Detection", use_container_width=True,
                    disabled=(orchestrator.steps['data_adaptation']['status'] != 'completed')):
            st.session_state['pipeline_step'] = 'anomaly_detection'
            st.rerun()
        
        if st.button("📊 4. Results", use_container_width=True,
                    disabled=(orchestrator.steps['anomaly_detection']['status'] != 'completed')):
            st.session_state['pipeline_step'] = 'visualization'
            st.rerun()
        
        st.markdown("---")
        
        # Reset pipeline button
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            st.session_state['orchestrator'] = PipelineOrchestrator()
            st.session_state['pipeline_step'] = 'file_upload'
            st.success("✅ Pipeline reset!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.info(f"**Version:** 1.0.0\n\n**Status:** Online\n\n**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Route to appropriate page based on pipeline step
    current_step = st.session_state['pipeline_step']
    
    if current_step == 'file_upload':
        file_upload_page(orchestrator)
    
    elif current_step == 'data_adaptation':
        if orchestrator.steps['file_upload']['status'] == 'completed':
            data_adaptation_page(orchestrator)
        else:
            st.warning("⚠️ Please upload a file first")
            if st.button("Go to File Upload"):
                st.session_state['pipeline_step'] = 'file_upload'
                st.rerun()
    
    elif current_step == 'anomaly_detection':
        if orchestrator.steps['data_adaptation']['status'] == 'completed':
            anomaly_detection_page(orchestrator)
        else:
            st.warning("⚠️ Please complete data adaptation first")
            if st.button("Go to Data Adaptation"):
                st.session_state['pipeline_step'] = 'data_adaptation'
                st.rerun()
    
    elif current_step == 'visualization':
        if orchestrator.steps['anomaly_detection']['status'] == 'completed':
            visualization_page(orchestrator)
        else:
            st.warning("⚠️ Please run anomaly detection first")
            if st.button("Go to Anomaly Detection"):
                st.session_state['pipeline_step'] = 'anomaly_detection'
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
            🔒 FinTech Forensics - Anomaly Detection System v1.0.0<br>
            Secure • Auditable • Explainable<br>
            © 2026 - All Rights Reserved
        </small>
    </div>
    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"""
        ### 🚨 Application Error
        
        An unexpected error occurred:
        ```
        {str(e)}
        ```
        
        Please check the logs for more details.
        """)

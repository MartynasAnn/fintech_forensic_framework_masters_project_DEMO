# gui_module.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional
import pickle
import io
import base64
from pathlib import Path
import time

# For PDF report generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ reportlab not installed. PDF export will be limited.")

# Configure logging with audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Performance limits
MAX_NODES_DISPLAY = 1000
MAX_EDGES_DISPLAY = 5000
MAX_QUERY_TIME = 30  # seconds
MAX_EXPORT_SIZE_MB = 50

# Set page configuration
st.set_page_config(
    page_title="FinTech Forensics - Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


class AccessControl:
    """
    Simple access control and authentication system.
    
    In production, integrate with proper authentication (OAuth2, LDAP, etc.)
    """
    
    ROLES = {
        'admin': ['view', 'analyze', 'export', 'manage_users', 'audit'],
        'analyst': ['view', 'analyze', 'export'],
        'viewer': ['view']
    }
    
    # Demo users (in production, use secure authentication)
    USERS = {
        'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Administrator'},
        'analyst': {'password': 'analyst123', 'role': 'analyst', 'name': 'Financial Analyst'},
        'viewer': {'password': 'viewer123', 'role': 'viewer', 'name': 'Compliance Viewer'}
    }
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info."""
        if username in AccessControl.USERS:
            user = AccessControl.USERS[username]
            # In production, use proper password hashing (bcrypt, argon2)
            if password == user['password']:
                logger.info(f"✅ User authenticated: {username} ({user['role']})")
                return {
                    'username': username,
                    'role': user['role'],
                    'name': user['name'],
                    'permissions': AccessControl.ROLES[user['role']]
                }
        logger.warning(f"❌ Failed authentication attempt for user: {username}")
        return None
    
    @staticmethod
    def has_permission(user: Dict, permission: str) -> bool:
        """Check if user has specific permission."""
        if user is None:
            return False
        return permission in user.get('permissions', [])
    
    @staticmethod
    def log_access(user: Dict, action: str, resource: str, metadata: Dict = None):
        """Log user access for audit trail."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'username': user['username'],
            'role': user['role'],
            'action': action,
            'resource': resource,
            'metadata': metadata or {},
            'ip_address': 'N/A'  # In production, capture real IP
        }
        
        # Create audit hash
        audit_str = json.dumps(audit_entry, sort_keys=True)
        audit_hash = hashlib.sha256(audit_str.encode()).hexdigest()
        audit_entry['hash'] = audit_hash
        
        logger.info(f"Audit: {user['username']} - {action} - {resource} | Hash: {audit_hash[:16]}")
        
        # Save to audit log
        try:
            with open('gui_access_audit.json', 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


class PerformanceMonitor:
    """Monitor and enforce performance limits."""
    
    @staticmethod
    def check_graph_size(graph: nx.Graph) -> Tuple[bool, str]:
        """Check if graph size is within limits."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes > MAX_NODES_DISPLAY:
            return False, f"Graph too large: {num_nodes} nodes (max: {MAX_NODES_DISPLAY})"
        if num_edges > MAX_EDGES_DISPLAY:
            return False, f"Graph too large: {num_edges} edges (max: {MAX_EDGES_DISPLAY})"
        
        return True, "OK"
    
    @staticmethod
    def time_limit(func):
        """Decorator to enforce time limits on operations."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed > MAX_QUERY_TIME:
                logger.warning(f"⚠️ Operation exceeded time limit: {elapsed:.2f}s")
            
            return result
        return wrapper


class GraphVisualizer:
    """
    Enhanced graph visualization with anomaly highlighting and explainability.
    """
    
    @staticmethod
    def create_interactive_graph(graph: nx.Graph, 
                                 highlight_anomalies: bool = True,
                                 show_labels: bool = True,
                                 layout: str = 'spring') -> go.Figure:
        """
        Create interactive Plotly graph visualization.
        
        Args:
            graph: NetworkX graph
            highlight_anomalies: Whether to highlight anomalous nodes
            show_labels: Whether to show node labels
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        """
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Separate normal and anomalous nodes
        normal_nodes = []
        anomaly_nodes = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if highlight_anomalies and node_data.get('is_anomaly', False):
                anomaly_nodes.append(node)
            else:
                normal_nodes.append(node)
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Create node traces
        def create_node_trace(nodes, color, name, size=10):
            node_x = [pos[node][0] for node in nodes]
            node_y = [pos[node][1] for node in nodes]
            
            hover_texts = []
            for node in nodes:
                node_data = graph.nodes[node]
                hover_text = f"<b>Node {node}</b><br>"
                hover_text += f"Anomaly Score: {node_data.get('anomaly_score', 0):.3f}<br>"
                hover_text += f"LOF Score: {node_data.get('lof_score', 0):.3f}<br>"
                hover_text += f"GAE Score: {node_data.get('gae_score', 0):.3f}<br>"
                hover_text += f"Detected By: {node_data.get('detected_by', 'N/A')}<br>"
                hover_text += f"PageRank: {node_data.get('pagerank', 0):.4f}"
                hover_texts.append(hover_text)
            
            return go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if show_labels else 'markers',
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts,
                textposition="top center",
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=name
            )
        
        # Create traces
        traces = [edge_trace]
        
        if normal_nodes:
            traces.append(create_node_trace(normal_nodes, '#1f77b4', 'Normal Nodes', size=8))
        
        if anomaly_nodes:
            traces.append(create_node_trace(anomaly_nodes, '#ff0000', 'Anomaly Nodes', size=15))
        
        # Create figure
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(
                    text='Transaction Network - Anomaly Detection',
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=600
            )
        )
        
        return fig
    
    @staticmethod
    def create_subgraph_visualization(graph: nx.Graph, center_node: str, radius: int = 2) -> go.Figure:
        """
        Create visualization of subgraph around a specific node.
        
        Args:
            graph: Full graph
            center_node: Central node for subgraph extraction
            radius: Number of hops from center node
        """
        # Extract subgraph
        subgraph_nodes = set([center_node])
        
        for _ in range(radius):
            new_nodes = set()
            for node in subgraph_nodes:
                # Add predecessors and successors
                new_nodes.update(graph.predecessors(node))
                new_nodes.update(graph.successors(node))
            subgraph_nodes.update(new_nodes)
        
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        # Highlight center node
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            if node == center_node:
                node_colors.append('#ff00ff')  # Purple for center
                node_sizes.append(20)
            elif subgraph.nodes[node].get('is_anomaly', False):
                node_colors.append('#ff0000')  # Red for anomalies
                node_sizes.append(15)
            else:
                node_colors.append('#1f77b4')  # Blue for normal
                node_sizes.append(10)
        
        pos = nx.spring_layout(subgraph, k=0.5)
        
        # Edge trace
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in subgraph.nodes()],
            y=[pos[node][1] for node in subgraph.nodes()],
            mode='markers',
            hovertext=[f"Node {node}" for node in subgraph.nodes()],
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Subgraph around Node {center_node} (radius={radius})',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
        )
        
        return fig


class ExplainabilityEngine:
    """
    Generate explanations for anomaly detections.
    """
    
    @staticmethod
    def generate_node_explanation(graph: nx.Graph, node: str) -> Dict:
        """
        Generate detailed explanation for why a node is anomalous.
        
        Returns:
            Dictionary with explanation components
        """
        node_data = graph.nodes[node]
        
        explanation = {
            'node_id': node,
            'is_anomaly': node_data.get('is_anomaly', False),
            'anomaly_score': node_data.get('anomaly_score', 0),
            'detection_methods': node_data.get('detected_by', 'N/A').split(','),
            'raw_explanation': node_data.get('anomaly_explanation', 'No explanation available'),
            'feature_contributions': {},
            'behavioral_patterns': [],
            'risk_factors': []
        }
        
        # Analyze feature contributions
        features = {
            'LOF Score': node_data.get('lof_score', 0),
            'GAE Score': node_data.get('gae_score', 0),
            'Degree Centrality': node_data.get('degree_centrality', 0),
            'Betweenness Centrality': node_data.get('betweenness_centrality', 0),
            'PageRank': node_data.get('pagerank', 0),
            'Clustering Coefficient': node_data.get('clustering_coefficient', 0)
        }
        
        # Sort features by contribution (absolute value)
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        explanation['feature_contributions'] = dict(sorted_features)
        
        # Identify behavioral patterns
        if node_data.get('lof_score', 0) > 0.7:
            explanation['behavioral_patterns'].append(
                "🔴 Unusual local neighborhood structure - node behavior significantly differs from peers"
            )
        
        if node_data.get('gae_score', 0) > 0.7:
            explanation['behavioral_patterns'].append(
                "🔴 Abnormal graph structural patterns - connectivity differs from expected patterns"
            )
        
        if node_data.get('pagerank', 0) > 0.01:
            explanation['behavioral_patterns'].append(
                "⚠️ High PageRank value - central node in transaction network"
            )
        
        if node_data.get('degree_centrality', 0) > 0.1:
            explanation['behavioral_patterns'].append(
                "⚠️ High degree centrality - many direct connections"
            )
        
        # Identify risk factors
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        
        if in_degree > 20:
            explanation['risk_factors'].append(f"High incoming transactions: {in_degree}")
        if out_degree > 20:
            explanation['risk_factors'].append(f"High outgoing transactions: {out_degree}")
        if in_degree + out_degree > 50:
            explanation['risk_factors'].append("Potential hub or intermediary account")
        
        return explanation
    
    @staticmethod
    def create_explanation_chart(explanation: Dict) -> go.Figure:
        """Create visual chart of feature contributions."""
        features = list(explanation['feature_contributions'].keys())
        values = list(explanation['feature_contributions'].values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker=dict(
                    color=values,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Score")
                )
            )
        ])
        
        fig.update_layout(
            title='Feature Contributions to Anomaly Score',
            xaxis_title='Score',
            yaxis_title='Feature',
            height=400,
            margin=dict(l=150)
        )
        
        return fig


class TemporalAnalysis:
    """
    Temporal analysis and time-based filtering.
    """
    
    @staticmethod
    def extract_temporal_subgraph(graph: nx.Graph, 
                                  start_date: datetime, 
                                  end_date: datetime) -> nx.Graph:
        """
        Extract subgraph based on temporal constraints.
        
        Args:
            graph: Full graph
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            Filtered subgraph
        """
        temporal_edges = []
        
        for u, v, data in graph.edges(data=True):
            timestamp = data.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        edge_time = pd.to_datetime(timestamp)
                    else:
                        edge_time = timestamp
                    
                    if start_date <= edge_time <= end_date:
                        temporal_edges.append((u, v))
                except:
                    pass
        
        subgraph = graph.edge_subgraph(temporal_edges).copy()
        return subgraph
    
    @staticmethod
    def analyze_temporal_patterns(graph: nx.Graph) -> pd.DataFrame:
        """
        Analyze temporal patterns in anomalies.
        
        Returns:
            DataFrame with temporal statistics
        """
        anomaly_times = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if node_data.get('is_anomaly', False):
                # Get timestamps of transactions involving this node
                for u, v, data in graph.edges(node, data=True):
                    timestamp = data.get('timestamp')
                    if timestamp:
                        anomaly_times.append(timestamp)
        
        if not anomaly_times:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame({'timestamp': anomaly_times})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['date'] = df['timestamp'].dt.date
        
        return df


class ReportGenerator:
    """
    Generate comprehensive investigation reports with explanations.
    """
    
    @staticmethod
    def generate_text_report(graph: nx.Graph, 
                            anomaly_results: Dict, 
                            statistics: Dict) -> str:
        """
        Generate detailed text report.
        
        Returns:
            Formatted text report
        """
        report = []
        report.append("="*80)
        report.append("FINTECH FORENSICS - ANOMALY DETECTION INVESTIGATION REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Report ID: {hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:16]}")
        report.append("\n" + "-"*80)
        report.append("EXECUTIVE SUMMARY")
        report.append("-"*80)
        
        # Summary statistics
        total_nodes = statistics.get('total_nodes', 0)
        num_anomalies = statistics.get('num_anomalies', 0)
        anomaly_pct = statistics.get('anomaly_percentage', 0)
        
        report.append(f"\nTotal Accounts Analyzed: {total_nodes}")
        report.append(f"Suspicious Accounts Detected: {num_anomalies} ({anomaly_pct:.2f}%)")
        
        # Detection breakdown
        breakdown = statistics.get('detection_breakdown', {})
        report.append(f"\nDetection Method Breakdown:")
        report.append(f"  • LOF Only: {breakdown.get('lof_only', 0)}")
        report.append(f"  • GAE Only: {breakdown.get('gae_only', 0)}")
        report.append(f"  • Both Methods: {breakdown.get('both', 0)}")
        
        # Evaluation metrics (if available)
        if 'evaluation_metrics' in statistics:
            metrics = statistics['evaluation_metrics']
            report.append("\n" + "-"*80)
            report.append("EVALUATION METRICS")
            report.append("-"*80)
            report.append(f"\nPrecision: {metrics.get('precision', 0):.3f}")
            report.append(f"Recall: {metrics.get('recall', 0):.3f}")
            report.append(f"F1 Score: {metrics.get('f1', 0):.3f}")
            report.append(f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
            report.append(f"PR-AUC: {metrics.get('pr_auc', 0):.3f}")
            report.append(f"Matthews Correlation Coefficient: {metrics.get('mcc', 0):.3f}")
            report.append(f"\nFalse Positive Rate: {metrics.get('fpr', 0):.3f}")
            report.append(f"False Negative Rate: {metrics.get('fnr', 0):.3f}")
        
        # Top anomalies
        report.append("\n" + "-"*80)
        report.append("TOP 10 MOST SUSPICIOUS ACCOUNTS")
        report.append("-"*80)
        
        anomalous_nodes = [(node, data.get('anomaly_score', 0)) 
                          for node, data in graph.nodes(data=True) 
                          if data.get('is_anomaly', False)]
        anomalous_nodes.sort(key=lambda x: x[1], reverse=True)
        
        for i, (node, score) in enumerate(anomalous_nodes[:10], 1):
            node_data = graph.nodes[node]
            report.append(f"\n{i}. Account ID: {node}")
            report.append(f"   Anomaly Score: {score:.3f}")
            report.append(f"   Detected By: {node_data.get('detected_by', 'N/A')}")
            report.append(f"   LOF Score: {node_data.get('lof_score', 0):.3f}")
            report.append(f"   GAE Score: {node_data.get('gae_score', 0):.3f}")
            
            explanation = ExplainabilityEngine.generate_node_explanation(graph, node)
            if explanation['risk_factors']:
                report.append(f"   Risk Factors:")
                for factor in explanation['risk_factors']:
                    report.append(f"     • {factor}")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    @staticmethod
    def export_to_csv(graph: nx.Graph, filepath: str = 'anomaly_report.csv'):
        """Export anomaly data to CSV."""
        data = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            data.append({
                'node_id': node,
                'is_anomaly': node_data.get('is_anomaly', False),
                'anomaly_score': node_data.get('anomaly_score', 0),
                'lof_score': node_data.get('lof_score', 0),
                'gae_score': node_data.get('gae_score', 0),
                'detected_by': node_data.get('detected_by', 'N/A'),
                'degree_centrality': node_data.get('degree_centrality', 0),
                'betweenness_centrality': node_data.get('betweenness_centrality', 0),
                'pagerank': node_data.get('pagerank', 0),
                'clustering_coefficient': node_data.get('clustering_coefficient', 0)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return df


def login_page():
    """Display login page."""
    st.title("🔐 FinTech Forensics - Login")
    
    st.markdown("""
    ### Secure Access Portal
    Please enter your credentials to access the anomaly detection system.
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = AccessControl.authenticate(username, password)
                if user:
                    st.session_state['user'] = user
                    st.success(f"✅ Welcome, {user['name']}!")
                    AccessControl.log_access(user, 'LOGIN', 'System', {'status': 'success'})
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        st.markdown("---")
        st.markdown("""
        **Demo Accounts:**
        - **Admin**: username: `admin`, password: `admin123`
        - **Analyst**: username: `analyst`, password: `analyst123`
        - **Viewer**: username: `viewer`, password: `viewer123`
        """)


def main_application():
    """Main application interface."""
    user = st.session_state.get('user')
    
    if not user:
        login_page()
        return
    
    # Header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 FinTech Forensics - Anomaly Detection System")
    with col2:
        st.markdown(f"**User:** {user['name']}")
        st.markdown(f"**Role:** {user['role'].upper()}")
        if st.button("Logout"):
            AccessControl.log_access(user, 'LOGOUT', 'System')
            del st.session_state['user']
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        pages = {
            "📊 Dashboard": "dashboard",
            "🔍 Graph Visualization": "visualization",
            "🎯 Node Investigation": "investigation",
            "⏱️ Temporal Analysis": "temporal",
            "📈 Explainability": "explainability",
            "📄 Report Generation": "reports"
        }
        
        if AccessControl.has_permission(user, 'audit'):
            pages["🔒 Audit Log"] = "audit"
        
        page = st.radio("Select Page", list(pages.keys()))
        current_page = pages[page]
        
        st.markdown("---")
        st.markdown("### System Status")
        st.success("✅ System Online")
        st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load graph data
    @st.cache_data
    def load_graph():
        try:
            graph = nx.read_gpickle('marked_transaction_graph.gpickle')
            logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes")
            return graph
        except FileNotFoundError:
            st.error("⚠️ Graph file not found. Please run the anomaly detection module first.")
            return None
    
    graph = load_graph()
    
    if graph is None:
        st.stop()
    
    # Check performance limits
    can_display, message = PerformanceMonitor.check_graph_size(graph)
    if not can_display:
        st.warning(f"⚠️ {message}. Some features may be limited.")
    
        # Route to appropriate page
    if current_page == "dashboard":
        dashboard_page(graph, user)
    elif current_page == "visualization":
        if AccessControl.has_permission(user, 'view'):
            visualization_page(graph, user)
        else:
            st.error("❌ You don't have permission to view this page")
    elif current_page == "investigation":
        if AccessControl.has_permission(user, 'analyze'):
            investigation_page(graph, user)
        else:
            st.error("❌ You don't have permission to access this page")
    elif current_page == "temporal":
        if AccessControl.has_permission(user, 'analyze'):
            temporal_analysis_page(graph, user)
        else:
            st.error("❌ You don't have permission to access this page")
    elif current_page == "explainability":
        if AccessControl.has_permission(user, 'analyze'):
            explainability_page(graph, user)
        else:
            st.error("❌ You don't have permission to access this page")
    elif current_page == "reports":
        if AccessControl.has_permission(user, 'export'):
            reports_page(graph, user)
        else:
            st.error("❌ You don't have permission to access this page")
    elif current_page == "audit":
        if AccessControl.has_permission(user, 'audit'):
            audit_log_page(user)
        else:
            st.error("❌ You don't have permission to access this page")


def dashboard_page(graph: nx.Graph, user: Dict):
    """Dashboard page with overview statistics."""
    st.header("📊 Dashboard - System Overview")
    
    AccessControl.log_access(user, 'VIEW', 'Dashboard')
    
    # Summary metrics
    anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    anomaly_percentage = len(anomaly_nodes) / total_nodes * 100 if total_nodes > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts", f"{total_nodes:,}")
    with col2:
        st.metric("Total Transactions", f"{total_edges:,}")
    with col3:
        st.metric("Anomalies Detected", f"{len(anomaly_nodes):,}")
    with col4:
        st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
    
    st.markdown("---")
    
    # Detection breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Detection Method Breakdown")
        
        detection_counts = {'LOF Only': 0, 'GAE Only': 0, 'Both Methods': 0}
        
        for node in anomaly_nodes:
            node_data = graph.nodes[node]
            detected_by = node_data.get('detected_by', '').split(',')
            
            if 'LOF' in detected_by and 'GAE' not in detected_by:
                detection_counts['LOF Only'] += 1
            elif 'GAE' in detected_by and 'LOF' not in detected_by:
                detection_counts['GAE Only'] += 1
            elif 'LOF' in detected_by and 'GAE' in detected_by:
                detection_counts['Both Methods'] += 1
        
        fig = px.pie(
            values=list(detection_counts.values()),
            names=list(detection_counts.keys()),
            title="Anomaly Detection Methods"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Anomaly Score Distribution")
        
        scores = [graph.nodes[n].get('anomaly_score', 0) for n in anomaly_nodes]
        
        if scores:
            fig = px.histogram(
                x=scores,
                nbins=30,
                title="Distribution of Anomaly Scores",
                labels={'x': 'Anomaly Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top anomalies table
    st.subheader("🔝 Top 10 Most Suspicious Accounts")
    
    anomaly_data = []
    for node in anomaly_nodes:
        node_data = graph.nodes[node]
        anomaly_data.append({
            'Account ID': node,
            'Anomaly Score': node_data.get('anomaly_score', 0),
            'LOF Score': node_data.get('lof_score', 0),
            'GAE Score': node_data.get('gae_score', 0),
            'Detected By': node_data.get('detected_by', 'N/A'),
            'PageRank': node_data.get('pagerank', 0)
        })
    
    df_anomalies = pd.DataFrame(anomaly_data)
    df_anomalies = df_anomalies.sort_values('Anomaly Score', ascending=False).head(10)
    st.dataframe(df_anomalies, use_container_width=True)


def visualization_page(graph: nx.Graph, user: Dict):
    """Graph visualization page."""
    st.header("🔍 Graph Visualization")
    
    AccessControl.log_access(user, 'VIEW', 'Graph Visualization')
    
    st.markdown("""
    Explore the transaction network with interactive visualization.
    Red nodes indicate detected anomalies, blue nodes are normal accounts.
    """)
    
    # Visualization options
    with st.expander("⚙️ Visualization Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highlight_anomalies = st.checkbox("Highlight Anomalies", value=True)
        with col2:
            show_labels = st.checkbox("Show Labels", value=False)
        with col3:
            layout = st.selectbox("Layout Algorithm", ['spring', 'circular', 'kamada_kawai'])
    
    # Sample graph if too large
    display_graph = graph
    if graph.number_of_nodes() > MAX_NODES_DISPLAY:
        st.warning(f"⚠️ Graph is large ({graph.number_of_nodes()} nodes). Sampling for visualization...")
        
        # Sample anomaly nodes + random normal nodes
        anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
        normal_nodes = [n for n, d in graph.nodes(data=True) if not d.get('is_anomaly', False)]
        
        sample_size = min(MAX_NODES_DISPLAY - len(anomaly_nodes), len(normal_nodes))
        sampled_normal = np.random.choice(normal_nodes, sample_size, replace=False)
        
        sampled_nodes = list(anomaly_nodes) + list(sampled_normal)
        display_graph = graph.subgraph(sampled_nodes).copy()
    
    # Create and display graph
    with st.spinner("Generating graph visualization..."):
        fig = GraphVisualizer.create_interactive_graph(
            display_graph,
            highlight_anomalies=highlight_anomalies,
            show_labels=show_labels,
            layout=layout
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Graph statistics
    st.markdown("---")
    st.subheader("📊 Graph Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", f"{display_graph.number_of_nodes():,}")
    with col2:
        st.metric("Edges", f"{display_graph.number_of_edges():,}")
    with col3:
        density = nx.density(display_graph)
        st.metric("Density", f"{density:.4f}")
    with col4:
        avg_degree = sum(dict(display_graph.degree()).values()) / display_graph.number_of_nodes()
        st.metric("Avg Degree", f"{avg_degree:.2f}")


def investigation_page(graph: nx.Graph, user: Dict):
    """Node investigation page with subgraph isolation."""
    st.header("🎯 Node Investigation & Subgraph Isolation")
    
    AccessControl.log_access(user, 'ANALYZE', 'Node Investigation')
    
    st.markdown("""
    Investigate specific nodes and their transaction patterns.
    Select a node to view its local neighborhood and detailed information.
    """)
    
    # Node selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        node_list = sorted(list(graph.nodes()))
        selected_node = st.selectbox("Select Account to Investigate", node_list)
    
    with col2:
        radius = st.slider("Subgraph Radius (hops)", 1, 5, 2)
    
    if selected_node:
        node_data = graph.nodes[selected_node]
        
        # Node information
        st.markdown("---")
        st.subheader(f"📋 Account Information: {selected_node}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            is_anomaly = node_data.get('is_anomaly', False)
            status = "🔴 ANOMALOUS" if is_anomaly else "🟢 NORMAL"
            st.markdown(f"**Status:** {status}")
            st.metric("Anomaly Score", f"{node_data.get('anomaly_score', 0):.3f}")
        
        with col2:
            st.metric("LOF Score", f"{node_data.get('lof_score', 0):.3f}")
            st.metric("GAE Score", f"{node_data.get('gae_score', 0):.3f}")
        
        with col3:
            st.metric("PageRank", f"{node_data.get('pagerank', 0):.6f}")
            st.metric("Degree", f"{graph.degree(selected_node)}")
        
        # Detected by
        if is_anomaly:
            detected_by = node_data.get('detected_by', 'N/A')
            st.info(f"🔍 **Detected by:** {detected_by}")
        
        # Subgraph visualization
        st.markdown("---")
        st.subheader(f"🕸️ Local Network ({radius}-hop neighborhood)")
        
        with st.spinner("Extracting subgraph..."):
            fig = GraphVisualizer.create_subgraph_visualization(graph, selected_node, radius)
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction details
        st.markdown("---")
        st.subheader("💸 Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Incoming Transactions**")
            in_edges = list(graph.in_edges(selected_node, data=True))
            in_data = []
            for u, v, data in in_edges[:10]:  # Limit to 10
                in_data.append({
                    'From': u,
                    'Amount': data.get('amount', 'N/A'),
                    'Timestamp': data.get('timestamp', 'N/A')
                })
            if in_data:
                st.dataframe(pd.DataFrame(in_data), use_container_width=True)
            else:
                st.info("No incoming transactions")
        
        with col2:
            st.markdown("**Outgoing Transactions**")
            out_edges = list(graph.out_edges(selected_node, data=True))
            out_data = []
            for u, v, data in out_edges[:10]:  # Limit to 10
                out_data.append({
                    'To': v,
                    'Amount': data.get('amount', 'N/A'),
                    'Timestamp': data.get('timestamp', 'N/A')
                })
            if out_data:
                st.dataframe(pd.DataFrame(out_data), use_container_width=True)
            else:
                st.info("No outgoing transactions")


def temporal_analysis_page(graph: nx.Graph, user: Dict):
    """Temporal analysis page."""
    st.header("⏱️ Temporal Analysis")
    
    AccessControl.log_access(user, 'ANALYZE', 'Temporal Analysis')
    
    st.markdown("""
    Analyze transaction patterns over time and identify temporal anomalies.
    """)
    
    # Check if timestamps are available
    has_timestamps = any('timestamp' in data for _, _, data in graph.edges(data=True))
    
    if not has_timestamps:
        st.warning("⚠️ No timestamp data available in the graph. Temporal analysis is limited.")
        return
    
    # Date range selection
    st.subheader("📅 Select Time Range")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    if st.button("🔍 Analyze Time Range"):
        with st.spinner("Extracting temporal subgraph..."):
            temporal_graph = TemporalAnalysis.extract_temporal_subgraph(
                graph, start_datetime, end_datetime
            )
            
            st.success(f"✅ Found {temporal_graph.number_of_nodes()} nodes and {temporal_graph.number_of_edges()} edges in time range")
            
            # Temporal patterns
            st.markdown("---")
            st.subheader("📊 Temporal Patterns")
            
            df_temporal = TemporalAnalysis.analyze_temporal_patterns(temporal_graph)
            
            if not df_temporal.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Anomalies by hour
                    hour_counts = df_temporal['hour'].value_counts().sort_index()
                    fig = px.bar(
                        x=hour_counts.index,
                        y=hour_counts.values,
                        title="Anomalies by Hour of Day",
                        labels={'x': 'Hour', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Anomalies by day of week
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    day_counts = df_temporal['day_of_week'].value_counts().sort_index()
                    fig = px.bar(
                        x=[day_names[i] for i in day_counts.index],
                        y=day_counts.values,
                        title="Anomalies by Day of Week",
                        labels={'x': 'Day', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Timeline
                st.markdown("---")
                st.subheader("📈 Anomaly Timeline")
                
                date_counts = df_temporal['date'].value_counts().sort_index()
                fig = px.line(
                    x=date_counts.index,
                    y=date_counts.values,
                    title="Anomalies Over Time",
                    labels={'x': 'Date', 'y': 'Number of Anomalies'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No temporal data available for anomalies")


def explainability_page(graph: nx.Graph, user: Dict):
    """Explainability page with SHAP-style explanations."""
    st.header("📈 Explainability & Feature Analysis")
    
    AccessControl.log_access(user, 'ANALYZE', 'Explainability')
    
    st.markdown("""
    Understand why specific accounts were flagged as anomalous.
    View feature contributions and risk factors.
    """)
    
    # Select anomalous node
    anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
    
    if not anomaly_nodes:
        st.warning("⚠️ No anomalies detected in the graph")
        return
    
    selected_node = st.selectbox("Select Anomalous Account", anomaly_nodes)
    
    if selected_node:
        # Generate explanation
        with st.spinner("Generating explanation..."):
            explanation = ExplainabilityEngine.generate_node_explanation(graph, selected_node)
        
        # Display explanation
        st.markdown("---")
        st.subheader(f"🔍 Explanation for Account {selected_node}")
        
        # Status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Anomaly Score", f"{explanation['anomaly_score']:.3f}")
        with col2:
            st.metric("Detection Methods", ', '.join(explanation['detection_methods']))
        with col3:
            risk_level = "🔴 HIGH" if explanation['anomaly_score'] > 0.8 else "🟡 MEDIUM" if explanation['anomaly_score'] > 0.6 else "🟢 LOW"
            st.markdown(f"**Risk Level:** {risk_level}")
        
        # Raw explanation
        st.markdown("---")
        st.subheader("📝 Automated Explanation")
        st.info(explanation['raw_explanation'])
        
        # Feature contributions
        st.markdown("---")
        st.subheader("📊 Feature Contributions")
        
        fig = ExplainabilityEngine.create_explanation_chart(explanation)
        st.plotly_chart(fig, use_container_width=True)
        
        # Behavioral patterns
        if explanation['behavioral_patterns']:
            st.markdown("---")
            st.subheader("🎯 Behavioral Patterns")
            for pattern in explanation['behavioral_patterns']:
                st.markdown(f"• {pattern}")
        
        # Risk factors
        if explanation['risk_factors']:
            st.markdown("---")
            st.subheader("⚠️ Risk Factors")
            for factor in explanation['risk_factors']:
                st.markdown(f"• {factor}")
        
        # Comparison with normal nodes
        st.markdown("---")
        st.subheader("📈 Comparison with Normal Accounts")
        
        normal_nodes = [n for n, d in graph.nodes(data=True) if not d.get('is_anomaly', False)]
        
        if normal_nodes:
            sample_normal = np.random.choice(normal_nodes, min(100, len(normal_nodes)), replace=False)
            
            comparison_data = {
                'Account Type': ['Selected (Anomaly)'] + ['Normal'] * len(sample_normal),
                'Anomaly Score': [explanation['anomaly_score']] + [graph.nodes[n].get('anomaly_score', 0) for n in sample_normal],
                'PageRank': [graph.nodes[selected_node].get('pagerank', 0)] + [graph.nodes[n].get('pagerank', 0) for n in sample_normal]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            
            fig = px.box(df_comparison, x='Account Type', y='Anomaly Score', 
                        title='Anomaly Score Comparison')
            st.plotly_chart(fig, use_container_width=True)


def reports_page(graph: nx.Graph, user: Dict):
    """Report generation and export page."""
    st.header("📄 Report Generation & Export")
    
    AccessControl.log_access(user, 'EXPORT', 'Reports')
    
    st.markdown("""
    Generate comprehensive investigation reports and export data.
    """)
    
    # Report options
    st.subheader("📋 Report Configuration")
    
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
    
    include_explanations = st.checkbox("Include Explanations for Top Anomalies", value=True)
    
    # Generate report
    if st.button("📊 Generate Report"):
        with st.spinner("Generating report..."):
            # Collect statistics
            anomaly_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_anomaly', False)]
            
            statistics = {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'num_anomalies': len(anomaly_nodes),
                'anomaly_percentage': len(anomaly_nodes) / graph.number_of_nodes() * 100,
                'detection_breakdown': {
                    'lof_only': len([n for n in anomaly_nodes if 'LOF' in graph.nodes[n].get('detected_by', '') and 'GAE' not in graph.nodes[n].get('detected_by', '')]),
                    'gae_only': len([n for n in anomaly_nodes if 'GAE' in graph.nodes[n].get('detected_by', '') and 'LOF' not in graph.nodes[n].get('detected_by', '')]),
                    'both': len([n for n in anomaly_nodes if 'LOF' in graph.nodes[n].get('detected_by', '') and 'GAE' in graph.nodes[n].get('detected_by', '')])
                }
            }
            
            # Generate text report
            report_text = ReportGenerator.generate_text_report(graph, {}, statistics)
            
            st.success("✅ Report generated successfully!")
            
            # Display report preview
            st.markdown("---")
            st.subheader("📄 Report Preview")
            st.text_area("Report Content", report_text, height=400)
            
            # Download buttons
            st.markdown("---")
            st.subheader("💾 Download Report")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download Text Report",
                    data=report_text,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Generate CSV
                df_export = ReportGenerator.export_to_csv(graph, 'temp.csv')
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"anomaly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Generate JSON
                json_data = {
                    'report_id': hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:16],
                    'generated_at': datetime.utcnow().isoformat(),
                    'generated_by': user['username'],
                    'statistics': statistics
                }
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"anomaly_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            AccessControl.log_access(user, 'EXPORT', 'Report Generated', {
                'report_type': report_type,
                'format': export_format,
                'num_anomalies': len(anomaly_nodes)
            })


def audit_log_page(user: Dict):
    """Audit log viewer (admin only)."""
    st.header("🔒 Audit Log")
    
    AccessControl.log_access(user, 'VIEW', 'Audit Log')
    
    st.markdown("""
    View system access and activity logs with cryptographic verification.
    """)
    
    # Load audit log
    try:
        audit_entries = []
        with open('gui_access_audit.json', 'r') as f:
            for line in f:
                try:
                    audit_entries.append(json.loads(line.strip()))
                except:
                    pass
        
        if not audit_entries:
            st.info("No audit entries found")
            return
        
        df_audit = pd.DataFrame(audit_entries)
        
        # Filters
        st.subheader("🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_user = st.multiselect("User", df_audit['username'].unique())
        with col2:
            filter_action = st.multiselect("Action", df_audit['action'].unique())
        with col3:
            filter_resource = st.multiselect("Resource", df_audit['resource'].unique())
        
        # Apply filters
        filtered_df = df_audit
        if filter_user:
            filtered_df = filtered_df[filtered_df['username'].isin(filter_user)]
        if filter_action:
            filtered_df = filtered_df[filtered_df['action'].isin(filter_action)]
        if filter_resource:
            filtered_df = filtered_df[filtered_df['resource'].isin(filter_resource)]
        
        # Display audit log
        st.markdown("---")
        st.subheader(f"📋 Audit Entries ({len(filtered_df)} records)")
        
        st.dataframe(
            filtered_df[['timestamp', 'username', 'role', 'action', 'resource', 'hash']].sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Export audit log
        st.markdown("---")
        csv_audit = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Audit Log (CSV)",
            data=csv_audit,
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except FileNotFoundError:
        st.warning("⚠️ Audit log file not found")
    except Exception as e:
        st.error(f"Error loading audit log: {e}")


# Entry point
if __name__ == "__main__":
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    
    main_application()


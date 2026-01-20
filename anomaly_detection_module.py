# anomaly_detection_module.py

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, confusion_matrix, matthews_corrcoef,
    roc_curve, precision_recall_curve, classification_report
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import Dict, Tuple, List, Optional
import json
import hashlib
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Configure logging with audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder for learning node embeddings and detecting anomalies.
    
    Architecture:
    - Encoder: 2-layer GCN that compresses node features
    - Decoder: Reconstructs node features
    
    Hyperparameters:
    - input_dim: Dimension of input features
    - hidden_dim: Hidden layer dimension (default: 32)
    - embedding_dim: Embedding dimension (default: 16)
    - dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, embedding_dim: int = 16, dropout: float = 0.1):
        super(GraphAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Encoder layers (GCN)
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, embedding_dim)
        
        # Decoder layers
        self.decoder = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"GAE Architecture: input={input_dim}, hidden={hidden_dim}, embedding={embedding_dim}, dropout={dropout}")

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes into embeddings using graph convolutions.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            z: Node embeddings [num_nodes, embedding_dim]
        """
        # First GCN layer: A * X * W1
        h = torch.matmul(adj, x)
        h = self.encoder1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second GCN layer: A * H * W2
        h = torch.matmul(adj, h)
        z = self.encoder2(h)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings back to node features.
        
        Args:
            z: Node embeddings [num_nodes, embedding_dim]
        
        Returns:
            x_reconstructed: Reconstructed features [num_nodes, input_dim]
        """
        h = self.decoder(z)
        h = F.relu(h)
        x_reconstructed = self.decoder_out(h)
        
        return x_reconstructed
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Returns:
            z: Node embeddings
            x_reconstructed: Reconstructed node features
        """
        z = self.encode(x, adj)
        x_reconstructed = self.decode(z)
        
        return z, x_reconstructed


class AnomalyDetectionModule:
    """
    Main Anomaly Detection Module with comprehensive evaluation.
    
    Includes:
    - LOF and GAE detection methods
    - Baseline comparisons (Isolation Forest, One-Class SVM, Statistical)
    - Ablation study (LOF alone, GAE alone, Combined)
    - Full evaluation metrics
    - Chain of custody and reproducibility
    """
    
    VERSION = "2.0.0"
    
    def __init__(self,
                 lof_neighbors: int = 20,
                 lof_contamination: float = 0.1,
                 gae_hidden_dim: int = 32,
                 gae_embedding_dim: int = 16,
                 gae_epochs: int = 100,
                 anomaly_threshold: float = 0.7,
                 fusion_weights: Tuple[float, float] = (0.5, 0.5),
                 random_seed: int = RANDOM_SEED,
                 enable_cross_validation: bool = False,
                 n_folds: int = 5,
                 cost_fp: float = 1.0,
                 cost_fn: float = 10.0):
        """
        Initialize the Anomaly Detection Module.
        """
        # Store parameters as instance attributes
        self.lof_neighbors = lof_neighbors
        self.lof_contamination = lof_contamination
        self.gae_hidden_dim = gae_hidden_dim
        self.gae_embedding_dim = gae_embedding_dim
        self.gae_epochs = gae_epochs
        self.anomaly_threshold = anomaly_threshold
        self.fusion_weights = fusion_weights
        self.random_seed = random_seed
        self.enable_cross_validation = enable_cross_validation
        self.n_folds = n_folds
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        
        # Initialize detectors (will be created during processing)
        self.lof_model = None
        self.gae_model = None
        self.scaler = StandardScaler()
        
        # Results storage
        self.evaluation_results = {}
        self.ablation_results = {}
        self.baseline_results = {}
        
        # Chain of custody
        self.custody_chain = []
        self._log_custody_event('Module Initialized', {
            'version': self.VERSION,
            'lof_neighbors': lof_neighbors,
            'gae_hidden_dim': gae_hidden_dim,
            'anomaly_threshold': anomaly_threshold,
            'random_seed': random_seed
        })
        
        logger.info(f"AnomalyDetectionModule v{self.VERSION} initialized")
    
    def _log_custody_event(self, event: str, metadata: Dict) -> None:
        """Log chain-of-custody event with cryptographic hash."""
        timestamp = datetime.utcnow().isoformat()
        event_data = {
            'timestamp': timestamp,
            'event': event,
            'metadata': metadata
        }
        
        event_str = json.dumps(event_data, sort_keys=True)
        event_hash = hashlib.sha256(event_str.encode()).hexdigest()
        event_data['hash'] = event_hash
        
        self.custody_chain.append(event_data)
        logger.info(f"Custody Event: {event} | Hash: {event_hash[:16]}...")
    
    def _prepare_data(self, graph: nx.Graph) -> Tuple[np.ndarray, List]:
        """
        Extract features from graph nodes.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Tuple of (feature_matrix, node_list)
        """
        logger.info("  ├─ Extracting node features from graph...")
        
        node_list = list(graph.nodes())
        features_list = []
        
        for node in node_list:
            node_data = graph.nodes[node]
            
            # Extract available features
            feature_vector = []
            
            # Graph-based features
            feature_vector.append(graph.degree(node))  # Degree
            feature_vector.append(node_data.get('degree_centrality', 0))
            feature_vector.append(node_data.get('betweenness_centrality', 0))
            feature_vector.append(node_data.get('closeness_centrality', 0))
            feature_vector.append(node_data.get('pagerank', 0))
            feature_vector.append(node_data.get('clustering_coefficient', 0))
            
            # Transaction-based features
            feature_vector.append(node_data.get('total_sent', 0))
            feature_vector.append(node_data.get('total_received', 0))
            feature_vector.append(node_data.get('num_transactions', 0))
            feature_vector.append(node_data.get('avg_transaction_amount', 0))
            
            # Temporal features (if available)
            feature_vector.append(node_data.get('transaction_frequency', 0))
            feature_vector.append(node_data.get('time_span', 0))
            
            features_list.append(feature_vector)
        
        X = np.array(features_list, dtype=np.float32)
        
        # Handle NaN and Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        logger.info(f"  │   ├─ Feature matrix shape: {X.shape}")
        logger.info(f"  │   └─ Features per node: {X.shape[1]}")
        
        return X, node_list
    
    def _run_lof(self, X: np.ndarray) -> np.ndarray:
        """
        Run Local Outlier Factor detection.
        
        Args:
            X: Feature matrix
            
        Returns:
            LOF anomaly scores (higher = more anomalous)
        """
        logger.info(f"  ├─ Training LOF with n_neighbors={self.lof_neighbors}...")
        
        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors,
            contamination=self.lof_contamination,
            novelty=False
        )
        
        # Fit and predict
        predictions = self.lof_model.fit_predict(X)
        
        # Get negative outlier factor and convert to anomaly score
        lof_scores_raw = -self.lof_model.negative_outlier_factor_
        
        # Normalize to [0, 1]
        lof_scores = (lof_scores_raw - lof_scores_raw.min()) / (lof_scores_raw.max() - lof_scores_raw.min() + 1e-8)
        
        logger.info(f"  │   ├─ LOF scores: min={lof_scores.min():.3f}, max={lof_scores.max():.3f}, mean={lof_scores.mean():.3f}")
        
        return lof_scores
    
    def _run_gae(self, graph: nx.Graph, X: np.ndarray, node_list: List) -> np.ndarray:
        """
        Run Graph Autoencoder detection.
        
        Args:
            graph: NetworkX graph
            X: Feature matrix
            node_list: List of node identifiers
            
        Returns:
            GAE anomaly scores (higher = more anomalous)
        """
        logger.info(f"  ├─ Training GAE with hidden_dim={self.gae_hidden_dim}, embedding_dim={self.gae_embedding_dim}...")
        logger.info(f"  ├─ Training for {self.gae_epochs} epochs...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        
        # Build adjacency matrix
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        adj_matrix = np.zeros((len(node_list), len(node_list)))
        
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # Add self-loops and normalize
        adj_matrix = adj_matrix + np.eye(len(node_list))
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        adj_tensor = torch.FloatTensor(adj_normalized)
        
        # Initialize and train GAE
        self.gae_model = GraphAutoencoder(
            input_dim=X.shape[1],
            hidden_dim=self.gae_hidden_dim,
            embedding_dim=self.gae_embedding_dim
        )
        
        optimizer = torch.optim.Adam(self.gae_model.parameters(), lr=0.01)
        
        self.gae_model.train()
        for epoch in range(self.gae_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.gae_model.encode(X_tensor, adj_tensor)
            reconstructed = self.gae_model.decode(embeddings)
            
            # Compute loss (reconstruction error)
            loss = torch.nn.functional.mse_loss(reconstructed, X_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  │   ├─ Epoch {epoch+1}/{self.gae_epochs}, Loss: {loss.item():.4f}")
        
        # Compute anomaly scores based on reconstruction error
        self.gae_model.eval()
        with torch.no_grad():
            embeddings = self.gae_model.encode(X_tensor, adj_tensor)
            reconstructed = self.gae_model.decode(embeddings)
            
            # Reconstruction error per node
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            
            # Normalize to [0, 1]
            gae_scores = (reconstruction_errors - reconstruction_errors.min()) / \
                        (reconstruction_errors.max() - reconstruction_errors.min() + 1e-8)
        
        logger.info(f"  │   ├─ GAE scores: min={gae_scores.min():.3f}, max={gae_scores.max():.3f}, mean={gae_scores.mean():.3f}")
        
        return gae_scores
    
    def _fuse_scores(self, lof_scores: np.ndarray, gae_scores: np.ndarray) -> np.ndarray:
        """
        Fuse LOF and GAE scores using weighted average.
        
        Args:
            lof_scores: LOF anomaly scores
            gae_scores: GAE anomaly scores
            
        Returns:
            Fused anomaly scores
        """
        w_lof, w_gae = self.fusion_weights
        fused = w_lof * lof_scores + w_gae * gae_scores
        
        logger.info(f"  │   ├─ Fusion formula: {w_lof:.2f} * LOF + {w_gae:.2f} * GAE")
        logger.info(f"  │   └─ Fused scores: min={fused.min():.3f}, max={fused.max():.3f}, mean={fused.mean():.3f}")
        
        return fused
    
    def _mark_anomalies(self, 
                       graph: nx.Graph, 
                       node_list: List,
                       lof_scores: np.ndarray,
                       gae_scores: np.ndarray,
                       fused_scores: np.ndarray,
                       is_anomaly: np.ndarray) -> nx.Graph:
        """
        Mark nodes in graph with anomaly scores and labels.
        
        Args:
            graph: NetworkX graph
            node_list: List of nodes
            lof_scores: LOF scores
            gae_scores: GAE scores
            fused_scores: Fused scores
            is_anomaly: Boolean anomaly labels
            
        Returns:
            Marked graph
        """
        marked_graph = graph.copy()
        
        for idx, node in enumerate(node_list):
            marked_graph.nodes[node]['lof_score'] = float(lof_scores[idx])
            marked_graph.nodes[node]['gae_score'] = float(gae_scores[idx])
            marked_graph.nodes[node]['anomaly_score'] = float(fused_scores[idx])
            marked_graph.nodes[node]['is_anomaly'] = bool(is_anomaly[idx])
            
            # Determine detection method
            lof_detected = lof_scores[idx] > self.anomaly_threshold
            gae_detected = gae_scores[idx] > self.anomaly_threshold
            
            if lof_detected and gae_detected:
                marked_graph.nodes[node]['detected_by'] = 'Both'
            elif lof_detected:
                marked_graph.nodes[node]['detected_by'] = 'LOF'
            elif gae_detected:
                marked_graph.nodes[node]['detected_by'] = 'GAE'
            else:
                marked_graph.nodes[node]['detected_by'] = 'None'
        
        logger.info(f"  │   └─ Marked {len(node_list)} nodes with anomaly information")
        
        return marked_graph
    
    def _evaluate(self, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray,
                 scores: np.ndarray) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            scores: Anomaly scores
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure binary labels
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC (requires scores, not binary predictions)
        try:
            auc_roc = roc_auc_score(y_true, scores)
        except:
            auc_roc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        results = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc_roc': float(auc_roc),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'classification_report': classification_report(y_true, y_pred, zero_division=0)
        }
        
        return results
    
    def _get_detection_breakdown(self, 
                                lof_scores: np.ndarray, 
                                gae_scores: np.ndarray) -> Dict:
        """
        Get breakdown of how anomalies were detected.
        
        Args:
            lof_scores: LOF anomaly scores
            gae_scores: GAE anomaly scores
            
        Returns:
            Dictionary with detection breakdown
        """
        lof_detected = lof_scores > self.anomaly_threshold
        gae_detected = gae_scores > self.anomaly_threshold
        
        both = np.sum(lof_detected & gae_detected)
        lof_only = np.sum(lof_detected & ~gae_detected)
        gae_only = np.sum(~lof_detected & gae_detected)
        
        return {
            'both': int(both),
            'lof_only': int(lof_only),
            'gae_only': int(gae_only)
        }
    
    def _run_baseline_comparisons(self, 
                                  X: np.ndarray, 
                                  y_true: Optional[np.ndarray] = None) -> Dict:
        """
        Run baseline detector comparisons.
        
        Args:
            X: Feature matrix
            y_true: Optional ground truth labels
            
        Returns:
            Dictionary of baseline results
        """
        results = {}
        
        # Isolation Forest
        logger.info("  │   ├─ Running Isolation Forest baseline...")
        iso_forest = IsolationForest(contamination=self.lof_contamination, random_state=self.random_seed)
        iso_pred = iso_forest.fit_predict(X)
        iso_pred = (iso_pred == -1).astype(int)  # Convert to binary
        
        if y_true is not None:
            results['isolation_forest'] = self._evaluate(y_true, iso_pred, -iso_forest.score_samples(X))
        
        # One-Class SVM
        logger.info("  │   ├─ Running One-Class SVM baseline...")
        ocsvm = OneClassSVM(nu=self.lof_contamination)
        ocsvm_pred = ocsvm.fit_predict(X)
        ocsvm_pred = (ocsvm_pred == -1).astype(int)  # Convert to binary
        
        if y_true is not None:
            results['one_class_svm'] = self._evaluate(y_true, ocsvm_pred, -ocsvm.score_samples(X))
        
        # Statistical baseline (simple z-score)
        logger.info("  │   └─ Running Statistical baseline...")
        from scipy import stats
        z_scores = np.abs(stats.zscore(X, axis=0))
        stat_scores = np.mean(z_scores, axis=1)
        stat_pred = (stat_scores > 3).astype(int)  # 3-sigma rule
        
        if y_true is not None:
            results['statistical'] = self._evaluate(y_true, stat_pred, stat_scores)
        
        return results
    
    def _run_ablation_study(self, 
                           X: np.ndarray, 
                           graph: nx.Graph,
                           node_list: List,
                           y_true: Optional[np.ndarray] = None) -> Dict:
        """
        Run ablation study to compare individual methods.
        
        Args:
            X: Feature matrix
            graph: NetworkX graph
            node_list: List of nodes
            y_true: Optional ground truth labels
            
        Returns:
            Dictionary of ablation results
        """
        results = {}
        
        # LOF only
        logger.info("  │   ├─ Testing LOF-only detection...")
        lof_scores = self._run_lof(X)
        lof_pred = (lof_scores > self.anomaly_threshold).astype(int)
        
        if y_true is not None:
            results['lof_only'] = self._evaluate(y_true, lof_pred, lof_scores)
        
        # GAE only
        logger.info("  │   ├─ Testing GAE-only detection...")
        gae_scores = self._run_gae(graph, X, node_list)
        gae_pred = (gae_scores > self.anomaly_threshold).astype(int)
        
        if y_true is not None:
            results['gae_only'] = self._evaluate(y_true, gae_pred, gae_scores)
        
        # Fusion
        logger.info("  │   └─ Testing fused detection...")
        fused_scores = self._fuse_scores(lof_scores, gae_scores)
        fused_pred = (fused_scores > self.anomaly_threshold).astype(int)
        
        if y_true is not None:
            results['fusion'] = self._evaluate(y_true, fused_pred, fused_scores)
        
        return results
    
    def process(self,
                graph: nx.Graph,
                y_true: Optional[np.ndarray] = None,
                run_ablation: bool = False,
                run_baselines: bool = False,
                compute_shap: bool = False) -> Tuple[nx.Graph, Dict]:
        """
        Main processing pipeline for anomaly detection.
        
        Pipeline:
        1. Prepare data and extract features
        2. Run LOF detection
        3. Run GAE detection
        4. Fuse scores and mark anomalies
        5. Run baseline comparisons (optional)
        6. Run ablation study (optional)
        7. Generate comprehensive results
        
        Args:
            graph: Input NetworkX graph
            y_true: Optional ground truth labels for evaluation
            run_ablation: Whether to run ablation study
            run_baselines: Whether to run baseline detectors
            compute_shap: Whether to compute SHAP values
            
        Returns:
            Tuple of (marked_graph, results_dict)
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 ANOMALY DETECTION MODULE - PROCESSING PIPELINE")
        logger.info("="*60 + "\n")
        
        # Step 1: Prepare data
        logger.info("📊 Step 1: Preparing graph data and features...")
        X, node_list = self._prepare_data(graph)
        logger.info(f"  └─ ✅ Extracted features for {len(node_list)} nodes (feature dim: {X.shape[1]})\n")
        
        # Step 2: LOF Detection
        logger.info("🔵 Step 2: Running LOF (Local Outlier Factor) detection...")
        lof_scores = self._run_lof(X)
        logger.info(f"  ├─ Mean LOF score: {np.mean(lof_scores):.3f}")
        logger.info(f"  ├─ Max LOF score: {np.max(lof_scores):.3f}")
        logger.info(f"  └─ ✅ LOF detection complete\n")
        
        # Step 3: GAE Detection
        logger.info("🟢 Step 3: Running GAE (Graph Autoencoder) detection...")
        gae_scores = self._run_gae(graph, X, node_list)
        logger.info(f"  ├─ Mean GAE score: {np.mean(gae_scores):.3f}")
        logger.info(f"  ├─ Max GAE score: {np.max(gae_scores):.3f}")
        logger.info(f"  └─ ✅ GAE detection complete\n")
        
        # Step 4: Fuse scores
        logger.info("⚖️ Step 4: Fusing detection scores...")
        fused_scores = self._fuse_scores(lof_scores, gae_scores)
        is_anomaly = fused_scores > self.anomaly_threshold
        num_anomalies = np.sum(is_anomaly)
        logger.info(f"  ├─ Fusion weights: LOF={self.fusion_weights[0]:.2f}, GAE={self.fusion_weights[1]:.2f}")
        logger.info(f"  ├─ Anomaly threshold: {self.anomaly_threshold:.2f}")
        logger.info(f"  ├─ Detected anomalies: {num_anomalies} ({100*num_anomalies/len(node_list):.2f}%)")
        logger.info(f"  └─ ✅ Score fusion complete\n")
        
        # Step 5: Mark graph
        logger.info("🏷️ Step 5: Marking anomalies in graph...")
        marked_graph = self._mark_anomalies(graph, node_list, lof_scores, gae_scores, 
                                            fused_scores, is_anomaly)
        logger.info(f"  └─ ✅ Graph annotated with anomaly scores\n")
        
        # Step 6: Evaluation (if ground truth provided)
        evaluation_results = {}
        if y_true is not None:
            logger.info("📈 Step 6: Evaluating against ground truth...")
            evaluation_results = self._evaluate(y_true, is_anomaly, fused_scores)
            logger.info(f"  ├─ Precision: {evaluation_results['precision']:.3f}")
            logger.info(f"  ├─ Recall: {evaluation_results['recall']:.3f}")
            logger.info(f"  ├─ AUC-ROC: {evaluation_results['auc_roc']:.3f}")
            logger.info(f"  └─ ✅ Evaluation complete\n")
        
        # Step 7: Baseline comparisons (optional)
        baseline_results = {}
        if run_baselines:
            logger.info("🔄 Step 7: Running baseline detector comparisons...")
            baseline_results = self._run_baseline_comparisons(X, y_true)
            if baseline_results:
                logger.info(f"  └─ ✅ Baseline comparison complete\n")
        
        # Step 8: Ablation study (optional)
        ablation_results = {}
        if run_ablation:
            logger.info("🧪 Step 8: Running ablation study...")
            ablation_results = self._run_ablation_study(X, graph, node_list, y_true)
            if ablation_results:
                logger.info(f"  └─ ✅ Ablation study complete\n")
        
        # Step 9: SHAP explanations (optional)
        shap_values = None
        if compute_shap and self.lof_model is not None:
            logger.info("🎯 Step 9: Computing SHAP explanations...")
            try:
                shap_values = self._compute_shap_values(X)
                if shap_values is not None:
                    logger.info(f"  └─ ✅ SHAP values computed for {X.shape[1]} features\n")
            except Exception as e:
                logger.warning(f"  └─ ⚠️ SHAP computation failed: {e}\n")
        
        # Step 10: Prepare results
        logger.info("📦 Step 10: Preparing results package...")
        results = {
            'node_list': node_list,
            'lof_scores': lof_scores,
            'gae_scores': gae_scores,
            'fused_scores': fused_scores,
            'is_anomaly': is_anomaly,
            'statistics': {
                'total_nodes': len(node_list),
                'num_anomalies': int(num_anomalies),
                'anomaly_percentage': 100 * num_anomalies / len(node_list),
                'avg_lof_score': float(np.mean(lof_scores)),
                'avg_gae_score': float(np.mean(gae_scores)),
                'avg_fused_score': float(np.mean(fused_scores)),
                'detection_breakdown': self._get_detection_breakdown(lof_scores, gae_scores)
            },
            'evaluation': evaluation_results,
            'baselines': baseline_results,
            'ablation': ablation_results,
            'shap_values': shap_values,
            'parameters': {
                'lof_neighbors': self.lof_neighbors,
                'lof_contamination': self.lof_contamination,
                'gae_hidden_dim': self.gae_hidden_dim,
                'gae_embedding_dim': self.gae_embedding_dim,
                'gae_epochs': self.gae_epochs,
                'anomaly_threshold': self.anomaly_threshold,
                'fusion_weights': self.fusion_weights
            }
        }
        logger.info(f"  └─ ✅ Results package prepared\n")
        
        # Log custody event
        self._log_custody_event('Anomaly Detection Complete', {
            'num_nodes': len(node_list),
            'num_anomalies': int(num_anomalies),
            'anomaly_rate': float(num_anomalies / len(node_list))
        })
        
        logger.info("="*60)
        logger.info("✅ ANOMALY DETECTION MODULE - PIPELINE COMPLETE")
        logger.info("="*60 + "\n")
        
        return marked_graph, results
    
    def _compute_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute SHAP values for feature importance.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values or None if computation fails
        """
        try:
            import shap
            
            # Create explainer for LOF
            if self.lof_model is not None:
                # Sample data for faster computation
                sample_size = min(100, X.shape[0])
                X_sample = X[np.random.choice(X.shape[0], sample_size, replace=False)]
                
                explainer = shap.KernelExplainer(
                    lambda x: -self.lof_model.score_samples(x),
                    X_sample
                )
                
                shap_values = explainer.shap_values(X_sample)
                
                return shap_values
            
        except Exception as e:
            logger.warning(f"  │   └─ SHAP computation failed: {e}")
            return None
        
        return None
    
    def export_results(self, results: Dict, output_path: str = 'anomaly_detection_results.json') -> None:
        """
        Export detection results to JSON file.
        
        Args:
            results: Results dictionary from process()
            output_path: Output file path
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            export_data = {
                'statistics': results['statistics'],
                'evaluation': results['evaluation'],
                'baselines': results['baselines'],
                'ablation': results['ablation'],
                'parameters': results['parameters'],
                'custody_chain': self.custody_chain,
                'timestamp': datetime.utcnow().isoformat(),
                'version': self.VERSION
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"✅ Results exported to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export results: {e}")
    
    def generate_report(self, results: Dict, output_path: str = 'anomaly_detection_report.txt') -> None:
        """
        Generate human-readable report.
        
        Args:
            results: Results dictionary from process()
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write(" ANOMALY DETECTION REPORT\n")
                f.write("="*80 + "\n\n")
                
                # Statistics
                f.write("📊 DETECTION STATISTICS\n")
                f.write("-"*80 + "\n")
                stats = results['statistics']
                f.write(f"Total Nodes:           {stats['total_nodes']}\n")
                f.write(f"Detected Anomalies:    {stats['num_anomalies']} ({stats['anomaly_percentage']:.2f}%)\n")
                f.write(f"Avg LOF Score:         {stats['avg_lof_score']:.3f}\n")
                f.write(f"Avg GAE Score:         {stats['avg_gae_score']:.3f}\n")
                f.write(f"Avg Fused Score:       {stats['avg_fused_score']:.3f}\n\n")
                
                # Detection breakdown
                breakdown = stats['detection_breakdown']
                f.write("🔍 DETECTION BREAKDOWN\n")
                f.write("-"*80 + "\n")
                f.write(f"Detected by Both:      {breakdown['both']}\n")
                f.write(f"Detected by LOF Only:  {breakdown['lof_only']}\n")
                f.write(f"Detected by GAE Only:  {breakdown['gae_only']}\n\n")
                
                # Evaluation metrics
                if results['evaluation']:
                    eval_results = results['evaluation']
                    f.write("📈 EVALUATION METRICS\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Precision:             {eval_results['precision']:.3f}\n")
                    f.write(f"Recall:                {eval_results['recall']:.3f}\n")
                    f.write(f"F1-Score:              {eval_results['f1']:.3f}\n")
                    f.write(f"AUC-ROC:               {eval_results['auc_roc']:.3f}\n\n")
                    
                    cm = eval_results['confusion_matrix']
                    f.write("Confusion Matrix:\n")
                    f.write(f"  True Negatives:      {cm['tn']}\n")
                    f.write(f"  False Positives:     {cm['fp']}\n")
                    f.write(f"  False Negatives:     {cm['fn']}\n")
                    f.write(f"  True Positives:      {cm['tp']}\n\n")
                
                # Baseline comparisons
                if results['baselines']:
                    f.write("🔄 BASELINE COMPARISONS\n")
                    f.write("-"*80 + "\n")
                    for method, metrics in results['baselines'].items():
                        f.write(f"\n{method.upper()}:\n")
                        f.write(f"  Precision: {metrics['precision']:.3f}\n")
                        f.write(f"  Recall:    {metrics['recall']:.3f}\n")
                        f.write(f"  F1-Score:  {metrics['f1']:.3f}\n")
                        f.write(f"  AUC-ROC:   {metrics['auc_roc']:.3f}\n")
                    f.write("\n")
                
                # Ablation study
                if results['ablation']:
                    f.write("🧪 ABLATION STUDY\n")
                    f.write("-"*80 + "\n")
                    for method, metrics in results['ablation'].items():
                        f.write(f"\n{method.upper()}:\n")
                        f.write(f"  Precision: {metrics['precision']:.3f}\n")
                        f.write(f"  Recall:    {metrics['recall']:.3f}\n")
                        f.write(f"  F1-Score:  {metrics['f1']:.3f}\n")
                        f.write(f"  AUC-ROC:   {metrics['auc_roc']:.3f}\n")
                    f.write("\n")
                
                # Parameters
                f.write("⚙️ CONFIGURATION PARAMETERS\n")
                f.write("-"*80 + "\n")
                params = results['parameters']
                f.write(f"LOF Neighbors:         {params['lof_neighbors']}\n")
                f.write(f"LOF Contamination:     {params['lof_contamination']}\n")
                f.write(f"GAE Hidden Dim:        {params['gae_hidden_dim']}\n")
                f.write(f"GAE Embedding Dim:     {params['gae_embedding_dim']}\n")
                f.write(f"GAE Epochs:            {params['gae_epochs']}\n")
                f.write(f"Anomaly Threshold:     {params['anomaly_threshold']}\n")
                f.write(f"Fusion Weights:        LOF={params['fusion_weights'][0]:.2f}, GAE={params['fusion_weights'][1]:.2f}\n\n")
                
                f.write("="*80 + "\n")
                f.write(f"Report generated: {datetime.utcnow().isoformat()}\n")
                f.write(f"Module version: {self.VERSION}\n")
                f.write("="*80 + "\n")
            
            logger.info(f"✅ Report generated: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate report: {e}")


# Example usage and testing
if __name__ == "__main__":
    # This section is for testing the module independently
    logger.info("🧪 Testing Anomaly Detection Module...")
    
    # Create a simple test graph
    G = nx.karate_club_graph()
    
    # Add some dummy features to nodes
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = nx.degree_centrality(G)[node]
        G.nodes[node]['betweenness_centrality'] = nx.betweenness_centrality(G)[node]
        G.nodes[node]['closeness_centrality'] = nx.closeness_centrality(G)[node]
        G.nodes[node]['pagerank'] = nx.pagerank(G)[node]
        G.nodes[node]['clustering_coefficient'] = nx.clustering(G, node)
        G.nodes[node]['total_sent'] = np.random.rand() * 1000
        G.nodes[node]['total_received'] = np.random.rand() * 1000
        G.nodes[node]['num_transactions'] = np.random.randint(1, 100)
        G.nodes[node]['avg_transaction_amount'] = np.random.rand() * 500
        G.nodes[node]['transaction_frequency'] = np.random.rand()
        G.nodes[node]['time_span'] = np.random.rand() * 365
    
    # Initialize module
    anomaly_module = AnomalyDetectionModule(
        lof_neighbors=5,
        gae_epochs=50,
        anomaly_threshold=0.7,
        fusion_weights=(0.5, 0.5)
    )
    
    # Process graph
    logger.info("🚀 Running anomaly detection pipeline...")
    marked_graph, results = anomaly_module.process(
        graph=G,
        run_ablation=False,
        run_baselines=False,
        compute_shap=False
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("📊 RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total nodes: {results['statistics']['total_nodes']}")
    logger.info(f"Detected anomalies: {results['statistics']['num_anomalies']}")
    logger.info(f"Anomaly percentage: {results['statistics']['anomaly_percentage']:.2f}%")
    logger.info(f"Avg LOF score: {results['statistics']['avg_lof_score']:.3f}")
    logger.info(f"Avg GAE score: {results['statistics']['avg_gae_score']:.3f}")
    logger.info(f"Avg fused score: {results['statistics']['avg_fused_score']:.3f}")
    
    breakdown = results['statistics']['detection_breakdown']
    logger.info(f"\nDetection breakdown:")
    logger.info(f"  Both methods: {breakdown['both']}")
    logger.info(f"  LOF only: {breakdown['lof_only']}")
    logger.info(f"  GAE only: {breakdown['gae_only']}")
    logger.info("="*60 + "\n")
    
    # Export results
    anomaly_module.export_results(results, 'test_results.json')
    anomaly_module.generate_report(results, 'test_report.txt')
    
    logger.info("✅ Testing complete!")




from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')



# Set the correct paths for your project
DATA_DIR = "."
SHAPEFILE_DIR = "Model/LB_shp"
RESULTS_DIR = "Model/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Starting London burglary risk prediction with GCN model...")

# Load LSOA boundaries from borough shapefiles
def load_borough_boundaries():
    print("Loading LSOA boundary data from borough shapefiles...")
    
    # Find all shapefile paths
    shp_files = glob.glob(os.path.join(SHAPEFILE_DIR, "*.shp"))
    
    all_boundaries = []
    for shp_file in shp_files:
        borough_name = os.path.basename(shp_file).replace(".shp", "")
        print(f"Processing {borough_name}...")
        
        # Load the shapefile
        gdf = gpd.read_file(shp_file)
        
        # Check for LSOA code column - using lsoa21cd as in your example
        if 'lsoa21cd' in gdf.columns:
            # Add consistent LSOA_code field
            gdf['LSOA_code'] = gdf['lsoa21cd']
            
            # Add borough name if not present
            if 'borough' not in gdf.columns:
                gdf['borough'] = borough_name
                
            all_boundaries.append(gdf)
            print(f"Added {len(gdf)} LSOAs from {borough_name}")
        else:
            print(f"Warning: {borough_name} shapefile does not contain LSOA information.")
    
    # Combine all borough boundaries
    if all_boundaries:
        london_boundaries = pd.concat(all_boundaries, ignore_index=True)
        print(f"Successfully loaded {len(london_boundaries)} LSOAs across {len(all_boundaries)} boroughs")
        return london_boundaries
    else:
        raise ValueError("No valid borough shapefiles found with LSOA information.")

# Load and preprocess crime data
def load_crime_data():
    print("Loading and preprocessing burglary data...")
    burglary_file = os.path.join(DATA_DIR, 'cleaned_spatial_monthly_burglary_data/2010-12_burglary_cleaned_spatial.csv')
    burglary_data = pd.read_csv(burglary_file)
    
    # Convert date and extract time components
    burglary_data['Month'] = pd.to_datetime(burglary_data['Month'])
    burglary_data['year'] = burglary_data['Month'].dt.year
    burglary_data['month_num'] = burglary_data['Month'].dt.month
    
    # IMPORTANT: Use the correct column name for LSOA code
    # The user specified it's "LSOA code" not "LSOA_code"
    burglary_data.rename(columns={'LSOA code': 'LSOA_code'}, inplace=True)
    
    # Aggregate to monthly counts by LSOA
    monthly_counts = burglary_data.groupby(['LSOA_code', 'year', 'month_num']).size().reset_index(name='burglary_count')
    
    print(f"Processed {len(burglary_data)} burglary incidents into {len(monthly_counts)} monthly LSOA records")
    return monthly_counts

# Load socioeconomic data
def load_socioeconomic_data():
    print("Loading socioeconomic indicators...")
    socio_file = os.path.join(DATA_DIR, 'Societal_wellbeing_dataset/final_merged_cleaned_lsoa_london_social_dataset.csv')
    socio_data = pd.read_csv(socio_file, encoding='latin1', low_memory=False)
    
    # Use the correct column name
    socio_data.rename(columns={'Lower Super Output Area': 'LSOA_code'}, inplace=True)
    
    print(f"Loaded socioeconomic data for {len(socio_data)} LSOAs")
    return socio_data

# Create LSOA features

# Create LSOA features with robust error handling
def create_lsoa_features(monthly_counts, socio_data):
    print("Engineering features for GCN model...")
    
    # Selected socioeconomic features with robust error handling
    selected_features = [
        'Population Density (2011)Persons per hectare',
        'Household Composition (2011);% One person household;',
        'Tenure (2011);Social rented (%)',
        'Tenure (2011);Private rented (%)',
        'Car or van availability (2011);No cars or vans in household (%);',
        'Dwelling type (2011);Flat, maisonette or apartment (%)',
        'Crime (rates);Burglary;2012/13',
        'The Indices of Deprivation 2010;Domains and sub-domains;IMD Score'
    ]
    
    # Ensure column names are consistent and handle potential missing columns
    def safe_get_column(df, column_names):
        for col in column_names:
            if col in df.columns:
                return df[col]
        raise ValueError(f"None of the columns {column_names} found in the dataframe")
    
    # Calculate historical burglary statistics
    lsoa_stats = monthly_counts.groupby('LSOA_code').agg({
        'burglary_count': ['mean', 'std', 'max', 'sum']
    }).reset_index()
    
    # Flatten column hierarchy
    lsoa_stats.columns = ['LSOA_code', 'avg_monthly_burglaries', 'std_burglaries', 
                          'max_burglaries', 'total_burglaries']
    
    # Calculate recent trends (most recent 6 months vs previous 6 months)
    monthly_counts_sorted = monthly_counts.sort_values(['year', 'month_num'], ascending=False)
    
    # Get the most recent 6 months
    recent_data = monthly_counts_sorted.drop_duplicates(['LSOA_code', 'year', 'month_num']).groupby('LSOA_code').head(6)
    recent_avg = recent_data.groupby('LSOA_code')['burglary_count'].mean().reset_index()
    recent_avg.columns = ['LSOA_code', 'recent_avg_burglaries']
    
    # Get previous 6 months
    lsoa_list = recent_data['LSOA_code'].unique()
    previous_data = monthly_counts_sorted[
        ~monthly_counts_sorted.index.isin(recent_data.index) & 
        monthly_counts_sorted['LSOA_code'].isin(lsoa_list)
    ].groupby('LSOA_code').head(6)
    
    previous_avg = previous_data.groupby('LSOA_code')['burglary_count'].mean().reset_index()
    previous_avg.columns = ['LSOA_code', 'previous_avg_burglaries']
    
    # Calculate trend
    trend_data = recent_avg.merge(previous_avg, on='LSOA_code', how='left')
    trend_data['burglary_trend'] = trend_data['recent_avg_burglaries'] - trend_data['previous_avg_burglaries']
    
    # Calculate seasonal patterns
    seasonal = monthly_counts.groupby(['LSOA_code', 'month_num'])['burglary_count'].mean().reset_index()
    
    # Winter months (Nov-Feb)
    winter_months = [11, 12, 1, 2]
    seasonal['is_winter'] = seasonal['month_num'].isin(winter_months)
    winter_rates = seasonal[seasonal['is_winter']].groupby('LSOA_code')['burglary_count'].mean().reset_index()
    winter_rates.columns = ['LSOA_code', 'winter_burglary_rate']
    
    # Summer months (May-Aug)
    summer_months = [5, 6, 7, 8]
    seasonal['is_summer'] = seasonal['month_num'].isin(summer_months)
    summer_rates = seasonal[seasonal['is_summer']].groupby('LSOA_code')['burglary_count'].mean().reset_index()
    summer_rates.columns = ['LSOA_code', 'summer_burglary_rate']
    
    # Extract socioeconomic features with robust column handling
    try:
        socio_features = socio_data[['LSOA_code'] + selected_features].copy()
    except KeyError:
        # If exact column names don't match, try more flexible matching
        socio_features = socio_data.copy()
        socio_features.columns = [col.strip() for col in socio_features.columns]
        
        # Try to find matching columns
        matched_features = []
        for feature in selected_features:
            feature_match = [col for col in socio_features.columns if feature.strip() in col]
            if feature_match:
                matched_features.append(feature_match[0])
        
        if not matched_features:
            raise ValueError("Could not find any matching socioeconomic features")
        
        socio_features = socio_features[['LSOA_code'] + matched_features]
    
    # Merge all features
    features = lsoa_stats.merge(trend_data[['LSOA_code', 'burglary_trend']], on='LSOA_code', how='left')
    features = features.merge(winter_rates, on='LSOA_code', how='left')
    features = features.merge(summer_rates, on='LSOA_code', how='left')
    features = features.merge(socio_features, on='LSOA_code', how='left')
    
    # Calculate seasonal ratio
    features['seasonal_ratio'] = features['winter_burglary_rate'] / features['summer_burglary_rate']
    features.fillna(0, inplace=True)
    
    print(f"Created feature dataset with {len(features)} LSOAs and {len(features.columns)-1} features")
    return features

# Create LSOA adjacency graph with spatial index optimization
def create_adjacency_graph(lsoa_boundaries):
    print("Building LSOA adjacency graph...")
    
    # Ensure unique LSOA codes
    lsoa_boundaries = lsoa_boundaries.drop_duplicates(subset='LSOA_code')
    
    # Create a dictionary to map LSOA codes to indices
    lsoa_codes = lsoa_boundaries['LSOA_code'].values
    
    # Create a robust mapping to ensure sequential indexing
    lsoa_to_idx = {code: idx for idx, code in enumerate(lsoa_codes)}
    
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes
    for code in lsoa_codes:
        G.add_node(lsoa_to_idx[code], lsoa_code=code)
    
    # Use spatial index for efficient neighbor identification
    spatial_index = lsoa_boundaries.sindex
    
    # Track progress
    total = len(lsoa_boundaries)
    edges_added = 0
    
    # Add edges based on adjacency with robust error handling
    for idx, lsoa in lsoa_boundaries.iterrows():
        if idx % 100 == 0:
            print(f"Processing node {idx}/{total}...")
        
        # Find potential neighbors efficiently using spatial index
        try:
            possible_neighbors_idx = list(spatial_index.intersection(lsoa.geometry.bounds))
            possible_neighbors = lsoa_boundaries.iloc[possible_neighbors_idx]
            
            # Filter to actual neighbors (those that share a boundary)
            neighbors = possible_neighbors[possible_neighbors.geometry.touches(lsoa.geometry)]
            
            # Add graph edges
            for _, neighbor in neighbors.iterrows():
                if neighbor['LSOA_code'] != lsoa['LSOA_code']:  # Avoid self-loops
                    idx_i = lsoa_to_idx[lsoa['LSOA_code']]
                    idx_j = lsoa_to_idx[neighbor['LSOA_code']]
                    
                    if not G.has_edge(idx_i, idx_j):
                        G.add_edge(idx_i, idx_j)
                        edges_added += 1
        except Exception as e:
            print(f"Error processing LSOA {lsoa['LSOA_code']}: {e}")
    
    print(f"Created graph with {len(G.nodes)} nodes and {edges_added} edges")
    
    # Convert to PyTorch Geometric format with strict indexing
    edge_index = []
    for edge in G.edges():
        # Add both directions for undirected graph
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])
    
    # Convert to tensor 
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Ensure all indices are within valid range
    max_node = len(lsoa_codes) - 1
    mask = (edge_index_tensor[0] <= max_node) & (edge_index_tensor[1] <= max_node)
    edge_index_filtered = edge_index_tensor[:, mask].contiguous()
    
    print(f"Original edge_index shape: {edge_index_tensor.shape}")
    print(f"Filtered edge_index shape: {edge_index_filtered.shape}")
    
    return G, edge_index_filtered, lsoa_to_idx


class SpatioTemporalConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Layer
    
    This layer combines spatial graph convolutions with temporal convolutions
    to jointly model spatio-temporal dependencies.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpatioTemporalConv, self).__init__()
        
        # Spatial Graph Convolution
        self.spatial_conv = GCNConv(in_channels, out_channels)
        
        # Temporal Convolution (1D conv along the time dimension)
        self.temporal_conv = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size-1) // 2  # Maintain temporal sequence length
        )
        
        # Batch normalization for stabilizing training
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
        # Residual connection to help with gradient flow
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, edge_index, sequence_length):
        """
        Forward pass through Spatio-Temporal Convolutional layer
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape [num_nodes, in_channels * sequence_length]
        edge_index : torch.Tensor
            Graph edge indices of shape [2, num_edges]
        sequence_length : int
            Length of temporal sequence
            
        Returns:
        --------
        torch.Tensor
            Output features of shape [num_nodes, out_channels * sequence_length]
        """
        batch_size = x.size(0)
        feature_size = x.size(1) // sequence_length
        
        # Reshape to separate nodes and temporal sequence
        # [num_nodes, channels * sequence_length] -> [num_nodes, channels, sequence_length]
        x_reshape = x.view(batch_size, feature_size, sequence_length)
        
        # Process each time step with spatial GCN
        outputs = []
        for t in range(sequence_length):
            # Extract features for current time step
            x_t = x_reshape[:, :, t]
            
            # Apply spatial convolution using GCN
            spatial_out = self.spatial_conv(x_t, edge_index)
            
            outputs.append(spatial_out.unsqueeze(-1))
        
        # Combine all time steps
        spatial_out = torch.cat(outputs, dim=-1)  # [num_nodes, out_channels, sequence_length]
        
        # Save residual connection
        res = self.residual(x_reshape)
        
        # Apply temporal convolution across the sequence dimension
        temporal_out = self.temporal_conv(spatial_out)
        
        # Apply batch normalization
        normalized = self.batch_norm(temporal_out)
        
        # Add residual connection
        out = F.relu(normalized + res)
        
        # Reshape back to original format
        # [num_nodes, channels, sequence_length] -> [num_nodes, channels * sequence_length]
        out = out.reshape(batch_size, -1)
        
        return out


class STGCNModel(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for Burglary Prediction
    
    This model captures both spatial dependencies between areas and 
    temporal patterns in crime data within a unified architecture.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, sequence_length=12, num_st_layers=3):
        super(STGCNModel, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Feature dimension per time step
        self.input_dim_per_step = input_dim // sequence_length
        
        # Initial feature embedding layer
        self.embedding = nn.Linear(self.input_dim_per_step, hidden_dim)
        
        # Stack of ST-GCN layers
        self.st_layers = nn.ModuleList()
        
        # First ST-GCN layer after embedding
        self.st_layers.append(SpatioTemporalConv(hidden_dim, hidden_dim))
        
        # Additional ST-GCN layers
        for _ in range(num_st_layers - 1):
            self.st_layers.append(SpatioTemporalConv(hidden_dim, hidden_dim))
        
        # Output layer for prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        """
        Forward pass through the ST-GCN model
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Contains:
                x: Node features [num_nodes, input_dim * sequence_length]
                edge_index: Graph edge indices [2, num_edges]
                
        Returns:
        --------
        torch.Tensor
            Predicted burglary risk [num_nodes, 1]
        """
        x, edge_index = data.x, data.edge_index
        batch_size = x.size(0)
        
        # Ensure edge_index indices are within range
        def filter_edge_index(edge_index, num_nodes):
            mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            return edge_index[:, mask]
        
        edge_index = filter_edge_index(edge_index, x.size(0))
        
        # Reshape input to embed each time step's features
        x_reshaped = x.view(batch_size, self.sequence_length, self.input_dim_per_step)
        
        # Process each time step through initial embedding
        embedded_sequence = []
        for t in range(self.sequence_length):
            x_t = x_reshaped[:, t, :]
            embedded_t = F.relu(self.embedding(x_t))
            embedded_sequence.append(embedded_t.unsqueeze(1))
        
        # Combine embedded features across time
        x = torch.cat(embedded_sequence, dim=1)  # [batch_size, sequence_length, hidden_dim]
        
        # Reshape for the ST-GCN layers: [batch_size, sequence_length * hidden_dim]
        x = x.reshape(batch_size, -1)
        
        # Process through ST-GCN layers
        for st_layer in self.st_layers:
            x = st_layer(x, edge_index, self.sequence_length)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final prediction
        out = self.output_layer(x)
        
        return out
    

class STGCNAttentionModel(nn.Module):
    """
    An enhanced ST-GCN model with attention mechanisms to focus on 
    important spatial and temporal patterns
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, sequence_length=12, num_st_layers=3):
        super(STGCNAttentionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Feature dimension per time step
        self.input_dim_per_step = input_dim // sequence_length
        
        # Initial feature embedding layer
        self.embedding = nn.Linear(self.input_dim_per_step, hidden_dim)
        
        # Stack of ST-GCN layers
        self.st_layers = nn.ModuleList()
        
        # First ST-GCN layer after embedding
        self.st_layers.append(SpatioTemporalConv(hidden_dim, hidden_dim))
        
        # Additional ST-GCN layers
        for _ in range(num_st_layers - 1):
            self.st_layers.append(SpatioTemporalConv(hidden_dim, hidden_dim))
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(sequence_length, sequence_length // 2),
            nn.ReLU(),
            nn.Linear(sequence_length // 2, 1)
        )
        
        # Output layer for prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        """
        Forward pass through the Attention-enhanced ST-GCN model
        """
        x, edge_index = data.x, data.edge_index
        batch_size = x.size(0)
        
        # Ensure edge_index indices are within range
        def filter_edge_index(edge_index, num_nodes):
            mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            return edge_index[:, mask]
        
        edge_index = filter_edge_index(edge_index, x.size(0))
        
        # Reshape input to embed each time step's features
        x_reshaped = x.view(batch_size, self.sequence_length, self.input_dim_per_step)
        
        # Process each time step through initial embedding
        embedded_sequence = []
        for t in range(self.sequence_length):
            x_t = x_reshaped[:, t, :]
            embedded_t = F.relu(self.embedding(x_t))
            embedded_sequence.append(embedded_t.unsqueeze(1))
        
        # Combine embedded features across time
        x = torch.cat(embedded_sequence, dim=1)  # [batch_size, sequence_length, hidden_dim]
        
        # Reshape for the ST-GCN layers: [batch_size, sequence_length * hidden_dim]
        x_st = x.reshape(batch_size, -1)
        
        # Process through ST-GCN layers
        for st_layer in self.st_layers:
            x_st = st_layer(x_st, edge_index, self.sequence_length)
            x_st = F.dropout(x_st, p=0.2, training=self.training)
        
        # Reshape for attention: [batch_size, sequence_length, hidden_dim]
        x_attention = x_st.view(batch_size, self.sequence_length, self.hidden_dim)
        
        # Apply temporal attention (which time steps are most important)
        temporal_weights = self.temporal_attention(x_attention).squeeze(-1)  # [batch_size, sequence_length]
        temporal_weights = F.softmax(temporal_weights, dim=1).unsqueeze(-1)  # [batch_size, sequence_length, 1]
        
        # Weight temporal features by attention
        x_weighted_temporal = x_attention * temporal_weights
        
        # Apply feature attention across the hidden dimension
        x_transposed = x_weighted_temporal.transpose(1, 2)  # [batch_size, hidden_dim, sequence_length]
        feature_weights = self.feature_attention(x_transposed).squeeze(-1)  # [batch_size, hidden_dim]
        feature_weights = F.softmax(feature_weights, dim=1).unsqueeze(-1)  # [batch_size, hidden_dim, 1]
        
        # Weight features by attention
        x_weighted_features = x_transposed * feature_weights
        
        # Aggregate to create the final feature representation
        x_final = torch.sum(x_weighted_features, dim=2)  # [batch_size, hidden_dim]
        
        # Final prediction
        out = self.output_layer(x_final)
        
        return out


# Function to prepare data for the ST-GCN model
def prepare_stgcn_data(lsoa_features, edge_index, monthly_counts, sequence_length=12, target_col='avg_monthly_burglaries'):
    """
    Prepare data specifically for the ST-GCN model, preserving spatio-temporal structure
    
    Parameters:
    -----------
    lsoa_features : pandas.DataFrame
        Features for each LSOA
    edge_index : torch.Tensor
        Graph edge indices of shape [2, num_edges]
    monthly_counts : pandas.DataFrame
        Monthly burglary counts
    sequence_length : int
        Length of temporal sequence to consider
    target_col : str
        Column name for the target variable
        
    Returns:
    --------
    torch_geometric.data.Data
        Data object for the ST-GCN model
    list
        Column names for features
    """
    print("Preparing data for ST-GCN model...")
    
    # Ensure features are in the correct order based on unique LSOA codes
    lsoa_features = lsoa_features.sort_values('LSOA_code')
    
    # Select numerical features
    spatial_feature_cols = [col for col in lsoa_features.columns 
                          if col not in ['LSOA_code', 'LSOA_CODE', 'lsoa_code'] and 
                          lsoa_features[col].dtype in ['int64', 'float64']]
    
    # Normalize spatial features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    lsoa_features[spatial_feature_cols] = scaler.fit_transform(lsoa_features[spatial_feature_cols])
    
    # Create sequences with a focus on preserving temporal patterns for each LSOA
    def create_spatio_temporal_features(features, monthly_data, lsoa_code, sequence_length=12):
        # Sort monthly data chronologically
        monthly_sorted = monthly_data[monthly_data['LSOA_code'] == lsoa_code].sort_values(['year', 'month_num'])
        
        # Extract burglary counts
        burglary_seq = monthly_sorted['burglary_count'].values
        
        # If not enough data, pad with zeros
        if len(burglary_seq) < sequence_length:
            burglary_seq = np.pad(burglary_seq, (sequence_length - len(burglary_seq), 0), mode='constant')
        
        # Take the last sequence_length entries
        burglary_seq = burglary_seq[-sequence_length:]
        
        # Get the static features for this LSOA
        lsoa_static_features = features[features['LSOA_code'] == lsoa_code][spatial_feature_cols].values[0]
        
        # Create features for each time step by combining static features with temporal value
        spatio_temporal_features = []
        for t in range(sequence_length):
            # Combine static features with temporal sequence value
            # This ensures each time step has both spatial context and temporal information
            time_step_features = np.append(lsoa_static_features, burglary_seq[t])
            spatio_temporal_features.append(time_step_features)
        
        # Stack all time steps
        return np.hstack(spatio_temporal_features)
    
    # Prepare sequences for each LSOA
    X_sequences = []
    y_targets = []
    
    # Track unique LSOAs
    unique_lsoas = lsoa_features['LSOA_code'].unique()
    
    import numpy as np
    
    for lsoa_code in unique_lsoas:
        try:
            features = create_spatio_temporal_features(lsoa_features, monthly_counts, lsoa_code, sequence_length)
            target = lsoa_features[lsoa_features['LSOA_code'] == lsoa_code][target_col].values[0]
            
            X_sequences.append(features)
            y_targets.append(target)
        except Exception as e:
            print(f"Error processing LSOA {lsoa_code}: {e}")
    
    # Convert to tensors
    X_features = torch.tensor(X_sequences, dtype=torch.float)
    y_targets = torch.tensor(y_targets, dtype=torch.float).view(-1, 1)
    
    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data
    data = Data(x=X_features, edge_index=edge_index, y=y_targets)
    
    # Calculate feature dimensions
    feature_dim_per_step = len(spatial_feature_cols) + 1  # static features + burglary count
    total_feature_dim = feature_dim_per_step * sequence_length
    
    print(f"Prepared ST-GCN data:")
    print(f"X_features shape: {X_features.shape}")
    print(f"y_targets shape: {y_targets.shape}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"Feature dimensions per time step: {feature_dim_per_step}")
    print(f"Total feature dimensions: {total_feature_dim}")
    
    # Create feature column names for interpretability
    feature_cols = []
    for t in range(sequence_length):
        for col in spatial_feature_cols:
            feature_cols.append(f"{col}_t{t}")
        feature_cols.append(f"burglary_count_t{t}")
    
    return data, feature_cols


# Train function for ST-GCN model
def train_stgcn_model(model, data, test_size=0.2, epochs=150, learning_rate=0.001):
    """
    Train the ST-GCN model with train-validation split
    
    Parameters:
    -----------
    model : torch.nn.Module
        The ST-GCN model
    data : torch_geometric.data.Data
        Data object containing features, graph structure and targets
    test_size : float
        Proportion of data to use for validation
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
        
    Returns:
    --------
    dict
        Training metrics and history
    """
    print("Training ST-GCN model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare train-validation split
    total_samples = data.x.shape[0]
    train_size = int(total_samples * (1 - test_size))
    
    # Create indices for train and validation sets
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Split data
    from torch_geometric.data import Data
    train_data = Data(
        x=data.x[train_indices], 
        edge_index=data.edge_index, 
        y=data.y[train_indices]
    )
    val_data = Data(
        x=data.x[val_indices], 
        edge_index=data.edge_index, 
        y=data.y[val_indices]
    )
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20
        # verbose=True
    )
    
    # Tracking losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_patience = 50
    early_stop_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        train_loss = criterion(out, train_data.y)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = criterion(val_out, val_data.y)
        
        # Step learning rate
        scheduler.step(val_loss)
        
        # Record losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress periodically
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    print(f"Final Training Loss: {train_loss.item():.4f}")
    print(f"Final Validation Loss: {val_loss.item():.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    import numpy as np
    
    # Visualization of training and validation losses
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses for ST-GCN Model')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create a DataFrame for easier analysis
    import pandas as pd
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })
    
    return {
        'model': model,
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'loss_dataframe': loss_df,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'best_val_loss': best_val_loss
    }


# Comparative model analysis function
def compare_model_performance(original_metrics, stgcn_metrics):
    """
    Compare performance between the original hybrid model and the ST-GCN model
    
    Parameters:
    -----------
    original_metrics : dict
        Performance metrics for the original model
    stgcn_metrics : dict
        Performance metrics for the ST-GCN model
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table
    """
    import pandas as pd
    
    # Create comparison dataframe
    metrics = ['Mean Squared Error', 'Mean Absolute Error', 'R-squared', 'Percentage Error']
    comparison = pd.DataFrame({
        'Metric': metrics,
        'Original LSTM-GCN Model': [original_metrics[m] for m in metrics],
        'ST-GCN Model': [stgcn_metrics[m] for m in metrics],
        'Improvement (%)': [
            ((original_metrics[m] - stgcn_metrics[m]) / original_metrics[m]) * 100 
            if m not in ['R-squared'] 
            else ((stgcn_metrics[m] - original_metrics[m]) / original_metrics[m]) * 100
            for m in metrics
        ]
    })
    
    return comparison


def evaluate_model(model, test_data):
    """
    Evaluate the model on test data
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained LSTM-GCN model
    test_data : torch_geometric.data.Data
        Test dataset
    
    Returns:
    --------
    dict: Performance metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():
        # Make predictions
        predicted = model(test_data)
        
        # Convert to numpy for sklearn metrics
        y_true = test_data.y.numpy()
        y_pred = predicted.numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage error
        percentage_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Plotting predicted vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Burglary Risk')
        plt.ylabel('Predicted Burglary Risk')
        plt.title('Predicted vs Actual Burglary Risk')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'predicted_vs_actual_STGCN.png'))
        plt.close()
        
        # Residual plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='r', linestyle='--')
        plt.xlabel('Predicted Burglary Risk')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'residual_plot_STGCN.png'))
        plt.close()
        
        return {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'R-squared': r2,
            'Percentage Error': percentage_error
        }

# Generate risk predictions
def generate_predictions(model, data, lsoa_features):
    print("Generating risk predictions...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate predictions
    with torch.no_grad():
        predicted_risks = model(data).numpy().flatten()
    
    # Create prediction results
    results = lsoa_features[['LSOA_code']].copy()
    results['predicted_risk'] = predicted_risks
    
    # Normalize risk scores to 0-100 scale
    min_risk = results['predicted_risk'].min()
    max_risk = results['predicted_risk'].max()
    results['risk_score'] = ((results['predicted_risk'] - min_risk) / (max_risk - min_risk)) * 100
    
    print(f"Generated risk predictions for {len(results)} LSOAs")
    # Create prediction results
    results = lsoa_features[['LSOA_code']].copy()
    results['predicted_risk'] = predicted_risks
    results['risk_score'] = ((results['predicted_risk'] - min_risk) / (max_risk - min_risk)) * 100
    
    # Save predictions to CSV
    results.to_csv(os.path.join(RESULTS_DIR, 'risk_predictions_STGCN.csv'), index=False)
    print(f"Prediction results saved at {RESULTS_DIR}")
    return results

# Robust risk map creation function
def create_risk_map(boundaries, risk_data, output_file='london_burglary_risk_map_STGCN.png'):
    print("Creating burglary risk heatmap...")
    
    # Merge boundaries with risk predictions
    risk_map = boundaries.merge(risk_data, on='LSOA_code', how='inner')
    
    # Handle risk score categorization more robustly
    def custom_risk_categorization(series):
        """
        Create custom risk categories based on percentiles
        to handle cases with limited unique values
        """
        # Calculate percentiles
        percentiles = [0, 20, 40, 60, 80, 100]
        
        # Compute percentile-based bins
        bins = np.percentile(series, percentiles)
        
        # Ensure unique bin edges by adding a small epsilon
        bins = np.unique(bins)
        if len(bins) < 2:
            # If all values are the same, create a simple range
            bins = [series.min(), series.max()]
        
        # Define labels
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        # Adjust labels if needed
        if len(bins) - 1 != len(labels):
            labels = labels[:len(bins)-1]
        
        return pd.cut(series, 
                     bins=bins, 
                     labels=labels, 
                     include_lowest=True)
    
    # Apply custom categorization
    risk_map['risk_category'] = custom_risk_categorization(risk_map['risk_score'])
    
    # Set up figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot the map with risk colors
    risk_map.plot(column='risk_score', 
                 cmap='RdYlGn_r', 
                 linewidth=0.1,
                 edgecolor='0.5',
                 legend=True,
                 ax=ax)
    
    # Customize the plot
    ax.set_title('Predicted Burglary Risk in London (LSOA Level)\nGraph Convolutional Network Model', fontsize=16)
    ax.set_axis_off()
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r')
    sm.set_array(risk_map['risk_score'])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Risk Score (Higher = Greater Risk)', fontsize=12)
    
    # Add model information
    plt.figtext(0.1, 0.01, 
               "Model: Graph Convolutional Network incorporating spatial relationships between areas\n" +
               "Data period: 2010-2025", 
               ha="left", fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, output_file), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved risk map to {os.path.join(RESULTS_DIR, output_file)}")
    return risk_map

# Updated visualization function to handle potential binning issues
def visualize_borough_risk(risk_map, output_file='borough_risk_summary_STGCN.png'):
    print("Creating borough-level risk summary...")
    
    # Calculate average risk by borough
    borough_risk = risk_map.groupby('borough')['risk_score'].mean().reset_index()
    borough_risk = borough_risk.sort_values('risk_score', ascending=False)
    
    # Create borough-level visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the bar chart
    bars = ax.bar(borough_risk['borough'], borough_risk['risk_score'])
    
    # Customize the plot
    ax.set_xlabel('London Borough', fontsize=12)
    ax.set_ylabel('Average Risk Score', fontsize=12)
    ax.set_title('Average Burglary Risk by London Borough', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add risk values on top of bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{borough_risk.iloc[i]["risk_score"]:.1f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, output_file), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved borough risk summary to {os.path.join(RESULTS_DIR, output_file)}")
    return borough_risk

# Visualize borough-level risk
def visualize_borough_risk(risk_map, output_file='borough_risk_summary_STGCN.png'):
    print("Creating borough-level risk summary...")
    
    # Calculate average risk by borough
    borough_risk = risk_map.groupby('borough')['risk_score'].mean().reset_index()
    borough_risk = borough_risk.sort_values('risk_score', ascending=False)
    
    # Create borough-level visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the bar chart
    bars = ax.bar(borough_risk['borough'], borough_risk['risk_score'])
    
    # Customize the plot
    ax.set_xlabel('London Borough', fontsize=12)
    ax.set_ylabel('Average Risk Score', fontsize=12)
    ax.set_title('Average Burglary Risk by London Borough', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add risk values on top of bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{borough_risk.iloc[i]["risk_score"]:.1f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, output_file), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved borough risk summary to {os.path.join(RESULTS_DIR, output_file)}")
    return borough_risk

# Diagnostic function to help investigate data consistency
def diagnose_lsoa_data_consistency(lsoa_boundaries, monthly_counts, socio_data):
    print("\n--- LSOA Data Consistency Diagnosis ---")
    
    # Check unique LSOA codes in each dataset
    print("Unique LSOA Codes:")
    print(f"LSOA Boundaries: {len(lsoa_boundaries['LSOA_code'].unique())}")
    print(f"Monthly Counts: {len(monthly_counts['LSOA_code'].unique())}")
    print(f"Socio Data: {len(socio_data['LSOA_code'].unique())}")
    
    # Find common LSOA codes
    lsoa_boundaries_codes = set(lsoa_boundaries['LSOA_code'].unique())
    monthly_counts_codes = set(monthly_counts['LSOA_code'].unique())
    socio_data_codes = set(socio_data['LSOA_code'].unique())
    
    # Intersection of codes
    common_codes = lsoa_boundaries_codes.intersection(monthly_counts_codes, socio_data_codes)
    print(f"\nLSOA Codes common to all datasets: {len(common_codes)}")
    
    # Codes unique to each dataset
    print("\nUnique LSOA Codes:")
    print(f"Unique in Boundaries: {len(lsoa_boundaries_codes - monthly_counts_codes - socio_data_codes)}")
    print(f"Unique in Monthly Counts: {len(monthly_counts_codes - lsoa_boundaries_codes - socio_data_codes)}")
    print(f"Unique in Socio Data: {len(socio_data_codes - lsoa_boundaries_codes - monthly_counts_codes)}")
    
    return common_codes


# Modified ST-GCN main function
def main_stgcn():
    """
    Main function to run the ST-GCN model pipeline
    """
    try:
        import os
        
        # Set constants
        RESULTS_DIR = "Model/results"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        print("Starting London burglary risk prediction with ST-GCN model...")
        
        # Load data
        # from load_data import load_borough_boundaries, load_crime_data, load_socioeconomic_data
        lsoa_boundaries = load_borough_boundaries()
        monthly_counts = load_crime_data()
        socio_data = load_socioeconomic_data()
        
        # Diagnose data consistency
        # from utils import diagnose_lsoa_data_consistency
        common_lsoas = diagnose_lsoa_data_consistency(
            lsoa_boundaries, 
            monthly_counts, 
            socio_data
        )
        
        # Ensure data consistency
        lsoa_boundaries = lsoa_boundaries[lsoa_boundaries['LSOA_code'].isin(common_lsoas)]
        socio_data = socio_data[socio_data['LSOA_code'].isin(common_lsoas)]
        monthly_counts = monthly_counts[monthly_counts['LSOA_code'].isin(common_lsoas)]
        
        # Create features and graph structure
        # from feature_engineering import create_lsoa_features, create_adjacency_graph
        lsoa_features = create_lsoa_features(monthly_counts, socio_data)
        G, edge_index, lsoa_to_idx = create_adjacency_graph(lsoa_boundaries)
        
        # Prepare data for ST-GCN
        data, feature_cols = prepare_stgcn_data(
            lsoa_features, 
            edge_index, 
            monthly_counts, 
            sequence_length=12
        )
        
        # Perform train-test split for evaluation
        from torch_geometric.data import Data
        import torch
        
        total_samples = data.x.shape[0]
        indices = torch.randperm(total_samples)
        train_size = int(total_samples * 0.8)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = Data(
            x=data.x[train_indices], 
            edge_index=data.edge_index, 
            y=data.y[train_indices]
        )
        test_data = Data(
            x=data.x[test_indices], 
            edge_index=data.edge_index, 
            y=data.y[test_indices]
        )
        
        # Initialize ST-GCN model
        input_dim = data.x.size(1)
        model = STGCNModel(
            input_dim=input_dim, 
            hidden_dim=64, 
            output_dim=1, 
            sequence_length=12, 
            num_st_layers=3
        )
        
        # Initialize ST-GCN with attention (alternative model)
        attention_model = STGCNAttentionModel(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=1,
            sequence_length=12,
            num_st_layers=3
        )
        
        # Train base ST-GCN model
        print("\nTraining base ST-GCN model...")
        base_results = train_stgcn_model(model, train_data, epochs=150)
        
        # Train attention-enhanced ST-GCN model
        print("\nTraining attention-enhanced ST-GCN model...")
        attention_results = train_stgcn_model(attention_model, train_data, epochs=150, learning_rate=0.005)
        
        # Evaluate models
        # from evaluation import evaluate_model
        
        print("\nEvaluating base ST-GCN model...")
        base_metrics = evaluate_model(model, test_data)
        # Print evaluation metrics
        print("\nModel Performance Metrics:")
        for metric, value in base_metrics.items():
            print(f"{metric}: {value}")
        
        print("\nEvaluating attention-enhanced ST-GCN model...")
        attention_metrics = evaluate_model(attention_model, test_data)
        # Print evaluation metrics
        print("\nModel Performance Metrics:")
        for metric, value in attention_metrics.items():
            print(f"{metric}: {value}")
            
            
        # Generate predictions with best model
        # from visualization import generate_predictions, create_risk_map, visualize_borough_risk
        
        # Choose the best model based on validation performance
        best_model = model if base_results['best_val_loss'] < attention_results['best_val_loss'] else attention_model
        model_name = "Base ST-GCN" if best_model == model else "Attention ST-GCN"
        print(f"\nUsing {model_name} for final predictions (best validation loss)")
        
        
        # Visualize training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(base_results['train_losses'], label='Training Loss', color='blue')
        plt.plot(base_results['val_losses'], label='Validation Loss', color='red')
        plt.title('Training and Validation Losses of STGCN Model')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error Loss')
        plt.ylim(0, 1)  # Set the x-axis range
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'base_results_STGCN.png'))
        plt.close()
        
        
        # Visualize training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(attention_results['train_losses'], label='Training Loss', color='blue')
        plt.plot(attention_results['val_losses'], label='Validation Loss', color='red')
        plt.title('Training and Validation Losses of Attention STGCN Model')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error Loss')
        plt.ylim(0, 1)  # Set the x-axis range
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'attention_results_STGCN.png'))
        plt.close()
        
        # Generate predictions and visualizations
        risk_predictions = generate_predictions(best_model, data, lsoa_features)
        risk_map = create_risk_map(lsoa_boundaries, risk_predictions, output_file='london_burglary_risk_map_STGCN.png')
        borough_summary = visualize_borough_risk(risk_map, output_file='borough_risk_summary_STGCN.png')
        
        print("ST-GCN Burglary Risk Prediction completed successfully!")
        print(f"Visualization outputs saved to {RESULTS_DIR}")
        
        return {
            'base_model': model,
            'attention_model': attention_model,
            'best_model_name': model_name,
            'risk_predictions': risk_predictions,
            'risk_map': risk_map,
            'base_metrics': base_metrics,
            'attention_metrics': attention_metrics
        }
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Execute if run directly
if __name__ == "__main__":
    results = main_stgcn()

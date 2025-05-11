import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import StandardScaler
import glob
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
    burglary_data['LSOA_code'] = burglary_data['LSOA code']
    
    # Aggregate to monthly counts by LSOA
    monthly_counts = burglary_data.groupby(['LSOA_code', 'year', 'month_num']).size().reset_index(name='burglary_count')
    
    print(f"Processed {len(burglary_data)} burglary incidents into {len(monthly_counts)} monthly LSOA records")
    return monthly_counts

# Load socioeconomic data
def load_socioeconomic_data():
    print("Loading socioeconomic indicators...")
    socio_file = os.path.join(DATA_DIR, 'Societal_wellbeing_dataset/final_merged_cleaned_lsoa_london_social_dataset.csv')
    socio_data = pd.read_csv(socio_file, encoding='latin1', low_memory=False)
    
    # Use the correct column name as in your example
    socio_data['LSOA_code'] = socio_data['Lower Super Output Area']
    
    print(f"Loaded socioeconomic data for {len(socio_data)} LSOAs")
    return socio_data

# Create LSOA features
def create_lsoa_features(monthly_counts, socio_data):
    print("Engineering features for GCN model...")
    
    # Selected socioeconomic features
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
    
    # Extract socioeconomic features
    socio_features = socio_data[['LSOA_code'] + selected_features].copy()
    
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
    
    # Create a dictionary to map LSOA codes to indices
    lsoa_codes = lsoa_boundaries['LSOA_code'].values
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
    
    # Add edges based on adjacency
    for idx, lsoa in lsoa_boundaries.iterrows():
        if idx % 100 == 0:
            print(f"Processing node {idx}/{total}...")
            
        # Find potential neighbors efficiently using spatial index
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
    
    print(f"Created graph with {len(G.nodes)} nodes and {edges_added} edges")
    
    # Convert to PyTorch Geometric format
    edge_index = []
    for edge in G.edges():
        # Add both directions for undirected graph
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return G, edge_index, lsoa_to_idx

# Prepare data for GCN
def prepare_gcn_data(lsoa_features, edge_index, target_col='avg_monthly_burglaries'):
    print("Preparing data for GCN model...")
    
    # Select numerical features
    feature_cols = [col for col in lsoa_features.columns 
                   if col != 'LSOA_code' and 
                   lsoa_features[col].dtype in ['int64', 'float64']]
    
    # Normalize features
    scaler = StandardScaler()
    lsoa_features[feature_cols] = scaler.fit_transform(lsoa_features[feature_cols])
    
    # Create feature matrix
    X = lsoa_features[feature_cols].values
    X = torch.tensor(X, dtype=torch.float)
    
    # Create target vector
    y = lsoa_features[target_col].values
    y = torch.tensor(y, dtype=torch.float).view(-1, 1)
    
    # Create PyTorch Geometric Data object
    data = Data(x=X, edge_index=edge_index, y=y)
    
    print(f"Prepared GCN data with {X.shape[1]} features")
    return data, feature_cols

# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, num_features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First graph convolution layer with neighborhood aggregation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second graph convolution layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Third graph convolution layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Final prediction layer
        x = self.fc(x)
        
        return x

# Train the GCN model
def train_gcn_model(model, data, epochs=100):
    print("Training GCN model...")
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    print(f"Final training loss: {loss.item():.4f}")
    return losses

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
    results.to_csv(os.path.join(RESULTS_DIR, 'risk_predictions.csv'), index=False)
    print(f"Prediction results saved at {RESULTS_DIR}")
    return results

# Create risk map visualization
def create_risk_map(boundaries, risk_data, output_file='london_burglary_risk_map.png'):
    print("Creating burglary risk heatmap...")
    
    # Merge boundaries with risk predictions
    risk_map = boundaries.merge(risk_data, on='LSOA_code', how='inner')
    
    # Classify risk into categories
    risk_map['risk_category'] = pd.qcut(risk_map['risk_score'], 
                                       q=5, 
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
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
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Risk Score (Higher = Greater Risk)', fontsize=12)
    
    # Add model information
    plt.figtext(0.1, 0.01, 
               "Model: Graph Convolutional Network incorporating spatial relationships between areas\n" +
               "Data period: 2010-2025", 
               ha="left", fontsize=10)
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, output_file), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved risk map to {os.path.join(RESULTS_DIR, output_file)}")
    return risk_map

# Visualize borough-level risk
def visualize_borough_risk(risk_map, output_file='borough_risk_summary.png'):
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

# Main execution function
def main():
    try:
        # Load data
        lsoa_boundaries = load_borough_boundaries()
        monthly_counts = load_crime_data()
        socio_data = load_socioeconomic_data()
        
        # Ensure data consistency by finding common LSOAs
        common_lsoas = set(lsoa_boundaries['LSOA_code']).intersection(
            set(socio_data['LSOA_code'])).intersection(
            set(monthly_counts['LSOA_code']))
        
        print(f"Found {len(common_lsoas)} LSOAs common to all datasets")
        
        # Filter datasets to include only common LSOAs
        lsoa_boundaries = lsoa_boundaries[lsoa_boundaries['LSOA_code'].isin(common_lsoas)]
        socio_data = socio_data[socio_data['LSOA_code'].isin(common_lsoas)]
        monthly_counts = monthly_counts[monthly_counts['LSOA_code'].isin(common_lsoas)]
        
        # Create features and graph structure
        lsoa_features = create_lsoa_features(monthly_counts, socio_data)
        G, edge_index, lsoa_to_idx = create_adjacency_graph(lsoa_boundaries)
        data, feature_cols = prepare_gcn_data(lsoa_features, edge_index)
        
        # Train GCN model - using fewer epochs for demo
        model = GCNModel(num_features=len(feature_cols))
        losses = train_gcn_model(model, data, epochs=50)
        
        # Generate predictions and visualizations
        risk_predictions = generate_predictions(model, data, lsoa_features)
        risk_map = create_risk_map(lsoa_boundaries, risk_predictions)
        borough_summary = visualize_borough_risk(risk_map)
        
        print("Burglary risk prediction demo completed successfully!")
        print(f"Visualization outputs saved to {RESULTS_DIR}")
        
        # Return results as demonstration artifact
        return {
            'model': model,
            'risk_predictions': risk_predictions,
            'risk_map': risk_map
        }
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Execute if run directly
if __name__ == "__main__":
    results = main()
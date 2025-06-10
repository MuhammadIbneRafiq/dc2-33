# Stadium-Enhanced Police Allocation System

## Quick Start

This enhanced system incorporates football stadium proximity as a risk factor in police resource allocation across London wards.

## What's New

### Stadium Risk Factors
Based on analysis of 3+ million crime records, we identified that stadiums significantly affect local crime patterns:

- **High Risk Areas** (Tottenham, Arsenal): +13-15% more police resources
- **Protective Areas** (Chelsea): Evidence-based resource optimization  
- **Medium Risk Areas** (West Ham): +5% adjustment for precaution

### Key Features
- **Evidence-Based**: Risk factors derived from statistical analysis
- **Distance-Weighted**: Closer to stadium = stronger effect
- **Transparent**: Full explanations for each allocation decision

## Usage

### Run Enhanced Allocation
```python
python Enhanced_PoliceAllocation_Stadium.py
```

### Configuration
Edit these parameters in the script:
- `STADIUM_INFLUENCE_RADIUS_KM = 3.0`  # Stadium effect radius
- Risk multipliers for different stadium types

## Output

The system generates:
1. **Enhanced allocation plan** (`enhanced_ward_allocation_plan.csv`)
2. **Console summary** with key statistics and changes
3. **Detailed explanations** for each allocation decision

## Evidence Base

### Tottenham Area
- **+13.2% burglary increase** during match months
- **Highly significant** (p=0.0147)
- **Strong positive correlation** (r=0.301)

### Chelsea Area  
- **-7.8% burglary decrease** during match months
- **Statistically significant** (p=0.0415)
- **Protective effect** from increased foot traffic/guardianship

### Arsenal Area
- **+9.4% burglary increase** during match months
- **Marginal significance** (p=0.0937)
- **Concerning trend** warranting preventive measures

## How It Works

1. **Load base risk data** (original system)
2. **Calculate stadium proximity** for each ward
3. **Apply evidence-based risk multipliers**
4. **Redistribute police resources** proportionally
5. **Generate allocation plan** with explanations

## Impact

- More targeted policing in high-risk stadium areas
- Evidence-based resource allocation
- Improved crime prevention through strategic deployment
- Transparency in allocation decisions

## Files

- `Enhanced_PoliceAllocation_Stadium.py` - Main enhanced system
- `PoliceAllocation1.py` - Original system for comparison
- This README - Documentation and usage guide 
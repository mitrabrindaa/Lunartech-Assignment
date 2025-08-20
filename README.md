assignment4-federated-dp/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── seed_everything.py                 # Global seed utility
├── data/                              # Generated hospital datasets
│   ├── hospital_A.csv                 # Hospital A data (non-IID)
│   ├── hospital_B.csv                 # Hospital B data (non-IID)
│   └── test_set.csv                   # External test set
├── models/                            # Saved trained models
├── notebooks/
│   └── assignment4_federated_dp.ipynb # Main implementation notebook
├── report/                            # Generated reports
├── results/                           # Experimental results and plots
│   ├── class_distribution_overview.png
│   ├── comprehensive_comparison.png
│   ├── feature_distributions.png
│   ├── Federated_Learning_DP_Report.pdf
│   ├── final_comparison.csv
│   ├── final_summary.txt
│   ├── hospital_split_summary.csv
│   ├── privacy_analysis.csv
│   └── privacy_analysis_plots.png
└── src/                               # Source code modules
    ├── __init__.py                    # Package initialization
    ├── data_processing.py             # Data loading and preprocessing
    ├── dp_client.py                   # Differential privacy client
    ├── federated_client.py            # Standard federated client
    ├── models.py                      # Neural network architectures
    ├── training.py                    # Training orchestration
    └── utils.py                       # Utility functions

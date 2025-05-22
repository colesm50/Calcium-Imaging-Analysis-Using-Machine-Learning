#!/usr/bin/env python3
"""
Suite2p Calcium Imaging Processing Pipeline
===========================================

This script processes calcium imaging data using Suite2p.
It includes registration, cell detection, and signal extraction.

Author: Cole Meador
Email: colesm50@gmail.com
Date: 22.05.2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from suite2p import run_s2p, default_ops
from suite2p import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_suite2p_ops():
    """
    Set up Suite2p operations (parameters) for processing.
    
    This function configures all the key parameters that control how Suite2p
    processes your calcium imaging data.
    
    Returns:
        dict: Dictionary containing all Suite2p operations/parameters
    """
    logger.info("Setting up Suite2p operations...")
    
    # Start with default operations
    ops = default_ops()
    
    # =================================================================
    # DATA INPUT/OUTPUT SETTINGS
    # =================================================================
    
    # Directory containing your raw data files
    ops['data_path'] = [r'C:\Users\c.meador\Desktop\CAIML\Calcium-Imaging-Analysis-Using-Machine-Learning\data']
    
    # Where to save processed results
    ops['save_path0'] = r'C:\Users\c.meador\Desktop\CAIML\Calcium-Imaging-Analysis-Using-Machine-Learning\results'
    
    # File format - 'tif', 'tiff', 'sbx', 'h5'
    ops['input_format'] = 'tif'
    
    # =================================================================
    # REGISTRATION SETTINGS (Motion Correction)
    # =================================================================
    
    # Enable/disable registration (motion correction)
    ops['do_registration'] = True
    
    # Registration method: 1 = rigid, 2 = non-rigid
    ops['nimg_init'] = 300  # Number of images to use for initial registration template
    ops['batch_size'] = 200  # Batch size for registration
    ops['maxregshift'] = 0.1  # Maximum allowed registration shift (fraction of image)
    ops['align_by_chan'] = 1  # Which channel to use for alignment (1 or 2)
    
    # Non-rigid registration settings
    ops['nonrigid'] = True  # Enable non-rigid registration
    ops['block_size'] = [128, 128]  # Size of blocks for non-rigid registration
    ops['snr_thresh'] = 1.2  # Signal-to-noise threshold for registration
    
    # =================================================================
    # CELL DETECTION SETTINGS (ROI Detection)
    # =================================================================
    
    # Diameter of cells in pixels (critical parameter!)
    ops['diameter'] = 15  # Typical range: 10-20 for 2P, adjust based on your data
    
    # Threshold for cell detection
    ops['tau'] = 1.0  # Timescale of transients (in seconds)
    ops['fs'] = 30.0  # Sampling rate (Hz) - IMPORTANT: set to your actual frame rate!
    
    # Cell detection thresholds
    ops['threshold_scaling'] = 1.0  # Scaling factor for detection threshold
    ops['max_overlap'] = 0.75  # Maximum overlap between ROIs (0-1)
    ops['high_pass'] = 100  # High-pass filtering cutoff (frames)
    
    # =================================================================
    # SIGNAL EXTRACTION SETTINGS
    # =================================================================
    
    # Neuropil correction
    ops['neucoeff'] = 0.7  # Neuropil coefficient (0.7 is typical)
    ops['allow_overlap'] = False  # Allow overlapping ROIs
    
    # Deconvolution settings
    ops['baseline'] = 'maximin'  # Baseline estimation method
    ops['win_baseline'] = 60.0  # Window for baseline estimation (seconds)
    ops['sig_baseline'] = 10.0  # Smoothing for baseline
    ops['prctile_baseline'] = 8.0  # Percentile for baseline
    
    # =================================================================
    # PROCESSING SETTINGS
    # =================================================================
    
    # Use GPU acceleration (if available)
    ops['use_GPU'] = True
    ops['GPU_id'] = 0  # GPU device ID (usually 0)
    
    # Memory and performance settings
    ops['num_workers'] = 0  # Number of parallel workers (0 = auto)
    ops['num_workers_roi'] = -1  # Workers for ROI detection (-1 = use all)
    
    # Plane-by-plane processing (for multi-plane data)
    ops['nplanes'] = 1  # Number of planes
    ops['nchannels'] = 1  # Number of channels
    
    # =================================================================
    # QUALITY CONTROL SETTINGS
    # =================================================================
    
    # Minimum number of pixels per ROI
    ops['min_neuropil_pixels'] = 350
    
    # Classifier settings (if using built-in classifier)
    ops['classifier_path'] = None  # Path to custom classifier (None = use default)
    
    logger.info("Suite2p operations configured successfully!")
    return ops

def run_processing_pipeline(ops):
    """
    Run the main Suite2p processing pipeline.
    
    This function executes the complete processing workflow:
    1. Registration (motion correction)
    2. Cell detection (ROI detection)
    3. Signal extraction
    4. Neuropil correction
    5. Spike deconvolution
    
    Args:
        ops (dict): Suite2p operations dictionary
        
    Returns:
        dict: Output statistics and results from Suite2p
    """
    logger.info("Starting Suite2p processing pipeline...")
    
    try:
        # Run the main Suite2p pipeline
        # This is the core function that does all the heavy lifting
        output_ops = run_s2p(ops=ops)
        
        logger.info("Suite2p processing completed successfully!")
        return output_ops
        
    except Exception as e:
        logger.error(f"Error during Suite2p processing: {str(e)}")
        raise

def load_and_visualize_results(save_path):
    """
    Load processed results and create basic visualizations.
    
    This function loads the Suite2p output files and creates some
    basic plots to visualize the results.
    
    Args:
        save_path (str): Path where Suite2p saved the results
    """
    logger.info("Loading and visualizing results...")
    
    try:
        # Load the processed data
        stat = np.load(os.path.join(save_path, 'plane0', 'stat.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(save_path, 'plane0', 'iscell.npy'))
        F = np.load(os.path.join(save_path, 'plane0', 'F.npy'))
        Fneu = np.load(os.path.join(save_path, 'plane0', 'Fneu.npy'))
        spks = np.load(os.path.join(save_path, 'plane0', 'spks.npy'))
        
        # Filter for cells (vs artifacts)
        cell_indices = iscell[:, 0] == 1
        n_cells = np.sum(cell_indices)
        
        logger.info(f"Found {n_cells} cells out of {len(iscell)} total ROIs")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Number of cells detected
        axes[0, 0].bar(['Total ROIs', 'Cells', 'Artifacts'], 
                       [len(iscell), n_cells, len(iscell) - n_cells])
        axes[0, 0].set_title('Cell Detection Summary')
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Cell probability distribution
        axes[0, 1].hist(iscell[:, 1], bins=50, alpha=0.7)
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Default threshold')
        axes[0, 1].set_title('Cell Probability Distribution')
        axes[0, 1].set_xlabel('Cell Probability')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        # Plot 3: Sample fluorescence traces (first 5 cells)
        cell_F = F[cell_indices]
        for i in range(min(5, n_cells)):
            axes[1, 0].plot(cell_F[i, :1000] + i*100, alpha=0.7, label=f'Cell {i+1}')
        axes[1, 0].set_title('Sample Fluorescence Traces (First 1000 frames)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Fluorescence (offset)')
        axes[1, 0].legend()
        
        # Plot 4: Deconvolved spike traces (first 5 cells)
        cell_spks = spks[cell_indices]
        for i in range(min(5, n_cells)):
            axes[1, 1].plot(cell_spks[i, :1000] + i*0.5, alpha=0.7, label=f'Cell {i+1}')
        axes[1, 1].set_title('Sample Spike Traces (First 1000 frames)')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Spike Rate (offset)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(save_path, 'processing_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Summary plot saved to: {plot_path}")
        
        plt.show()
        
        return {
            'n_total_rois': len(iscell),
            'n_cells': n_cells,
            'cell_indices': cell_indices,
            'stat': stat,
            'F': F,
            'Fneu': Fneu,
            'spks': spks
        }
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise

def main():
    """
    Main function to run the complete Suite2p processing pipeline.
    
    This orchestrates the entire workflow from parameter setup to
    final visualization.
    """
    logger.info("=" * 60)
    logger.info("SUITE2P CALCIUM IMAGING PROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Set up processing parameters
    logger.info("Step 1: Setting up processing parameters...")
    ops = setup_suite2p_ops()
    
    # Step 2: Validate data paths
    logger.info("Step 2: Validating data paths...")
    data_path = ops['data_path'][0]
    save_path = ops['save_path0']
    
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please update the 'data_path' in setup_suite2p_ops() function")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Save path: {save_path}")
    
    # Step 3: Run Suite2p processing
    logger.info("Step 3: Running Suite2p processing pipeline...")
    logger.info("This may take a while depending on your data size...")
    
    try:
        output_ops = run_processing_pipeline(ops)
        
        # Step 4: Load and visualize results
        logger.info("Step 4: Loading and visualizing results...")
        results = load_and_visualize_results(save_path)
        
        # Step 5: Print summary
        logger.info("Step 5: Processing complete!")
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total ROIs detected: {results['n_total_rois']}")
        logger.info(f"Cells identified: {results['n_cells']}")
        logger.info(f"Results saved to: {save_path}")
        logger.info("=" * 60)
        
        # Optional: Return results for further analysis
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.info("Check your data paths and parameters, then try again.")
        raise

if __name__ == "__main__":
    # Run the processing pipeline
    results = main()
    
    # Example of how to access results for further analysis:
    if results:
        print(f"\nProcessing completed successfully!")
        print(f"You can now access your results:")
        print(f"- Cell fluorescence traces: results['F']")
        print(f"- Neuropil traces: results['Fneu']")
        print(f"- Spike traces: results['spks']")
        print(f"- Cell statistics: results['stat']")
        print(f"- Cell classification: results['cell_indices']")
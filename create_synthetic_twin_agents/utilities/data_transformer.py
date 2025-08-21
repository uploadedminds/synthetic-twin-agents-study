"""
Data transformation utilities for synthetic twin agents study.

This module provides functions to transform survey data from long format
to wide format for analysis.
"""

import os
from typing import List, Optional

import pandas as pd


def transform_survey_data_to_wide_format(
    input_path: str,
    output_path: str,
    required_columns: Optional[List[str]] = None,
    additional_fields: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Transform survey data from long format to wide format.
    
    This function reads CSV files from the input directory, pivots the data
    from long format (one row per question) to wide format (one row per participant),
    and saves the combined results to the output path.
    
    Parameters
    ----------
    input_path : str
        Directory path containing CSV files to process.
    output_path : str
        File path where the combined wide-format data will be saved.
    required_columns : Optional[List[str]], default=None
        List of required columns that must be present in each CSV file.
        If None, uses default columns: ['agent.prolific_pid', 'scenario.question_name', 'answer.question']
    additional_fields : Optional[List[str]], default=None
        List of additional agent-level fields to extract from the data.
        If None, uses default fields for personality scores.
    verbose : bool, default=True
        Whether to print progress messages and warnings.
    
    Returns
    -------
    pd.DataFrame
        The combined wide-format DataFrame.
    
    Raises
    ------
    FileNotFoundError
        If the input_path directory does not exist.
    ValueError
        If no valid CSV files are found or if required columns are missing.
    """
    
    # Set default required columns if not provided
    if required_columns is None:
        required_columns = [
            'agent.prolific_pid',
            'scenario.question_name', 
            'answer.question'
        ]
    
    # Set default additional fields if not provided
    if additional_fields is None:
        additional_fields = [
            "agent.prolific_pid",
            "agent.extraversion_score",
            "agent.agreeableness_score",
            "agent.conscientiousness_score",
            "agent.neuroticism_score",
            "agent.openness_score"
        ]
    
    # Check if input directory exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    # Initialize an empty list to store transformed DataFrames
    all_transformed_data = []
    processed_files = 0
    skipped_files = 0
    
    # Iterate through all files in the folder
    for file_name in os.listdir(input_path):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_path, file_name)
            
            if verbose:
                print(f"Processing file: {file_name}")
            
            try:
                # Load the CSV file
                data = pd.read_csv(file_path)
                
                # Check if the required columns exist
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    if verbose:
                        print(f"Required columns missing in file {file_name}: {missing_columns}")
                    skipped_files += 1
                    continue
                
                # Pivot the data from long to wide format
                reshaped_data = data.pivot(
                    index='agent.prolific_pid',        # Rows become unique agent IDs
                    columns='scenario.question_name',  # Columns become question names
                    values='answer.question'           # Values populate the table
                )
                
                # Reset column names for better readability
                reshaped_data.columns = reshaped_data.columns.astype(str)
                
                # Add a column to identify the source file
                reshaped_data['source_file'] = file_name
                
                # Extract additional agent-level fields if present
                for field in additional_fields:
                    if field in data.columns:
                        reshaped_data[field] = data.groupby('agent.prolific_pid')[field].first()
                
                # Rename specific columns if they exist in the reshaped data
                column_mapping = {
                    "agent.prolific_pid": "PROLIFIC_PID",
                    "agent.extraversion_score": "extraversion_score",
                    "agent.agreeableness_score": "agreeableness_score",
                    "agent.conscientiousness_score": "conscientiousness_score",
                    "agent.neuroticism_score": "neuroticism_score",
                    "agent.openness_score": "openness_score"
                }
                
                # Only rename columns that exist
                existing_columns = {k: v for k, v in column_mapping.items() if k in reshaped_data.columns}
                if existing_columns:
                    reshaped_data.rename(columns=existing_columns, inplace=True)
                
                # Append the reshaped data to the list
                all_transformed_data.append(reshaped_data)
                processed_files += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error processing file {file_name}: {str(e)}")
                skipped_files += 1
    
    # Check if any files were processed
    if not all_transformed_data:
        raise ValueError(f"No valid CSV files found in {input_path} or all files were skipped")
    
    # Combine all transformed data into a single DataFrame
    final_dataframe = pd.concat(all_transformed_data, axis=0, ignore_index=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the combined DataFrame to a new CSV file
    final_dataframe.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Successfully processed {processed_files} files, skipped {skipped_files} files")
        print(f"Combined data saved to: {output_path}")
        print(f"Final DataFrame shape: {final_dataframe.shape}")
    
    return final_dataframe

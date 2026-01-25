import pandas as pd
import json
import os
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_human_annotations(csv_path):
    """Load human annotations from CSV file."""
    df = pd.read_csv(csv_path)
    # Convert Y/N to boolean (Y = True = Redundant)
    df['redundant_human'] = df['Redundant Y/N'].str.strip().str.upper() == 'Y'
    return df[['Step', 'redundant_human']]

def load_judge_annotations(json_path):
    """Load judge annotations from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle nested structure - steps might be under 'judge_response'
    if 'judge_response' in data and 'steps' in data['judge_response']:
        steps_data = data['judge_response']['steps']
    elif 'steps' in data:
        steps_data = data['steps']
    else:
        raise KeyError(f"Could not find steps in JSON. Keys found: {list(data.keys())}")
    
    steps = []
    for step in steps_data:
        steps.append({
            'Step': step['step_number'],
            'redundant_judge': step['redundant']
        })
    return pd.DataFrame(steps)

def compute_metrics(human_labels, judge_labels):
    """Compute various agreement metrics."""
    # Convert to numpy arrays
    y_human = np.array(human_labels)
    y_judge = np.array(judge_labels)
    
    # Basic agreement
    agreement = np.mean(y_human == y_judge)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_human, y_judge)
    
    # Confusion matrix (human as ground truth, judge as predictions)
    # TN: both say not redundant, FP: human says no, judge says yes
    # FN: human says yes, judge says no, TP: both say redundant
    tn, fp, fn, tp = confusion_matrix(y_human, y_judge).ravel()
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'agreement': agreement,
        'kappa': kappa,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_steps': len(y_human),
        'human_redundant_count': np.sum(y_human),
        'judge_redundant_count': np.sum(y_judge)
    }

def plot_confusion_matrix(cm, title, output_path, normalize=False):
    """Plot and save a confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                    xticklabels=['Not Redundant', 'Redundant'],
                    yticklabels=['Not Redundant', 'Redundant'],
                    cbar_kws={'label': 'Proportion'},
                    annot_kws={'size': 18})
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Redundant', 'Redundant'],
                    yticklabels=['Not Redundant', 'Redundant'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 18})
    
    plt.title(title, fontsize=20, pad=20)
    plt.ylabel('Human Annotation (Ground Truth)', fontsize=18)
    plt.xlabel('Judge Prediction', fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(results_df, output_path):
    """Create a bar plot comparing metrics across instances."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['agreement', 'kappa', 'precision', 'recall']
    titles = ['Agreement Rate', "Cohen's Kappa", 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        bars = ax.bar(range(len(results_df)), results_df[metric], 
                      color='steelblue', alpha=0.8)
        ax.axhline(y=results_df[metric].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {results_df[metric].mean():.3f}')
        ax.set_xlabel('Instance', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels([f"I{i+1}" for i in range(len(results_df))], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregate_confusion_matrix(results_df, output_path):
    """Plot the aggregate confusion matrix across all instances."""
    total_tn = results_df['tn'].sum()
    total_fp = results_df['fp'].sum()
    total_fn = results_df['fn'].sum()
    total_tp = results_df['tp'].sum()
    
    cm_aggregate = np.array([[total_tn, total_fp],
                             [total_fn, total_tp]])
    
    plot_confusion_matrix(cm_aggregate, 
                         'Aggregate Confusion Matrix (All Instances)',
                         output_path)
    
    # Also create normalized version
    output_path_norm = output_path.parent / f"{output_path.stem}_normalized{output_path.suffix}"
    plot_confusion_matrix(cm_aggregate, 
                         'Aggregate Confusion Matrix (Normalized)',
                         output_path_norm,
                         normalize=True)

def analyze_disagreements(merged_df):
    """Identify and categorize disagreements."""
    disagreements = merged_df[merged_df['redundant_human'] != merged_df['redundant_judge']].copy()
    
    # FP: Judge says redundant, Human says not redundant
    fp_cases = disagreements[disagreements['redundant_judge'] & ~disagreements['redundant_human']]
    
    # FN: Judge says not redundant, Human says redundant
    fn_cases = disagreements[~disagreements['redundant_judge'] & disagreements['redundant_human']]
    
    return {
        'disagreements': disagreements,
        'false_positives': fp_cases,
        'false_negatives': fn_cases
    }

def main():
    # Paths
    human_csv_dir = Path('../human_annotation_csvs/base_dev_Qwen3-Coder-30b')
    judge_json_dir = Path('../judge_analysis_majority_vote_k_5/base_dev_Qwen3-Coder-30b/policy_v2')
    output_dir = Path('../iaa/policy_v2_majority_vote_k_5/base_dev_Qwen3-Coder-30b')
    viz_dir = output_dir / 'visualizations'
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Open output file for writing
    output_file = output_dir / 'iaa_analysis_report.txt'
    
    # Find all CSV files
    csv_files = list(human_csv_dir.glob('*.csv'))
    
    if not csv_files:
        msg = f"No CSV files found in {human_csv_dir}"
        print(msg)
        with open(output_file, 'w') as f:
            f.write(msg + '\n')
        return
    
    msg = f"Found {len(csv_files)} CSV files to analyze\n"
    print(msg)
    
    # Write to file as we go
    with open(output_file, 'w') as f:
        f.write(msg)
    
    # Store results for all instances
    all_results = []
    all_disagreements = []
    
    for csv_file in csv_files:
        instance_id = csv_file.stem
        output_lines = []
        
        header = f"\n{'='*60}\nAnalyzing: {instance_id}\n{'='*60}"
        print(header)
        output_lines.append(header)
        
        # Find corresponding JSON file
        json_files = list(judge_json_dir.glob(f"{instance_id}*judge*result.json"))
        
        if not json_files:
            msg = f"Warning: No matching judge JSON found for {instance_id}"
            print(msg)
            output_lines.append(msg)
            with open(output_file, 'a') as f:
                f.write('\n'.join(output_lines) + '\n')
            continue
        
        json_file = json_files[0]  # Take first match if multiple
        
        try:
            # Load annotations
            human_df = load_human_annotations(csv_file)
            judge_df = load_judge_annotations(json_file)
            
            # Merge on step number
            merged = human_df.merge(judge_df, on='Step', how='inner')
            
            if len(merged) == 0:
                msg = f"Warning: No matching steps found for {instance_id}"
                print(msg)
                output_lines.append(msg)
                with open(output_file, 'a') as f:
                    f.write('\n'.join(output_lines) + '\n')
                continue
            
            # Compute metrics
            metrics = compute_metrics(
                merged['redundant_human'].values,
                merged['redundant_judge'].values
            )
            
            # Create confusion matrix visualization for this instance
            cm = np.array([[metrics['tn'], metrics['fp']],
                          [metrics['fn'], metrics['tp']]])
            cm_path = viz_dir / f"{instance_id}_confusion_matrix.png"
            plot_confusion_matrix(cm, f"Confusion Matrix: {instance_id}", cm_path)
            
            # Build output
            output_lines.append(f"\nTotal Steps: {metrics['total_steps']}")
            output_lines.append(f"Human marked as redundant: {metrics['human_redundant_count']}")
            output_lines.append(f"Judge marked as redundant: {metrics['judge_redundant_count']}")
            output_lines.append(f"\nAgreement Rate: {metrics['agreement']:.2%}")
            output_lines.append(f"Cohen's Kappa: {metrics['kappa']:.3f}")
            output_lines.append(f"\nConfusion Matrix (Human as ground truth):")
            output_lines.append(f"  True Negatives (TN):  {metrics['tn']} (both say not redundant)")
            output_lines.append(f"  False Positives (FP): {metrics['fp']} (judge says redundant, human says no)")
            output_lines.append(f"  False Negatives (FN): {metrics['fn']} (judge says not redundant, human says yes)")
            output_lines.append(f"  True Positives (TP):  {metrics['tp']} (both say redundant)")
            output_lines.append(f"\nPerformance Metrics:")
            output_lines.append(f"  Precision: {metrics['precision']:.3f}")
            output_lines.append(f"  Recall:    {metrics['recall']:.3f}")
            output_lines.append(f"  F1 Score:  {metrics['f1']:.3f}")
            
            # Print to console
            for line in output_lines:
                print(line)
            
            # Analyze disagreements
            disagreement_analysis = analyze_disagreements(merged)
            
            if len(disagreement_analysis['false_positives']) > 0:
                fp_msg = f"\nFalse Positives (Steps where judge incorrectly marked as redundant):\n{disagreement_analysis['false_positives']['Step'].tolist()}"
                print(fp_msg)
                output_lines.append(fp_msg)
            
            if len(disagreement_analysis['false_negatives']) > 0:
                fn_msg = f"\nFalse Negatives (Steps where judge missed redundancy):\n{disagreement_analysis['false_negatives']['Step'].tolist()}"
                print(fn_msg)
                output_lines.append(fn_msg)
            
            # Write to file
            with open(output_file, 'a') as f:
                f.write('\n'.join(output_lines) + '\n')
            
            # Store results
            result_row = {
                'instance_id': instance_id,
                **metrics
            }
            all_results.append(result_row)
            
            # Store disagreements with instance_id
            for _, row in disagreement_analysis['disagreements'].iterrows():
                all_disagreements.append({
                    'instance_id': instance_id,
                    'step': row['Step'],
                    'human_redundant': row['redundant_human'],
                    'judge_redundant': row['redundant_judge'],
                    'type': 'FP' if row['redundant_judge'] else 'FN'
                })
        
        except Exception as e:
            error_msg = f"Error processing {instance_id}: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            output_lines.append(error_msg)
            with open(output_file, 'a') as f:
                f.write('\n'.join(output_lines) + '\n')
    
    # Create summary dataframes
    if all_results:
        results_df = pd.DataFrame(all_results)
        disagreements_df = pd.DataFrame(all_disagreements)
        
        # Create aggregate visualizations
        print("\nGenerating aggregate visualizations...")
        plot_aggregate_confusion_matrix(results_df, viz_dir / 'aggregate_confusion_matrix.png')
        plot_metrics_comparison(results_df, viz_dir / 'metrics_comparison.png')
        
        summary_lines = []
        summary_lines.append(f"\n{'='*60}")
        summary_lines.append("OVERALL SUMMARY")
        summary_lines.append('='*60)
        
        summary_lines.append(f"\nTotal instances analyzed: {len(results_df)}")
        summary_lines.append(f"\nAggregate Statistics:")
        summary_lines.append(f"  Mean Agreement Rate: {results_df['agreement'].mean():.2%} (±{results_df['agreement'].std():.2%})")
        summary_lines.append(f"  Mean Cohen's Kappa: {results_df['kappa'].mean():.3f} (±{results_df['kappa'].std():.3f})")
        summary_lines.append(f"  Mean Precision: {results_df['precision'].mean():.3f} (±{results_df['precision'].std():.3f})")
        summary_lines.append(f"  Mean Recall: {results_df['recall'].mean():.3f} (±{results_df['recall'].std():.3f})")
        summary_lines.append(f"  Mean F1 Score: {results_df['f1'].mean():.3f} (±{results_df['f1'].std():.3f})")
        
        summary_lines.append(f"\nTotal Confusion Matrix:")
        summary_lines.append(f"  Total TN: {results_df['tn'].sum()}")
        summary_lines.append(f"  Total FP: {results_df['fp'].sum()}")
        summary_lines.append(f"  Total FN: {results_df['fn'].sum()}")
        summary_lines.append(f"  Total TP: {results_df['tp'].sum()}")
        
        # Print to console
        for line in summary_lines:
            print(line)
        
        # Write to file
        with open(output_file, 'a') as f:
            f.write('\n'.join(summary_lines) + '\n')
        
        # Save results to output directory
        results_csv = output_dir / 'iaa_results_summary.csv'
        disagreements_csv = output_dir / 'iaa_disagreements.csv'
        
        results_df.to_csv(results_csv, index=False)
        disagreements_df.to_csv(disagreements_csv, index=False)
        
        save_msg = f"\nResults saved to:"
        save_msg += f"\n  - {output_file}"
        save_msg += f"\n  - {results_csv}"
        save_msg += f"\n  - {disagreements_csv}"
        save_msg += f"\n  - Visualizations in {viz_dir}/"
        
        print(save_msg)
        with open(output_file, 'a') as f:
            f.write(save_msg + '\n')
        
        return results_df, disagreements_df
    else:
        msg = "No results to summarize"
        print(msg)
        with open(output_file, 'a') as f:
            f.write(msg + '\n')
        return None, None

if __name__ == "__main__":
    results_df, disagreements_df = main()
import matplotlib.pyplot as plt
import numpy as np
from endpoint_metrics_helper import analyze_routing_detailed

# Create visualizations for GPU metrics
def visualize_gpu_metrics(util_data):
    """Create comprehensive GPU-focused visualizations."""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Prepare data for both inference components
    ic_names = list(util_data.keys())
    colors = ['steelblue', 'coral']
    
    # ========== Row 1: GPU Utilization by GPU ID ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    for idx, ic_name in enumerate(ic_names):
        data = util_data[ic_name]
        gpu_metrics = data['gpu_metrics']
        
        # Aggregate by GPU ID
        gpu_summary = {}
        for metric in gpu_metrics:
            gpu_id = metric['gpu_id']
            if gpu_id not in gpu_summary:
                gpu_summary[gpu_id] = {'compute': [], 'memory': []}
            
            if 'GPUUtilizationNormalized' in metric['metrics']:
                gpu_summary[gpu_id]['compute'].append(metric['metrics']['GPUUtilizationNormalized']['avg'])
            if 'GPUMemoryUtilizationNormalized' in metric['metrics']:
                gpu_summary[gpu_id]['memory'].append(metric['metrics']['GPUMemoryUtilizationNormalized']['avg'])
        # gpu_summary["gpu_2"] = {}
        # gpu_summary["gpu_3"] = {}
        
        # Plot GPU Compute
        gpu_ids = sorted(gpu_summary.keys())
        # gpu_ids = ["gpu_0", "gpu_1", "gpu_2", "gpu_3"]
        compute_avgs = [np.mean(gpu_summary[gid]['compute']) if gpu_summary[gid]['compute'] else 0 
                       for gid in gpu_ids]
        
        x = np.arange(len(gpu_ids))
        width = 0.35
        offset = width * idx - width/2
        
        bars = ax1.bar(x + offset, compute_avgs, width, label=ic_name.split('-')[-1], 
                      color=colors[idx], alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, compute_avgs):
            if val > 0.5:
                ax1.text(bar.get_x() + bar.get_width()/2, val + 2, 
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot GPU Memory
        memory_avgs = [np.mean(gpu_summary[gid]['memory']) if gpu_summary[gid]['memory'] else 0 
                      for gid in gpu_ids]
        
        bars = ax2.bar(x + offset, memory_avgs, width, label=ic_name, 
                      color=colors[idx], alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, memory_avgs):
            if val > 0.5:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 2, 
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Configure GPU Compute plot
    ax1.set_xlabel('GPU ID', fontsize=10)
    ax1.set_ylabel('GPU Compute %', fontsize=10)
    ax1.set_title('GPU Compute Utilization by GPU ID', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gpu_ids)
    # ax1.set_xticklabels(sorted(["gpu_0", "gpu_1", "gpu_2", "gpu_3"]))
    ax1.set_ylim(0, 100)
    ax1.axhline(30, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Low (30%)')
    ax1.axhline(80, color='red', linestyle='--', alpha=0.5, linewidth=1, label='High (80%)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Configure GPU Memory plot
    ax2.set_xlabel('GPU ID', fontsize=10)
    ax2.set_ylabel('GPU Memory %', fontsize=10)
    ax2.set_title('GPU Memory Utilization by GPU ID', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gpu_ids)
    # ax2.set_xticklabels(sorted(["gpu_0", "gpu_1", "gpu_2", "gpu_3"]))
    ax2.set_ylim(0, 100)
    ax2.axhline(85, color='red', linestyle='--', alpha=0.5, linewidth=1, label='High (85%)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========== Row 2: Distribution across containers ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    for idx, ic_name in enumerate(ic_names):
        data = util_data[ic_name]
        gpu_metrics = data['gpu_metrics']
        
        # Get all compute and memory values
        compute_vals = []
        memory_vals = []
        
        for metric in gpu_metrics:
            if 'GPUUtilizationNormalized' in metric['metrics']:
                compute_vals.append(metric['metrics']['GPUUtilizationNormalized']['avg'])
            if 'GPUMemoryUtilizationNormalized' in metric['metrics']:
                memory_vals.append(metric['metrics']['GPUMemoryUtilizationNormalized']['avg'])
        
        # Box plot for compute
        bp1 = ax3.boxplot([compute_vals], positions=[idx], widths=0.6, 
                          patch_artist=True, tick_labels=[ic_name])
        for patch in bp1['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.7)
        
        # Box plot for memory
        bp2 = ax4.boxplot([memory_vals], positions=[idx], widths=0.6, 
                          patch_artist=True, tick_labels=[ic_name])
        for patch in bp2['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.7)
    
    # Configure distribution plots
    ax3.set_ylabel('GPU Compute %', fontsize=10)
    ax3.set_title('GPU Compute Distribution Across Containers', fontsize=12, fontweight='bold')
    # ax3.set_ylim(5, 15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4.set_ylabel('GPU Memory %', fontsize=10)
    ax4.set_title('GPU Memory Distribution Across Containers', fontsize=12, fontweight='bold')
    # ax4.set_ylim(80, 85)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========== Row 3 & 4: Per-container heatmaps for BOTH ICs ==========
    for ic_idx, ic_name in enumerate(ic_names):
        ax = fig.add_subplot(gs[2 + ic_idx, :])
        
        data = util_data[ic_name]
        gpu_metrics = data['gpu_metrics']
        
        # Create list of containers with their GPU and memory utilization
        container_data = []
        for metric in gpu_metrics:
            if 'GPUMemoryUtilizationNormalized' in metric['metrics']:
                container_data.append({
                    'instance': metric['instance_id'][-8:],  # Last 8 chars
                    'container': metric['container_id'][:8],  # First 8 chars
                    'gpu_id': metric['gpu_id'],
                    'memory': metric['metrics']['GPUMemoryUtilizationNormalized']['avg']
                })
        
        # Sort by instance, then GPU
        container_data.sort(key=lambda x: (x['instance'], x['gpu_id']))
        
        # Get unique GPU IDs
        all_gpu_ids = sorted(set(c['gpu_id'] for c in container_data))
        
        # Create heatmap matrix - one row per container
        heatmap_data = []
        row_labels = []
        
        for container in container_data:
            row = [0] * len(all_gpu_ids)
            gpu_idx = all_gpu_ids.index(container['gpu_id'])
            row[gpu_idx] = container['memory']
            heatmap_data.append(row)
            row_labels.append(f"{container['instance']}/{container['container']}")
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_gpu_ids)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(all_gpu_ids, fontsize=9)
        ax.set_yticklabels(row_labels, fontsize=7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('GPU Memory %', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(all_gpu_ids)):
                val = heatmap_data[i][j]
                if val > 0:
                    text = ax.text(j, i, f'{val:.1f}', ha="center", va="center", 
                                  color="white" if val > 50 else "black", fontsize=7)
        
        ax.set_title(f'GPU Memory by Container - {ic_name} ({len(container_data)} containers)', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('GPU ID', fontsize=10)
        ax.set_ylabel('Instance/Container', fontsize=10)
    plt.show()


# Extract GPU metrics from the new structure
def extract_gpu_summary(util_data_entry):
    """Extract per-GPU summary from detailed metrics."""
    gpu_metrics = util_data_entry.get('gpu_metrics', [])
    
    # Aggregate by GPU ID
    gpu_summary = {}
    for metric in gpu_metrics:
        gpu_id = metric['gpu_id']
        if gpu_id not in gpu_summary:
            gpu_summary[gpu_id] = {'compute': [], 'memory': []}
        
        if 'GPUUtilizationNormalized' in metric['metrics']:
            gpu_summary[gpu_id]['compute'].append(metric['metrics']['GPUUtilizationNormalized']['avg'])
        if 'GPUMemoryUtilizationNormalized' in metric['metrics']:
            gpu_summary[gpu_id]['memory'].append(metric['metrics']['GPUMemoryUtilizationNormalized']['avg'])
    
    # Average across all containers for each GPU
    for gpu_id in gpu_summary:
        if gpu_summary[gpu_id]['compute']:
            gpu_summary[gpu_id]['compute_avg'] = np.mean(gpu_summary[gpu_id]['compute'])
        else:
            gpu_summary[gpu_id]['compute_avg'] = 0
            
        if gpu_summary[gpu_id]['memory']:
            gpu_summary[gpu_id]['memory_avg'] = np.mean(gpu_summary[gpu_id]['memory'])
        else:
            gpu_summary[gpu_id]['memory_avg'] = 0
    
    return gpu_summary


def visualize_gpu_utilization(util_data, ic_names, gpu_counts):

    IC_NAME_A, IC_NAME_B = ic_names[0], ic_names[1]
    MODEL_A_GPU_COUNT, MODEL_B_GPU_COUNT = gpu_counts[0], gpu_counts[1]
    
    util_a_summary = extract_gpu_summary(util_data.get(IC_NAME_A, {}))
    util_b_summary = extract_gpu_summary(util_data.get(IC_NAME_B, {}))
    
    comp_metrics_a = util_data.get(IC_NAME_A, {}).get('component_metrics', {})
    comp_metrics_b = util_data.get(IC_NAME_B, {}).get('component_metrics', {})
    
    GPU_COUNT = max(MODEL_A_GPU_COUNT, MODEL_B_GPU_COUNT)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, GPU_COUNT, hspace=0.4, wspace=0.3)
    
    # Configuration labels
    config1_label = f'Config A'
    config2_label = f'Config B'
    
    # Get all GPU IDs
    all_gpu_ids = sorted(set(list(util_a_summary.keys()) + list(util_b_summary.keys())))
    
    # Row 1: GPU Compute Utilization per GPU
    for idx, gpu_id in enumerate(all_gpu_ids[:GPU_COUNT]):
        ax = fig.add_subplot(gs[0, idx])
        
        gpu_compute = [
            util_a_summary.get(gpu_id, {}).get('compute_avg', 0),
            util_b_summary.get(gpu_id, {}).get('compute_avg', 0)
        ]
        
        x = np.arange(2)
        bars = ax.bar(x, gpu_compute, color=['steelblue', 'coral'], alpha=0.7, width=0.5)
        ax.axhline(30, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(80, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('GPU %', fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f'{gpu_id} Compute', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['A', 'B'], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, gpu_compute)):
            if val > 0.5:
                ax.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=8)
    
    # Row 2: GPU Memory Utilization per GPU
    for idx, gpu_id in enumerate(all_gpu_ids[:GPU_COUNT]):
        ax = fig.add_subplot(gs[1, idx])
        
        gpu_memory = [
            util_a_summary.get(gpu_id, {}).get('memory_avg', 0),
            util_b_summary.get(gpu_id, {}).get('memory_avg', 0)
        ]
        
        x = np.arange(2)
        bars = ax.bar(x, gpu_memory, color=['steelblue', 'coral'], alpha=0.7, width=0.5)
        ax.axhline(85, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('GPU Mem %', fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f'{gpu_id} Memory', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['A', 'B'], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, gpu_memory)):
            if val > 0.5:
                ax.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=8)
    
    # Row 3: Component-level metrics (CPU, Memory, Latency)
    ax_cpu = fig.add_subplot(gs[2, 0])
    ax_mem = fig.add_subplot(gs[2, 1])
    ax_latency = fig.add_subplot(gs[2, 2])
    ax_errors = fig.add_subplot(gs[2, 3])
    
    # CPU Utilization
    cpu_vals = [
        comp_metrics_a.get('CPUUtilizationNormalized', {}).get('avg', 0),
        comp_metrics_b.get('CPUUtilizationNormalized', {}).get('avg', 0)
    ]
    x = np.arange(2)
    bars = ax_cpu.bar(x, cpu_vals, color=['steelblue', 'coral'], alpha=0.7, width=0.5)
    ax_cpu.set_ylabel('CPU %', fontsize=9)
    ax_cpu.set_title('CPU Utilization', fontsize=10, fontweight='bold')
    ax_cpu.set_xticks(x)
    ax_cpu.set_xticklabels(['A', 'B'], fontsize=9)
    ax_cpu.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, cpu_vals)):
        if val > 0.5:
            ax_cpu.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Memory Utilization
    mem_vals = [
        comp_metrics_a.get('MemoryUtilizationNormalized', {}).get('avg', 0),
        comp_metrics_b.get('MemoryUtilizationNormalized', {}).get('avg', 0)
    ]
    bars = ax_mem.bar(x, mem_vals, color=['steelblue', 'coral'], alpha=0.7, width=0.5)
    ax_mem.set_ylabel('Memory %', fontsize=9)
    ax_mem.set_title('Memory Utilization', fontsize=10, fontweight='bold')
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(['A', 'B'], fontsize=9)
    ax_mem.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, mem_vals)):
        if val > 0.5:
            ax_mem.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Model Latency
    latency_vals = [
        comp_metrics_a.get('ModelLatency', {}).get('avg', 0),
        comp_metrics_b.get('ModelLatency', {}).get('avg', 0)
    ]
    bars = ax_latency.bar(x, latency_vals, color=['steelblue', 'coral'], alpha=0.7, width=0.5)
    ax_latency.set_ylabel('Latency (ms)', fontsize=9)
    ax_latency.set_title('Model Latency', fontsize=10, fontweight='bold')
    ax_latency.set_xticks(x)
    ax_latency.set_xticklabels(['A', 'B'], fontsize=9)
    ax_latency.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, latency_vals)):
        if val > 0.5:
            ax_latency.text(i, val + max(latency_vals)*0.02, f'{val:.0f}ms', 
                          ha='center', va='bottom', fontsize=8)
    
    # Total Errors
    error_vals = [
        sum([comp_metrics_a.get(m, {}).get('total', 0) 
             for m in ['Invocation4XXErrors', 'Invocation5XXErrors', 'InvocationModelErrors']]),
        sum([comp_metrics_b.get(m, {}).get('total', 0) 
             for m in ['Invocation4XXErrors', 'Invocation5XXErrors', 'InvocationModelErrors']])
    ]
    bars = ax_errors.bar(x, error_vals, color=['steelblue', 'coral'], alpha=0.7, width=0.5)
    ax_errors.set_ylabel('Error Count', fontsize=9)
    ax_errors.set_title('Total Errors', fontsize=10, fontweight='bold')
    ax_errors.set_xticks(x)
    ax_errors.set_xticklabels(['A', 'B'], fontsize=9)
    ax_errors.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, error_vals)):
        if val > 0.5:
            ax_errors.text(i, val + 0.5, f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.show()


def visualize_routing_balance(ic_names, routing_data):

    IC_NAME_A, IC_NAME_B = ic_names[0], ic_names[1]
    routing_a, routing_b = routing_data[0], routing_data[1]
    # routing_a = analyze_routing_detailed(IC_NAME_A, hours=1)
    # routing_b = analyze_routing_detailed(IC_NAME_B, hours=1)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ========== Row 1: Total Invocations Over Time ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    if 'total_invocations' in routing_a and 'total_invocations' in routing_b:
        # Get time series data (you'll need to store this in analyze_routing_detailed)
        # For now, let's show the summary
        total_a = routing_a['total_invocations']['sum']
        total_b = routing_b['total_invocations']['sum']
        
        x = np.arange(2)
        bars = ax1.bar(x, [total_a, total_b], color=['steelblue', 'coral'], 
                      alpha=0.7, width=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Config A\n({routing_a.get("model_copies", "?")} copies)', 
                             f'Config B\n({routing_b.get("model_copies", "?")} copies)'])
        ax1.set_ylabel('Total Invocations', fontsize=11)
        ax1.set_title('Total Invocations Comparison', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, [total_a, total_b]):
            ax1.text(bar.get_x() + bar.get_width()/2, val + max(total_a, total_b)*0.02, 
                    f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ========== Row 2: Per-Copy Distribution ==========
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Config A - Per-Copy Distribution
    if 'container_invocations' in routing_a:
        containers_a = routing_a['container_invocations']
        totals_a = [c['total'] for c in containers_a]
        labels_a = [f"{c['container_id']}" for c in containers_a]
        
        bars = ax2.barh(range(len(totals_a)), totals_a, color='steelblue', alpha=0.7)
        ax2.set_yticks(range(len(totals_a)))
        ax2.set_yticklabels(labels_a, fontsize=8)
        ax2.set_xlabel('Invocations', fontsize=10)
        ax2.set_title(f'Config A: Per-Copy Distribution ({len(containers_a)} copies)', 
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, totals_a)):
            ax2.text(val + max(totals_a)*0.02, i, f'{val:,.0f}', 
                    va='center', fontsize=8)
        
        # Add average line
        avg_a = np.mean(totals_a)
        ax2.axvline(avg_a, color='darkblue', linestyle='--', linewidth=2, 
                   label=f'Avg: {avg_a:,.0f}')
        ax2.legend(fontsize=9)
    
    # Config B - Per-Copy Distribution
    if 'container_invocations' in routing_b:
        containers_b = routing_b['container_invocations']
        totals_b = [c['total'] for c in containers_b]
        labels_b = [f"{c['container_id']}" for c in containers_b]
        
        bars = ax3.barh(range(len(totals_b)), totals_b, color='coral', alpha=0.7)
        ax3.set_yticks(range(len(totals_b)))
        ax3.set_yticklabels(labels_b, fontsize=8)
        ax3.set_xlabel('Invocations', fontsize=10)
        ax3.set_title(f'Config B: Per-Copy Distribution ({len(containers_b)} copies)', 
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, totals_b)):
            ax3.text(val + max(totals_b)*0.02, i, f'{val:,.0f}', 
                    va='center', fontsize=8)
        
        # Add average line
        avg_b = np.mean(totals_b)
        ax3.axvline(avg_b, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Avg: {avg_b:,.0f}')
        ax3.legend(fontsize=9)
    
    # ========== Row 3: Balance Analysis ==========
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Config A - Balance Visualization
    if 'container_invocations' in routing_a:
        totals_a = [c['total'] for c in routing_a['container_invocations']]
        total_sum_a = sum(totals_a)
        percentages_a = [(t / total_sum_a * 100) if total_sum_a > 0 else 0 for t in totals_a]
        expected_pct_a = 100 / len(totals_a) if totals_a else 0
        
        x = np.arange(len(percentages_a))
        bars = ax4.bar(x, percentages_a, color='steelblue', alpha=0.7)
        ax4.axhline(expected_pct_a, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected: {expected_pct_a:.1f}%')
        ax4.set_xlabel('Copy Index', fontsize=10)
        ax4.set_ylabel('% of Total Traffic', fontsize=10)
        ax4.set_ylim(0, 100)
        ax4.set_title(f'Config A: Traffic Balance (Imbalance: {routing_a.get("imbalance", 0):.2f}x)', 
                     fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{i+1}' for i in range(len(percentages_a))])
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, percentages_a):
            if val > 0.5:
                ax4.text(bar.get_x() + bar.get_width()/2, val + 1, 
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Config B - Balance Visualization
    if 'container_invocations' in routing_b:
        totals_b = [c['total'] for c in routing_b['container_invocations']]
        total_sum_b = sum(totals_b)
        percentages_b = [(t / total_sum_b * 100) if total_sum_b > 0 else 0 for t in totals_b]
        expected_pct_b = 100 / len(totals_b) if totals_b else 0
        
        x = np.arange(len(percentages_b))
        bars = ax5.bar(x, percentages_b, color='coral', alpha=0.7)
        ax5.axhline(expected_pct_b, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected: {expected_pct_b:.1f}%')
        ax5.set_xlabel('Copy Index', fontsize=10)
        ax5.set_ylabel('% of Total Traffic', fontsize=10)
        ax5.set_ylim(0, 105)
        ax5.set_title(f'Config B: Traffic Balance (Imbalance: {routing_b.get("imbalance", 0):.2f}x)', 
                     fontsize=11, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'{i+1}' for i in range(len(percentages_b))])
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, percentages_b):
            if val > 0.5:
                ax5.text(bar.get_x() + bar.get_width()/2, val + 1, 
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.show()
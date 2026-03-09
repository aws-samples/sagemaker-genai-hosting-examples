import asyncio
import boto3
import json
import random
import time
from datetime import datetime, timedelta, timezone
import ipywidgets as widgets
from IPython.display import display
import concurrent.futures
from tqdm.notebook import tqdm
import threading

cloudwatch = boto3.client('cloudwatch')
sm_client = boto3.client('sagemaker')
smr_client = boto3.client('sagemaker-runtime')

prompts = [
    "Explain machine learning in simple terms.",
    "What is the difference between AI and ML?",
    "How do neural networks work?",
    "What are transformers in deep learning?"
]


def invoke_endpoint(prompt, endpoint, inference_component):
    """Invoke the inference component."""
    payload = {'inputs': prompt, 'parameters': {'max_new_tokens': 100, 'temperature': 0.7}}
    response = smr_client.invoke_endpoint(
        EndpointName=endpoint,
        InferenceComponentName=inference_component,  # Route to specific component
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    return json.loads(response['Body'].read().decode())


def generate_load(endpoint_name, ic_names, num_requests=50, max_workers=5, delay_between_requests=0.1):
    """Generate concurrent load with dual progress bars.
    
    Args:
        num_requests: Total number of requests to send PER ENDPOINT (total = num_requests * 2)
        max_workers: Number of concurrent workers
        delay_between_requests: Seconds to wait between submitting each request (default 0.1)
                                Higher values create more even distribution across copies
    """
    results = {'success': 0, 'error': 0}
    total_requests = num_requests * 2  # Sending to both endpoints
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Progress bar 1: Submitting requests
        pbar_submit = tqdm(total=total_requests, desc="Submitting requests", unit="req", position=0, leave=True)
        
        # Function to monitor completions in background
        def monitor_completions():
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                    results['success'] += 1
                except Exception as e:
                    results['error'] += 1
        
        # Submit requests with delays to avoid bursts
        for i in range(num_requests):
            # Submit to both endpoints
            for ic in ic_names:
                futures.append(executor.submit(invoke_endpoint, random.choice(prompts), endpoint_name, ic))
                pbar_submit.update(1)
            
            # Start monitoring completions after first batch is submitted
            if i == 0:
                monitor_thread = threading.Thread(target=monitor_completions, daemon=True)
                monitor_thread.start()
            
            # Optional delay between request pairs
            if delay_between_requests > 0 and i < num_requests - 1:
                time.sleep(delay_between_requests)

        
        pbar_submit.close()
        
        # Wait for monitoring thread to finish
        monitor_thread.join()
    
    return results


def create_metrics_widget(endpoint_name, region='us-east-1'):
    """Create interactive widget to get endpoint GPU and instance metrics."""
    
    time_option = widgets.Dropdown(
        options=['Last 5 min', 'Last 10 min', 'Last 30 min', 'Last 1 hour', 'Custom Range'],
        value='Last 10 min',
        description='Time Range:'
    )
    
    start_picker = widgets.DatetimePicker(
        description='Start Time:',
        disabled=True,
        value=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    
    end_picker = widgets.DatetimePicker(
        description='End Time:',
        disabled=True,
        value=datetime.now(timezone.utc)
    )
    
    button = widgets.Button(description='Get Metrics', button_style='primary')
    output = widgets.Output()
    
    def on_time_option_change(change):
        if change['new'] == 'Custom Range':
            start_picker.disabled = False
            end_picker.disabled = False
        else:
            start_picker.disabled = True
            end_picker.disabled = True
    
    time_option.observe(on_time_option_change, names='value')
    
    def get_metrics(b):
        output.clear_output()
        with output:
            if time_option.value == 'Last 5 min':
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(minutes=5)
            elif time_option.value == 'Last 10 min':
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(minutes=10)
            elif time_option.value == 'Last 30 min':
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(minutes=30)
            elif time_option.value == 'Last 1 hour':
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=1)
            else:
                start_time = start_picker.value
                end_time = end_picker.value
            
            cw = boto3.client('cloudwatch', region_name=region)
            
            response = cw.get_metric_data(
                MetricDataQueries=[
                    {
                        'Id': 'm1',
                        'Expression': f'SEARCH(\'{{/aws/sagemaker/InferenceComponents,EndpointName,VariantName,InstanceId,ContainerId,InferenceComponentName,GpuId}} MetricName="GPUUtilizationNormalized" EndpointName="{endpoint_name}"\', \'SampleCount\', 10)'
                    },
                    {
                        'Id': 'e1',
                        'Expression': 'SUM(m1)'
                    },
                    {
                        'Id': 'm2',
                        'MetricStat': {
                            'Metric': {
                                'Namespace': '/aws/sagemaker/Endpoints',
                                'MetricName': 'CPUUtilizationNormalized',
                                'Dimensions': [
                                    {'Name': 'EndpointName', 'Value': endpoint_name},
                                    {'Name': 'VariantName', 'Value': 'AllTraffic'}
                                ]
                            },
                            'Period': 10,
                            'Stat': 'SampleCount'
                        }
                    }
                ],
                StartTime=start_time,
                EndTime=end_time
            )
            
            used_gpus = None
            num_instances = None
            
            for result in response['MetricDataResults']:
                if result['Id'] == 'e1' and result['Values']:
                    used_gpus = result['Values'][0]
                elif result['Id'] == 'm2' and result['Values']:
                    num_instances = result['Values'][0]
            
            total_gpus = num_instances * 4 if num_instances else 0
            free_gpus = total_gpus - used_gpus if (total_gpus and used_gpus) else 0
            
            print(f"Endpoint: {endpoint_name}")
            print(f"Time Range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n{'='*50}")
            print(f"Number of Instances: {num_instances}")
            print(f"Total GPUs Available: {total_gpus}")
            print(f"Used GPUs: {used_gpus}")
            print(f"Free GPUs: {free_gpus}")
    
    button.on_click(get_metrics)
    
    display(time_option, start_picker, end_picker, button, output)


def get_model_cost(inference_component_name, cost_per_hour, region='us-east-1', time_range='Last 10 min', start_time=None, end_time=None):
    """
    Get cumulative cost for a specific inference component.
    
    Args:
        inference_component_name: Name of the inference component
        cost_per_hour: Cost per hour for the model (e.g., 5.752)
        region: AWS region
        time_range: One of 'Last 5 min', 'Last 10 min', 'Last 30 min', 'Last 1 hour', or 'Custom'
        start_time: Start time for custom range (timezone aware datetime)
        end_time: End time for custom range (timezone aware datetime)
    
    Returns:
        dict with cumulative_cost and latest_cost_per_10s
    """
    
    if time_range == 'Last 5 min':
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=5)
    elif time_range == 'Last 10 min':
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=10)
    elif time_range == 'Last 30 min':
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=30)
    elif time_range == 'Last 1 hour':
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
    else:
        start = start_time
        end = end_time
    
    cw = boto3.client('cloudwatch', region_name=region)
    
    response = cw.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'e1',
                'Expression': f'SEARCH(\'{{/aws/sagemaker/InferenceComponents,EndpointName,VariantName,InstanceId,ContainerId,InferenceComponentName,GpuId}} MetricName="GPUUtilizationNormalized" InferenceComponentName="{inference_component_name}"\', \'SampleCount\', 10)'
            },
            {
                'Id': 'e2',
                'Expression': 'SUM(e1)'
            },
            {
                'Id': 'e3',
                'Expression': f'e2 * {cost_per_hour} / 4 / 360'
            },
            {
                'Id': 'e4',
                'Expression': 'RUNNING_SUM(e3)'
            }
        ],
        StartTime=start,
        EndTime=end
    )
    
    cumulative_cost = None
    latest_cost_per_10s = None
    
    for result in response['MetricDataResults']:
        if result['Id'] == 'e4' and result['Values']:
            cumulative_cost = result['Values'][0]
        elif result['Id'] == 'e3' and result['Values']:
            latest_cost_per_10s = result['Values'][0]
    
    return {
        'inference_component': inference_component_name,
        'time_range': f"{start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')}",
        'cumulative_cost': cumulative_cost,
        'latest_cost_per_10s': latest_cost_per_10s
    }


def create_cost_widget(inference_component_name, cost_per_hour, region='us-east-1'):
    """Create interactive widget to get model cost metrics."""
    
    time_option = widgets.Dropdown(
        options=['Last 5 min', 'Last 10 min', 'Last 30 min', 'Last 1 hour', 'Custom Range'],
        value='Last 10 min',
        description='Time Range:'
    )
    
    start_picker = widgets.DatetimePicker(
        description='Start Time:',
        disabled=True,
        value=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    
    end_picker = widgets.DatetimePicker(
        description='End Time:',
        disabled=True,
        value=datetime.now(timezone.utc)
    )
    
    button = widgets.Button(description='Get Cost', button_style='success')
    output = widgets.Output()
    
    def on_time_option_change(change):
        if change['new'] == 'Custom Range':
            start_picker.disabled = False
            end_picker.disabled = False
        else:
            start_picker.disabled = True
            end_picker.disabled = True
    
    time_option.observe(on_time_option_change, names='value')
    
    def get_cost(b):
        output.clear_output()
        with output:
            if time_option.value == 'Custom Range':
                result = get_model_cost(
                    inference_component_name, 
                    cost_per_hour, 
                    region, 
                    'Custom',
                    start_picker.value,
                    end_picker.value
                )
            else:
                result = get_model_cost(
                    inference_component_name, 
                    cost_per_hour, 
                    region, 
                    time_option.value
                )
            
            print(f"Inference Component: {result['inference_component']}")
            print(f"Time Range: {result['time_range']}")
            print(f"\n{'='*50}")
            print(f"Cumulative Cost: ${result['cumulative_cost']:.4f}" if result['cumulative_cost'] else "Cumulative Cost: No data")
            print(f"Latest Cost per 10s: ${result['latest_cost_per_10s']:.6f}" if result['latest_cost_per_10s'] else "Latest Cost per 10s: No data")
    
    button.on_click(get_cost)
    
    display(time_option, start_picker, end_picker, button, output)


def create_dashboard(dashboard_name, endpoint_name, inference_components, cost_per_hour, region='us-east-1'):
    """
    Create CloudWatch dashboard for endpoint monitoring.
    
    Args:
        dashboard_name: Name for the dashboard
        endpoint_name: SageMaker endpoint name
        inference_components: List of dicts with 'name' and 'label' keys
                             e.g., [{'name': 'IC-xxx-model-a', 'label': 'MODEL_A'}, ...]
        cost_per_hour: Cost per hour for GPU (assumes same for all models)
        region: AWS region
    
    Returns:
        Dashboard ARN
    """
    import json
    
    cw = boto3.client('cloudwatch', region_name=region)
    
    widgets = []
    
    # Widget 1: Cluster Details (always at position 0,0)
    cluster_widget = {
        "type": "metric",
        "x": 0,
        "y": 0,
        "width": 6,
        "height": 8,
        "properties": {
            "metrics": [
                [ { "expression": f"SUM(SEARCH('{{/aws/sagemaker/InferenceComponents,EndpointName,VariantName,InstanceId,ContainerId,InferenceComponentName,GpuId}} MetricName=\"GPUUtilizationNormalized\" EndpointName=\"{endpoint_name}\"', 'SampleCount', 10))", "label": "used-gpus", "id": "e1", "period": 10, "region": region } ],
                [ { "expression": "m2*4 -e1", "label": "free-gpus", "id": "e2", "region": region } ],
                [ "/aws/sagemaker/Endpoints", "CPUUtilizationNormalized", "EndpointName", endpoint_name, "VariantName", "AllTraffic", { "id": "m2", "label": "number-of-instances", "region": region } ]
            ],
            "view": "timeSeries",
            "stacked": False,
            "region": region,
            "stat": "SampleCount",
            "period": 10,
            "title": "Cluster Details"
        }
    }
    widgets.append(cluster_widget)
    
    # Widget 2+: Cost widgets for each inference component
    for idx, ic in enumerate(inference_components):
        ic_name = ic['name']
        ic_label = ic.get('label', ic_name)
        
        # Calculate position (6 units to the right for first, then stack vertically)
        x_pos = 6 if idx == 0 else 6
        y_pos = idx * 8
        
        cost_widget = {
            "type": "metric",
            "x": x_pos,
            "y": y_pos,
            "width": 7,
            "height": 8,
            "properties": {
                "metrics": [
                    [ { "expression": f"SEARCH('{{/aws/sagemaker/InferenceComponents,EndpointName,VariantName,InstanceId,ContainerId,InferenceComponentName,GpuId}} MetricName=\"GPUUtilizationNormalized\" InferenceComponentName=\"{ic_name}\"', 'SampleCount', 10)", "label": "Get GPUS being used", "id": "e1", "visible": False } ],
                    [ { "expression": "SUM(e1)", "label": "Number of GPUs", "id": "e2", "visible": False } ],
                    [ { "expression": f"e2 * {cost_per_hour} / 4 / 360", "label": "Cost per 10s", "id": "e3" } ],
                    [ { "expression": "RUNNING_SUM(e3)", "label": "$ cost", "id": "e4" } ]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": region,
                "stat": "SampleCount",
                "period": 10,
                "title": f"$ cost for {ic_label} {ic_name}"
            }
        }
        widgets.append(cost_widget)
    
    dashboard_body = {
        "widgets": widgets
    }
    
    response = cw.put_dashboard(
        DashboardName=dashboard_name,
        DashboardBody=json.dumps(dashboard_body)
    )
    
    print(f"Dashboard created: {dashboard_name}")
    print(f"URL: https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={dashboard_name}")
    
    return response['DashboardValidationMessages']


def get_metric_data(namespace, metric_name, dimensions, stat='Average', period=60, hours=1):
    """Query CloudWatch metrics."""
    from datetime import timezone
    now = datetime.now(timezone.utc)
    response = cloudwatch.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{'Name': k, 'Value': v} for k, v in dimensions.items()],
        StartTime=now - timedelta(hours=hours),
        EndTime=now,
        Period=period,
        Statistics=[stat]
    )
    return sorted(response['Datapoints'], key=lambda x: x['Timestamp'])


def get_current_model_copies(inference_component_name):
    """Get the current number of model copies from inference component."""
    try:
        response = sm_client.describe_inference_component(
            InferenceComponentName=inference_component_name
        )
        
        # Get copy count from runtime config
        runtime_config = response.get('RuntimeConfig', {})
        current_copy_count = runtime_config.get('CurrentCopyCount', 0)
        desired_copy_count = runtime_config.get('DesiredCopyCount', 0)
        
        # Get GPU allocation per copy
        spec = response.get('Specification', {})
        compute_reqs = spec.get('ComputeResourceRequirements', {})
        gpus_per_copy = compute_reqs.get('NumberOfAcceleratorDevicesRequired', 1)
        
        return {
            'current_copy_count': current_copy_count,
            'desired_copy_count': desired_copy_count,
            'gpus_per_copy': gpus_per_copy,
            'status': response.get('InferenceComponentStatus', 'Unknown')
        }
    except Exception as e:
        print(f"Error getting model copies: {e}")
        return None


def calculate_cost_per_1k(inference_component_name, endpoint_name, gpu_count, hourly_cost, hours=1):
    """Calculate cost per 1K invocations, broken down by model copy."""
    
    # Get total invocations for the inference component
    invocations = get_metric_data(
        'AWS/SageMaker', 'InvocationsPerCopy',
        {'InferenceComponentName': inference_component_name},
        stat='Sum', period=3600, hours=hours
    )
    total = sum(d['Sum'] for d in invocations) if invocations else 0

    # Get model copy information for per-copy breakdown
    copy_info = get_current_model_copies(inference_component_name)
        
    # Calculate what fraction of the instance this IC is using
    total_gpus_used = copy_info['current_copy_count'] * copy_info['gpus_per_copy']
    gpu_fraction = total_gpus_used / gpu_count  # Fraction of instance GPUs used by this IC
    
    # Calculate total cost (proportional to GPU usage)
    cost = hourly_cost * gpu_fraction * hours
    cost_per_1k = (cost / (total / 1000)) if total > 0 else 0
    
    results = {
        'invocations': total,
        'total_cost': cost,
        'cost_per_1k': cost_per_1k,
        'hours': hours,
        'gpu_fraction': gpu_fraction
    }
        
    if copy_info and copy_info['current_copy_count'] > 0:
        # Calculate per-copy metrics
        invocations_per_copy = total
        cost_per_copy = cost / copy_info['current_copy_count']
        
        results.update({
            'model_copies': copy_info['current_copy_count'],
            'gpus_per_copy': copy_info['gpus_per_copy'],
            'total_gpus_used': total_gpus_used,
            'invocations_per_copy': invocations_per_copy,
            'cost_per_copy': cost_per_copy,
            'cost_per_1k_per_copy': (cost_per_copy / (invocations_per_copy / 1000)) if invocations_per_copy > 0 else 0
        })
    
    return results
    

def analyze_routing_detailed(inference_component_name, hours=1):
    """Analyze traffic distribution and identify hot copies."""
    
    # Check what dimensions are available for Invocations
    invocation_metrics = cloudwatch.list_metrics(
        Namespace='AWS/SageMaker',
        MetricName='Invocations',
        Dimensions=[
            {'Name': 'InferenceComponentName', 'Value': inference_component_name}
        ]
    )
    
    # Get invocations per container (each container = one copy)
    container_invocations = []
    for metric_info in invocation_metrics['Metrics']:
        dims_dict = {d['Name']: d['Value'] for d in metric_info['Dimensions']}
        
        # Skip if it doesn't have ContainerId (aggregated metric)
        if 'ContainerId' not in dims_dict:
            continue
        
        # Query each unique dimension combination
        data = get_metric_data(
            'AWS/SageMaker',
            'Invocations',
            dims_dict,
            stat='Sum',
            period=60,
            hours=hours
        )
        
        if data:
            total = sum(d['Sum'] for d in data)
            container_invocations.append({
                'container_id': dims_dict.get('ContainerId', 'N/A')[:12],
                'instance_id': dims_dict.get('InstanceId', 'N/A')[-8:],
                'gpu_id': dims_dict.get('GpuId', 'N/A'),
                'total': total,
                'avg': total / len(data) if data else 0
            })
    
    # Sort by total invocations (hottest first)
    container_invocations.sort(key=lambda x: x['total'], reverse=True)
    
    # TOTAL INVOCATIONS (aggregated)
    total_data = get_metric_data(
        'AWS/SageMaker', 
        'Invocations',
        {'InferenceComponentName': inference_component_name},
        stat='Sum', 
        period=60, 
        hours=hours
    )
    
    results = {}
    
    if total_data:
        total_values = [d['Sum'] for d in total_data]
        results['total_invocations'] = {
            'sum': sum(total_values),
            'avg_per_period': sum(total_values) / len(total_values),
            'max_per_period': max(total_values)
        }
    
    if container_invocations:
        results['container_invocations'] = container_invocations
        
        # Calculate imbalance
        totals = [c['total'] for c in container_invocations]
        if totals and min(totals) > 0:
            results['imbalance'] = max(totals) / min(totals)
        else:
            results['imbalance'] = float('inf')
    
    # Get model copy info
    copy_info = get_current_model_copies(inference_component_name)
    if copy_info:
        results['model_copies'] = copy_info['current_copy_count']
        results['gpus_per_copy'] = copy_info['gpus_per_copy']
    
    return results


def analyze_utilization(endpoint_name, inference_component_name, gpu_id, hours=1):
    """Analyze resource utilization and performance metrics.
    
    GPU metrics are per-GPU, CPU/Memory are instance-level, errors/latency are component-level."""
    
    metrics = {}
    
    # Ensure gpu_id is a string
    if isinstance(gpu_id, int):
        gpu_id_str = f"gpu_{gpu_id}"
    else:
        gpu_id_str = gpu_id
    
    # GPU metrics - PER GPU (with GpuId dimension)
    for metric in ['GPUUtilizationNormalized', 'GPUMemoryUtilizationNormalized']:
        data = get_metric_data(
            '/aws/sagemaker/InferenceComponents', 
            metric,
            {
                'InferenceComponentName': inference_component_name,
                'GpuId': gpu_id_str,
            },
            stat='Average', 
            period=300, 
            hours=hours
        )
        if data:
            values = [d['Average'] for d in data]
            metrics[metric] = {
                'avg': sum(values)/len(values),
                'max': max(values),
                'min': min(values)
            }
    
    # CPU and Memory - INSTANCE LEVEL (no GpuId dimension)
    for metric in ['CPUUtilizationNormalized', 'MemoryUtilizationNormalized']:
        data = get_metric_data(
            '/aws/sagemaker/InferenceComponents', 
            metric,
            {
                'InferenceComponentName': inference_component_name,
            },
            stat='Average', 
            period=300, 
            hours=hours
        )
        if data:
            values = [d['Average'] for d in data]
            metrics[metric] = {
                'avg': sum(values)/len(values),
                'max': max(values),
                'min': min(values)
            }
    
    # Error rates - COMPONENT LEVEL (no GpuId dimension)
    for metric in ['Invocation4XXErrors', 'Invocation5XXErrors', 'InvocationModelErrors']:
        data = get_metric_data(
            'AWS/SageMaker', 
            metric,
            {'InferenceComponentName': inference_component_name},
            stat='Sum', 
            period=300, 
            hours=hours
        )
        if data:
            metrics[metric] = {'total': sum(d['Sum'] for d in data)}
    
    # Latency metrics - COMPONENT LEVEL (no GpuId dimension)
    for metric in ['ModelLatency', 'OverheadLatency']:
        data = get_metric_data(
            'AWS/SageMaker', 
            metric,
            {'InferenceComponentName': inference_component_name},
            stat='Average', 
            period=300, 
            hours=hours
        )
        if data:
            values = [d['Average']/1000 for d in data]
            metrics[metric] = {
                'avg': sum(values)/len(values),
                'p50': sorted(values)[len(values)//2] if values else 0,
                'max': max(values)
            }
    
    return metrics


def get_all_gpu_metrics_detailed(inference_component_name, hours=1):
    """Get GPU metrics with full granularity: endpoint, variant, instance, container, GPU."""
    
    # Get all GPU metric combinations for this inference component
    all_metrics = []
    
    for metric_name in ['GPUUtilizationNormalized', 'GPUMemoryUtilizationNormalized']:
        response = cloudwatch.list_metrics(
            Namespace='/aws/sagemaker/InferenceComponents',
            MetricName=metric_name,
            Dimensions=[
                {'Name': 'InferenceComponentName', 'Value': inference_component_name}
            ]
        )
        
        for metric_info in response['Metrics']:
            dims_dict = {d['Name']: d['Value'] for d in metric_info['Dimensions']}
            
            # Only process metrics that have GpuId (skip aggregated ones)
            if 'GpuId' in dims_dict:
                # Query this specific metric
                data = get_metric_data(
                    '/aws/sagemaker/InferenceComponents',
                    metric_name,
                    dims_dict,
                    stat='Average',
                    period=300,
                    hours=hours
                )
                
                if data:
                    values = [d['Average'] for d in data]
                    
                    # Find or create entry for this combination
                    key = (
                        dims_dict.get('EndpointName', 'N/A'),
                        dims_dict.get('VariantName', 'N/A'),
                        dims_dict.get('InstanceId', 'N/A'),
                        dims_dict.get('ContainerId', 'N/A'),
                        dims_dict.get('GpuId', 'N/A')
                    )
                    
                    # Find existing entry or create new
                    existing = next((m for m in all_metrics if m['key'] == key), None)
                    if not existing:
                        existing = {
                            'key': key,
                            'endpoint': dims_dict.get('EndpointName', 'N/A'),
                            'variant': dims_dict.get('VariantName', 'N/A'),
                            'instance_id': dims_dict.get('InstanceId', 'N/A'),
                            'container_id': dims_dict.get('ContainerId', 'N/A')[:12],  # Shortened
                            'gpu_id': dims_dict.get('GpuId', 'N/A'),
                            'metrics': {}
                        }
                        all_metrics.append(existing)
                    
                    existing['metrics'][metric_name] = {
                        'avg': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values)
                    }
    
    return all_metrics


def analyze_utilization_detailed(inference_component_name, hours=1):
    """Analyze resource utilization with full granularity."""
    
    # Get detailed GPU metrics
    gpu_metrics = get_all_gpu_metrics_detailed(inference_component_name, hours)
    
    # Get component-level metrics (CPU, Memory, Errors, Latency)
    component_metrics = {}
    
    # CPU and Memory - INSTANCE LEVEL
    for metric in ['CPUUtilizationNormalized', 'MemoryUtilizationNormalized']:
        data = get_metric_data(
            '/aws/sagemaker/InferenceComponents',
            metric,
            {'InferenceComponentName': inference_component_name},
            stat='Average',
            period=300,
            hours=hours
        )
        if data:
            values = [d['Average'] for d in data]
            component_metrics[metric] = {
                'avg': sum(values) / len(values),
                'max': max(values),
                'min': min(values)
            }
    
    # Error rates
    for metric in ['Invocation4XXErrors', 'Invocation5XXErrors', 'InvocationModelErrors']:
        data = get_metric_data(
            'AWS/SageMaker',
            metric,
            {'InferenceComponentName': inference_component_name},
            stat='Sum',
            period=300,
            hours=hours
        )
        if data:
            component_metrics[metric] = {'total': sum(d['Sum'] for d in data)}
    
    # Latency metrics
    for metric in ['ModelLatency', 'OverheadLatency']:
        data = get_metric_data(
            'AWS/SageMaker',
            metric,
            {'InferenceComponentName': inference_component_name},
            stat='Average',
            period=300,
            hours=hours
        )
        if data:
            values = [d['Average'] / 1000 for d in data]
            component_metrics[metric] = {
                'avg': sum(values) / len(values),
                'p50': sorted(values)[len(values) // 2] if values else 0,
                'max': max(values)
            }
    
    return {
        'gpu_metrics': gpu_metrics,
        'component_metrics': component_metrics
    }


def cleanup_endpoint(endpoint_name, ic_names, endpoint_config_name, model_names):
    """Clean up all resources for an endpoint with multiple inference components."""
    try:
        print(f"\n{'='*60}")
        print(f"Cleaning up: {endpoint_name}")
        print(f"{'='*60}")
        
        # Step 1: Delete all inference components
        print(f"[1/4] Deleting inference components...")
        for ic_name in ic_names:
            try:
                sm_client.delete_inference_component(InferenceComponentName=ic_name)
                sm_client.get_waiter('inference_component_deleted').wait(InferenceComponentName=ic_name)
                print(f"  ✓ {ic_name} deletion initiated")
            except Exception as e:
                print(f"  ⚠️  Error deleting {ic_name}: {e}")
                
        # Step 2: Delete endpoint
        print(f"[2/4] Deleting endpoint: {endpoint_name}...")
        try:
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            sm_client.get_waiter('endpoint_deleted').wait(EndpointName=endpoint_name)
            print(f"  ✓ Endpoint deletion initiated")
        except Exception as e:
            print(f"  ⚠️  Error deleting endpoint: {e}")
        
        # Step 3: Delete endpoint config
        print(f"[3/4] Deleting endpoint config: {endpoint_config_name}...")
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            print(f"  ✓ Endpoint config deleted")
        except Exception as e:
            print(f"  ⚠️  Error deleting endpoint config: {e}")
        
        # Step 4: Delete all models
        print(f"[4/4] Deleting models...")
        for model_name in model_names:
            try:
                sm_client.delete_model(ModelName=model_name)
                print(f"  ✓ {model_name} deleted")
            except Exception as e:
                print(f"  ⚠️  Error deleting {model_name}: {e}")
        
        print(f"\n✅ Cleanup complete for {endpoint_name}")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed for {endpoint_name}: {e}")
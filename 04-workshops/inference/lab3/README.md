# Deploy Multiple SOTA LLMs on a single endpoint with Scale-to-Zero

In this notebook, you'll learn how to deploy multiple state-of-the-art foundation models on a single SageMaker endpoint with cost optimization through autoscaling and scale-to-zero capabilities.

We'll start with model optimization using Fast Model Loader, then configure autoscaling policies, and finish with traffic testing to see the scaling behavior in action.

### 1. Model Optimization with Fast Model Loader

In this section, we will:
- Set up **Llama 3.1 8B Instruct** using SageMaker ModelBuilder
- Configure **Fast Model Loader** for streaming model weights from S3 to GPU for faster deployment
- Deploy the model on an **Inference Component-based endpoint**

### 2. Configure Autoscaling with Scale-to-Zero

In this section, we will implement autoscaling:
- Set up **Target Tracking policies** for 1→N scaling 
- Configure **Step Scaling policies** for 0→1 scaling

### 3. Test Scaling Behavior with Live Traffic

In this section, we will validate our scaling setup:
- Verify **scale-to-zero behavior** after no traffic
- Generate **concurrent request load** to trigger autoscaling
- Monitor **inference component scaling** in real-time

### 4. (Optional) Deploy Multiple Models on Same Endpoint

In this optional section, we will:
- Deploy **Mistral 7B Instruct** as a second model on the existing endpoint
- Configure **independent autoscaling** for each model

### 5. Clean Up

- Delete resources created during the workshop

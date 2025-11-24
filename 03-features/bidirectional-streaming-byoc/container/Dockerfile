FROM python:3.9-slim

# Set a docker label to advertise bidirectional streaming support on the container
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port 8080
EXPOSE 8080

# Set entrypoint and default command
ENTRYPOINT ["python", "app.py"]
CMD ["serve"]
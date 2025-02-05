# Steps to install streamlit app

Install Python, pip, and any other required libraries:
sudo apt-get update
sudo apt-get install python3 python3-pip
pip3 install streamlit boto3 # and any other required packages


On Ec2 running Amazon linux
sudo yum update -y
# Install Python 3 (if not already installed):
sudo yum install python3 -y
# Install pip (if not already installed):
sudo yum install python3-pip -y
# Install required packages:
pip3 install --user streamlit boto3 #

# Command To Start the streamlit app 
streamlit run main.py
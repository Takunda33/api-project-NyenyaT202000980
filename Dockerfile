# Use the official Python 3.9 base image
FROM python:3.9

# Set environment variable to prevent Python from buffering output
ENV PYTHONUNBUFFERED 1

# Create a directory for your project
RUN mkdir /approject

# Set the working directory to /appproject
WORKDIR /approject

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the project files into the container
COPY ./approject /approject

# Base image
FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building Poppler and installing other packages
RUN apt-get update && \
    apt-get install -y \
    poppler-utils \
    git \
    build-essential \
    g++  \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirementsv2.txt .

# Install dependencies
# RUN pip install --no-cache-dir -r requirementsv2.txt
RUN pip install  -r requirementsv2.txt


# RUN pip install --no-cache-dir -e git+https://github.com/facebookresearch/detectron2.git@31bebdea147c96f8a00a0d55931858bf727ae370#egg=detectron2
RUN pip install -e git+https://github.com/facebookresearch/detectron2.git@31bebdea147c96f8a00a0d55931858bf727ae370#egg=detectron2

# Copy the current directory contents into the container at /app
COPY . .
COPY ./.env /app/
# RUN ls -al  

# Expose port 8000 to the outside world
EXPOSE 12001

# Run the Gunicorn server
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:12001", "app.app:app"]

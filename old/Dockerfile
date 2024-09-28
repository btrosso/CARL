# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /CARL/

# Copy the requirements.txt file into the container at /CARL/
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Set the command to run your application
CMD ["python", "Core.py"]

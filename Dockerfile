# Base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Install project dependencies
RUN pip install -r requirements.txt

# Set the command to run the Python file
CMD ["python", "watch_next.py"]
# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies first (for better caching)
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

# Copy the application code
COPY ./app /app  

# Expose port 8080
EXPOSE 8080

# Set the default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
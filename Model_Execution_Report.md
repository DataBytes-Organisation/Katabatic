Model Execution Report
Docker Setup Errors
Dockerfile Errors
Incorrect Entry Point:
Issue: The Dockerfile initially specified app.py as the entry point, which did not match the actual script name.
Correction: Verified and corrected the entry point to main.py, which is the actual script used for execution.
Updated Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]  # Updated entry point
docker-compose.yml Errors
Volume Mapping Conflicts:

Issue: The initial volume mapping - .:/app led to conflicts by overwriting files during the build process.
Correction: Adjusted volume mapping to avoid overwriting files and to ensure consistency. Removed volume mapping for production and added persistent storage for PostgreSQL data.
Updated docker-compose.yml:
version: '3'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - db
  db:
    image: postgres:13
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydatabase
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:

Database Initialization:

Issue: Missing steps to initialize the database schema or seed initial data.
Correction: Added a database initialization script to set up the schema and seed data as needed.
Model Execution
Executed Model
Model Executed: GANBLR
Context: The GANBLR model was executed inside the Docker container.
Fit in Docker: The model fit well within the Docker container after resolving dependency issues. The Dockerfile and requirements.txt were updated to accommodate the model’s requirements.
Procedure
Build Docker Images:

Ran docker-compose build to build Docker images based on the updated Dockerfile and docker-compose.yml.
Run Docker Containers:

Used docker-compose up to start the containers, ensuring both application and database services were running correctly.
Execute Models:

Inside the running container, accessed the shell using docker exec -it <container_id> /bin/sh and executed the GANBLR model by running the relevant Python scripts.
Errors or Issues Observed
Dependency Issues:

Encountered issues related to missing or incompatible dependencies.
Resolution: Adjusted requirements.txt to include the necessary dependencies for GANBLR.
Model Execution Errors:

Noted errors related to environment variables or paths.
Resolution: Corrected these issues by updating configuration files and Docker settings to ensure proper execution.
Changes to requirements.txt
Old vs. New requirements.txt
Old requirements.txt:
pyitlib
pgmpy
scikit-learn


new requirements.txt:
pyitlib
pgmpy
scikit-learn
tensorflow
sdv

Docker Dependencies and Configurations
Dockerfile Changes:

Entry Point: Corrected from app.py to main.py.
Dependency Installation: Improved by copying requirements.txt first and then installing dependencies.
docker-compose.yml Changes:

Volume Mapping: Adjusted to prevent overwriting files and to include a volume for PostgreSQL data persistence.
Database Configuration: Added persistent storage for the PostgreSQL database.
Evidence of Changes:

Dockerfile:
- CMD ["python", "app.py"]
+ CMD ["python", "main.py"]


docker-compose.yml:
- volumes:
-   - .:/app
+ volumes:
+   - pgdata:/var/lib/postgresql/data


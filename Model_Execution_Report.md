# Model Execution Steps
Executed Model
Model Executed: GANBLR
Context: The GANBLR model was executed inside the Docker container.
Fit in Docker: The model fit well within the Docker container after resolving dependency issues. The Dockerfile and requirements.txt were updated to accommodate the model’s requirements.
# Procedure
Build Docker Images:

Ran docker-compose build to build Docker images based on the updated Dockerfile and docker-compose.yml.
Run Docker Containers:

Used docker-compose up to start the containers, ensuring both application and database services were running correctly.
Execute Models:

Inside the running container, accessed the shell using docker exec -it <container_id> /bin/sh and executed the GANBLR model by running the relevant Python scripts.

Errors or Issues Observed during the implementation
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


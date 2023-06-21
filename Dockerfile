# Base Image to use
FROM python:3.10.8

RUN apt update
RUN apt install git

#Expose port 8080
EXPOSE 8080

# Copy Requirements.txt file into app directory
COPY requirements.txt diver_app/requirements.txt

# Install all requirements in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r diver_app/requirements.txt

#Copy all files in current directory into app directory
COPY . /diver_app

#Change Working Directory to app directory
WORKDIR /diver_app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
#Dockerfile for the server side

#import the base image from DockerHub ([image]:[version])
FROM python:latest

#import the files to the image (ADD [filename] [path/to/folder])
ADD ShanghAI.py /server/
ADD bottle.py /server/


#set the working directory (the place where all commands will be executed) ([path/to/workdir])
WORKDIR /server/

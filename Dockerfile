FROM centos:7
LABEL maintainer="test666"
COPY . .
RUN yum -y update
RUN yum -y install epel-release
RUN yum install -y python36 python36-libs python36-devel
RUN python3.6 -m pip install pip --upgrade
RUN pip install -r requirements.txt
ENV this_host="localhost"
ENV this_port=8001
EXPOSE 8001
CMD ["python3", "./app/app.py"]

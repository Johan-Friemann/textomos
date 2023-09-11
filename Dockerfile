FROM opensuse/leap:latest

RUN zypper refresh
RUN zypper update -y

CMD nvidia-smi
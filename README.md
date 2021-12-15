# AWS_Graviton_Performance
Check AWS Gravition (2 and 3) Instance Performance compared by AWS Intel Instance
## Building TensorFlow Docker Image for Graviton2 in AWS ECR
### Install docker & git
```
sudo yum update -y
sudo yum install git docker docker-registry -y
```
### Start docker service
```
systemctl enable docker.service
systemctl start docker.service
systemctl status docker.service
```
### Configure ubuntu environment
```
sudo apt update
sudo apt install python3-pip
```
### Clone to build tensorflow
```
git clone https://github.com/ARM-software/Tool-Solutions.git
cd Tool-Solutions/docker/tensorflow-aarch64
```
### Build tensorflow
```
./build.sh  --onednn armpl --build-type tensorflow --jobs 16 --bazel_memory_limit 10485760
```
- It takes about 2~3 hours

### Set AWS Configure to push AWS ECR
```
aws configure
```
### Login AWS ECR
- before this job, you should make your ECR in your aws account
```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
```
### Docker push
```
export ACCOUNT_ID = <your account id>
docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/<your image name>
```
### Pull the AWS ECR docker image on AWS Graviton Instance
```
docker pull 741926482963.dkr.ecr.us-west-2.amazonaws.com/<your image name>
```



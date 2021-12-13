# AWS_Graviton_Performance
Check AWS Gravition (2 and 3) Instance Performance compared by AWS Intel Instance


### Configure ubuntu environment
```
sudo apt update
sudo apt install python3-pip
```

### Install python packages
```
pip3 install -r requirements.txt
```

### Set AWS Configure to use S3 bucket
```
aws configure
```

# Building TensorFlow on Graviton2

Install bazel build system:
```
# find the latest release of bazel for arm64 https://github.com/bazelbuild/bazel/releases/latest
# using older bazel versions may not be supported to build tensorflow
REL=4.2.1  # this is the latest stable release as of this writing
wget https://github.com/bazelbuild/bazel/releases/download/${REL}/bazel-${REL}-linux-arm64
chmod +x bazel-${REL}-linux-arm64
sudo mv bazel-${REL}-linux-arm64 /usr/local/bin
sudo ln -s /usr/local/bin/bazel-${REL}-linux-arm64 /usr/local/bin/bazel
```

Build TensorFlow on Graviton2 with Ubuntu 20.04:
```
sudo apt install build-essential python python3-pip
sudo pip3 install numpy keras_preprocessing
git clone https://github.com/tensorflow/tensorflow $HOME/tensorflow
cd $HOME/tensorflow
./configure
bazel build --config=opt --copt=-O3 --copt=-march=armv8.2-a+fp16+rcpc+dotprod+crypto --copt=-flax-vector-conversions //tensorflow/tools/pip_package:build_pip_package
```

Run an inference task:
```
cd $HOME/tensorflow/tensorflow/examples/label_image/data
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
tar xf inception_v3_2016_08_28_frozen.pb.tar.gz
cd $HOME/tensorflow
bazel build --config=opt --copt=-O3 --copt=-march=armv8.2-a+fp16+rcpc+dotprod+crypto --copt=-flax-vector-conversions tensorflow/examples/label_image/...
bazel-bin/tensorflow/examples/label_image/label_image
```

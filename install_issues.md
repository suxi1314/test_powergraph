按照官方指南https://github.com/jegonzal/PowerGraph 在ubuntu系统中安装遇到两个问题
# 1. deps/download-*.cmake中的url或md5无效，修改这些文件中的url如下:

## download-libbz2.cmake
http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz

## download-boost.cmake
https://sourceforge.net/projects/boost/files/boost/1.53.0/boost_1_53_0.tar.gz/download

## download-libtcmalloc.cmake
https://github.com/gperftools/gperftools/archive/gperftools-2.0.tar.gz  
MD5='2b412c4c8cf20b226bfc1d062ad25c7c'

## download-zookeeper.cmake
https://archive.apache.org/dist/zookeeper/zookeeper-3.5.1-alpha/zookeeper-3.5.1-alpha.tar.gz

## download-libevent.cmake
https://github.com/downloads/libevent/libevent/libevent-2.0.18-stable.tar.gz

## download-hadoop.cmake
https://archive.apache.org/dist/hadoop/core/hadoop-1.0.1/hadoop-1.0.1.tar.gz

## download-opencv.cmake
http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip/download

## download-eigen.cmake
http://bitbucket.org/eigen/eigen/get/3.1.2.tar.bz2

# 2. make时
## 报错
src/zookeeper.c:2436:5: error: null argument where non-null required (argument 1) [-Werror=nonnull]
     fprintf(LOGSTREAM,"Completion queue: ");
     ^~~~~~~
## 改为将log重定向到stderr
     fprintf(stderr,"Completion queue: ");

## Noya

### Environment

```sh
Key Library: Seal 3.5.6 (you can download it from  https://github.com/microsoft/SEAL/releases/tag/v3.6.5)
Build Tool : cmake
Complier : clang/g++ 
OS : MacOS/Linux 
```



>Note: Before you start the project, you need to combine the file **resources/CIFAR-3x32x32-test1.txt** and **resources/CIFAR-3x32x32-test2.txt** together into **resources/CIFAR-3x32x32-test.txt**



### Generate Models' Executable File 

This command will automatically compile and generate models to the **build/release** directory which include CryptoNets, LoLa and Noya-x.  

```sh
sh run.sh  
```

The name of executable file follows the format: \<Dataset\>\<Model\>\[Version\]

Among them, CifarNoyaTest is a test model like Noya-1 running on plaintext. 

You can run all the models through this way:

```sh
cd ./build/release
./XXXX（可执行文件）
```


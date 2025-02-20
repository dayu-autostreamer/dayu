## 
1. 在x86机器上交叉编译make build WHAT=model-switch-detection
2. 如果成功，到114.212.87.136:8080 查看
3. copy模型文件到服务器/data/dwy...和边缘端（https://box.nju.edu.cn/d/0dcaabb5362c4dfc8008/）， 修改代码的调用模型部分get Context，模拟car detection
4. 执行 readme里的命令，然后前端启动
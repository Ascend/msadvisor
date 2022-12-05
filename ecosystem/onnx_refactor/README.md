# 图优化知识库使用说明

1. 安装部署CANN

   下载链接：<https://www.hiascend.com/software/cann>

   安装指令参考：./Ascend-cann-toolkit*.run --full

2. 配置环境变量

   执行指令：source {install_path}/ascend-toolkit/set_env.sh

   {install_path}是CANN包安装路径

3. 准备onnx文件

   模型下载链接：<https://www.hiascend.com/software/modelzoo>

4. 安装部署onnx图优化知识库

   执行指令：cd {path}/auto-optimizer/; pip3 install .

   {path}是msadvisor代码仓根路径

5. 执行指令完成模型图优化

   执行指令：msadvisor -c {path}/ecosystem/onnx_refactor/ecosystem.json -d {datapath}

   {path}是msadvisor代码仓根路径，ecosystem.json是知识库配置；

   {datapath}是onnx所在路径；

   执行完成后，在{datapath}/out下会生成优化后的onnx文件，如果没有优化，则不会生成新的onnx文件。

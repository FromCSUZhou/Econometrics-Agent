
# Chat-Interpreter 安装指南

欢迎使用 Chat-Interpreter！本指南将帮助你顺利安装并运行项目。

## 安装步骤

### 1. 克隆项目仓库

首先，克隆 Chat-Interpreter 项目到你的本地机器：

```bash
git clone https://github.com/FromCSUZhou/ChatInterpreter
```

### 2. 进入项目目录

进入项目根目录：

```bash
cd ChatInterpreter
```

### 3. 创建并激活 Conda 环境

创建一个新的 Conda 环境并激活它：

```bash
conda create -n ChatInterpreter python=3.11
conda activate ChatInterpreter
```

### 4. 安装项目依赖

在激活的环境中，安装项目的依赖：

```bash
pip install -e .
pip install -r requirements.txt -U
```

### 5. 进入 ML_Assistant 目录并安装依赖

进入 `ML_Assistant` 目录并安装相关依赖：

```bash
cd ML_Assistant
pip install -e .
pip install -r requirements.txt
```

### 6. 进入 web 目录并安装前端依赖

返回项目根目录，进入 `web` 目录并安装前端依赖：

```bash
cd ../web
npm install
```

### 7. 构建前端项目

构建前端项目，生成编译输出文件：

```bash
npm run build
```

构建完成后，`web` 目录下会生成一个 `build` 文件夹，其中包含了前端编译输出文件。

### 8. 启动应用

最后，运行启动脚本来启动应用：

```bash
bash start.sh
```

### 9. 访问应用

现在，你的应用正在运行，可以通过以下地址访问：

[http://0.0.0.0:8080](http://0.0.0.0:8080)

Enjoy! 😄

## 常见问题

如果在安装过程中遇到任何问题，请检查以下几点：

- 确保你的网络连接正常，能够访问 GitHub 和 npm 仓库。
- 确保你已经安装了 Conda 和 Node.js。
- 如果遇到依赖安装问题，尝试使用 `--no-cache-dir` 选项重新安装依赖。
